"""Dual-latent proprioceptive force and resultant-point estimator.

The force encoder and the proprioceptive state encoder are trained jointly.
Only the proprioceptive branch is used at inference time.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

STATE_DIM = 64
ACTION_DIM = 29
TAU_DIM = 29
N_FRAMES = 10


def split_policy_observation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Split flattened SuperMimic [B,930] observations into state/action windows."""
  if x.ndim != 2 or x.shape[1] != 930:
    raise ValueError(f"Expected x with shape [B,930], got {tuple(x.shape)}")
  batch = x.shape[0]
  joint_pos = x[:, 0:290].reshape(batch, N_FRAMES, 29)
  joint_vel = x[:, 290:580].reshape(batch, N_FRAMES, 29)
  base_ang_vel = x[:, 580:610].reshape(batch, N_FRAMES, 3)
  projected_gravity = x[:, 610:640].reshape(batch, N_FRAMES, 3)
  action = x[:, 640:930].reshape(batch, N_FRAMES, ACTION_DIM)
  state = torch.cat((joint_pos, joint_vel, base_ang_vel, projected_gravity), dim=-1)
  return state, action


class GaussianEncoder(nn.Module):
  def __init__(self, input_dim: int, latent_dim: int = 32):
    super().__init__()
    self.body = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
    )
    self.mu = nn.Linear(128, latent_dim)
    self.logvar = nn.Linear(128, latent_dim)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.body(x)
    return self.mu(h), self.logvar(h).clamp(-10.0, 5.0)


class NominalDynamics(nn.Module):
  """Predict a no-contact state delta from the pre-transition state and action."""

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(STATE_DIM + ACTION_DIM, 256),
      nn.ELU(),
      nn.Linear(256, 256),
      nn.ELU(),
      nn.Linear(256, STATE_DIM),
    )

  def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat((state, action), dim=-1))


class DualVAEForceEstimator(nn.Module):
  x_mean: torch.Tensor
  x_std: torch.Tensor
  tau_mean: torch.Tensor
  tau_std: torch.Tensor

  def __init__(
    self,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    latent_dim: int = 32,
    tau_mean: torch.Tensor | None = None,
    tau_std: torch.Tensor | None = None,
  ):
    super().__init__()
    self.register_buffer("x_mean", x_mean.float())
    self.register_buffer("x_std", x_std.float().clamp_min(1e-6))
    self.use_tau = tau_mean is not None and tau_std is not None
    self.register_buffer(
      "tau_mean",
      tau_mean.float() if tau_mean is not None else torch.zeros(TAU_DIM),
    )
    self.register_buffer(
      "tau_std",
      tau_std.float().clamp_min(1e-6) if tau_std is not None else torch.ones(TAU_DIM),
    )
    self.latent_dim = latent_dim

    state_input_dim = N_FRAMES * STATE_DIM + (N_FRAMES - 1) * STATE_DIM
    state_input_dim += N_FRAMES * ACTION_DIM + STATE_DIM
    if self.use_tau:
      state_input_dim += N_FRAMES * TAU_DIM
    force_input_dim = N_FRAMES * 7 + (N_FRAMES - 1) * 6

    self.state_encoder = GaussianEncoder(state_input_dim, latent_dim)
    self.force_encoder = GaussianEncoder(force_input_dim, latent_dim)
    self.nominal_dynamics = NominalDynamics()

    self.state_projector = nn.Sequential(
      nn.Linear(latent_dim, 64), nn.ELU(), nn.Linear(64, latent_dim)
    )
    self.force_projector = nn.Sequential(
      nn.Linear(latent_dim, 64), nn.ELU(), nn.Linear(64, latent_dim)
    )
    head_input_dim = latent_dim + STATE_DIM + ACTION_DIM + STATE_DIM
    if self.use_tau:
      head_input_dim += TAU_DIM
    self.head = nn.Sequential(
      nn.Linear(head_input_dim, 256),
      nn.ELU(),
      nn.Linear(256, 256),
      nn.ELU(),
      nn.Linear(256, 7),
    )

  def state_branch(
    self,
    x: torch.Tensor,
    tau_hist: torch.Tensor | None = None,
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
  ]:
    x_norm = (x - self.x_mean) / self.x_std
    state, action = split_policy_observation(x_norm)
    state_delta = state[:, 1:] - state[:, :-1]

    # The current action is the action whose transition produced the current state.
    g_pred = self.nominal_dynamics(state[:, -2], action[:, -1])
    residual = state_delta[:, -1] - g_pred.detach()

    pieces = [
      state.flatten(1),
      state_delta.flatten(1),
      action.flatten(1),
      residual,
    ]
    tau_norm = None
    if self.use_tau:
      if tau_hist is None or tau_hist.shape[-2:] != (N_FRAMES, TAU_DIM):
        raise ValueError(f"Expected tau_hist [B,{N_FRAMES},{TAU_DIM}]")
      tau_norm = (tau_hist - self.tau_mean) / self.tau_std
      pieces.append(tau_norm.flatten(1))
    state_features = torch.cat(
      (*pieces,),
      dim=-1,
    )
    mu_s, logvar_s = self.state_encoder(state_features)
    return mu_s, logvar_s, g_pred, state_delta[:, -1], tau_norm

  def force_branch(
    self,
    force_hist: torch.Tensor,
    point_hist: torch.Tensor,
    contact_hist: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    f = force_hist / 100.0
    p = point_hist / 0.2
    c = contact_hist.float().unsqueeze(-1)
    df = f[:, 1:] - f[:, :-1]
    dp = p[:, 1:] - p[:, :-1]
    features = torch.cat(
      (f.flatten(1), p.flatten(1), c.flatten(1), df.flatten(1), dp.flatten(1)), dim=-1
    )
    return self.force_encoder(features)

  @staticmethod
  def _sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

  def forward(
    self,
    x: torch.Tensor,
    force_hist: torch.Tensor,
    point_hist: torch.Tensor,
    contact_hist: torch.Tensor,
    tau_hist: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    mu_s, logvar_s, g_pred, state_delta, tau_norm = self.state_branch(x, tau_hist)
    mu_f, logvar_f = self.force_branch(force_hist, point_hist, contact_hist)

    x_norm = (x - self.x_mean) / self.x_std
    state, action = split_policy_observation(x_norm)
    z_s = self._sample(mu_s, logvar_s) if self.training else mu_s
    residual = state_delta - g_pred.detach()
    head_pieces = [z_s, state[:, -1], action[:, -1], residual]
    if tau_norm is not None:
      head_pieces.append(tau_norm[:, -1])
    head_input = torch.cat(head_pieces, dim=-1)
    raw = self.head(head_input)

    return {
      "force_n": 100.0 * raw[:, :3],
      "point_m": 0.2 * raw[:, 3:6],
      "contact_logit": raw[:, 6],
      "mu_s": mu_s,
      "logvar_s": logvar_s,
      "mu_f": mu_f,
      "logvar_f": logvar_f,
      "g_pred": g_pred,
      "state_delta": state_delta,
      "align_s": self.state_projector(mu_s),
      "align_f": self.force_projector(mu_f),
    }

  @torch.inference_mode()
  def predict(
    self,
    x: torch.Tensor,
    tau_hist: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    self.eval()
    mu_s, logvar_s, g_pred, state_delta, tau_norm = self.state_branch(x, tau_hist)
    x_norm = (x - self.x_mean) / self.x_std
    state, action = split_policy_observation(x_norm)
    residual = state_delta - g_pred
    head_pieces = [mu_s, state[:, -1], action[:, -1], residual]
    if tau_norm is not None:
      head_pieces.append(tau_norm[:, -1])
    raw = self.head(torch.cat(head_pieces, dim=-1))
    return {
      "force_n": 100.0 * raw[:, :3],
      "point_m": 0.2 * raw[:, 3:6],
      "contact_prob": torch.sigmoid(raw[:, 6]),
      "mu_s": mu_s,
      "logvar_s": logvar_s,
    }


def load_dual_vae_force(
  path: str | Path, device: str = "cuda:0"
) -> DualVAEForceEstimator:
  saved = torch.load(Path(path), map_location="cpu", weights_only=True)
  model = DualVAEForceEstimator(
    saved["x_mean"],
    saved["x_std"],
    latent_dim=int(saved["config"]["latent_dim"]),
    tau_mean=saved.get("tau_mean"),
    tau_std=saved.get("tau_std"),
  )
  model.load_state_dict(
    saved["model"], strict=bool(saved["config"].get("use_tau", False))
  )
  return model.to(device).eval().requires_grad_(False)
