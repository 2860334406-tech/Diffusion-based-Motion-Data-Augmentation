"""Guided DDIM sampling for the trajectory-level waist augmentation model.

This is the GTA Stage-B replacement: instead of "partial noise + amplified return",
we do "partial noise + attribute target + differentiable physical constraints".
The three constraint residuals (root yaw, hip-yaw compensation, waist pitch/roll)
are evaluated on the denoised x0 estimate each step and their gradient is added to
the noise prediction, in the spirit of OmniControl's analytic guidance.
"""

from __future__ import annotations

import math

import torch

from smp.pretrain.waist_aug_dataset import (
  COND_DIM,
  FEATURE_DIM,
  J_L_HIP_YAW,
  J_R_HIP_YAW,
  J_WAIST_PITCH,
  J_WAIST_ROLL,
  denormalise,
  normalise,
)

J_BASE = 9  # joints start at feature index 9

DEFAULT_CONSTRAINT = {
  "root_yaw_rad": math.radians(5.0),
  "hip_yaw_rad": math.radians(8.0),
  "waist_tilt_rad": math.radians(3.0),
}


# --------------------------------------------------------------------------- #
# DDIM
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ddim_step(
  scheduler,
  eps: torch.Tensor,
  x_t: torch.Tensor,
  t_cur: int,
  t_next: int,
  eta: float = 0.0,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """One deterministic/stochastic DDIM update from timestep t_cur to t_next."""
  abar = scheduler.alphas_cumprod.to(x_t.device)
  a_t = abar[t_cur]
  a_n = abar[t_next] if t_next >= 0 else torch.ones_like(a_t)
  x0 = (x_t - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
  # Clamp x0 to the training range so early steps cannot explode.
  x0 = x0.clamp(-4.0, 4.0)
  sigma = 0.0
  if eta > 0.0 and t_next >= 0:
    sigma = eta * torch.sqrt((1 - a_n) / (1 - a_t)) * torch.sqrt(1 - a_t / a_n)
  dir_coef = torch.sqrt(torch.clamp(1 - a_n - sigma**2, min=0.0))
  x_prev = torch.sqrt(a_n) * x0 + dir_coef * eps
  if sigma > 0:
    noise = torch.randn(x_t.shape, device=x_t.device, generator=generator)
    x_prev = x_prev + sigma * noise
  return x_prev


def timestep_sequence(num_timesteps: int, num_steps: int, device) -> list[int]:
  """Evenly spaced descending DDIM timesteps ending at 0."""
  if num_steps >= num_timesteps:
    return list(range(num_timesteps - 1, -1, -1))
  ts = torch.linspace(num_timesteps - 1, 0, num_steps + 1).round().long()
  return [int(v) for v in ts]


# --------------------------------------------------------------------------- #
# differentiable constraints
# --------------------------------------------------------------------------- #
def _root_yaw(x38: torch.Tensor) -> torch.Tensor:
  """Yaw of the root 6D rotation (col0), in the segment's anchor frame."""
  col0 = x38[..., 3:6]
  return torch.atan2(col0[..., 1], col0[..., 0])


def constraint_residual(
  x38: torch.Tensor,
  thresholds: dict[str, float] | None = None,
) -> torch.Tensor:
  """Scalar penalty on (B,H,38) *unnormalised* trajectories; 0 when satisfied."""
  th = thresholds or DEFAULT_CONSTRAINT
  yaw = _root_yaw(x38)
  r_root = torch.relu(yaw.abs() - th["root_yaw_rad"]).pow(2).mean()

  j = x38[..., J_BASE:]
  hip_l = j[..., J_L_HIP_YAW]
  hip_r = j[..., J_R_HIP_YAW]
  d_hip = torch.maximum(
    (hip_l - hip_l[:, :1]).abs(), (hip_r - hip_r[:, :1]).abs()
  )
  r_hip = torch.relu(d_hip - th["hip_yaw_rad"]).pow(2).mean()

  wp = j[..., J_WAIST_PITCH]
  wr = j[..., J_WAIST_ROLL]
  mp = wp.median(dim=1, keepdim=True).values
  mr = wr.median(dim=1, keepdim=True).values
  r_waist = (
    torch.relu((wp - mp).abs() - th["waist_tilt_rad"]).pow(2).mean()
    + torch.relu((wr - mr).abs() - th["waist_tilt_rad"]).pow(2).mean()
  )

  return (
    r_root / th["root_yaw_rad"] ** 2
    + r_hip / th["hip_yaw_rad"] ** 2
    + r_waist / th["waist_tilt_rad"] ** 2
  )


def constraint_report(x38: torch.Tensor) -> dict[str, float]:
  """Human-readable violations (degrees) for one batch of unnormalised windows."""
  yaw = _root_yaw(x38)
  j = x38[..., J_BASE:]
  hip = torch.maximum(
    (j[..., J_L_HIP_YAW] - j[:, :1, J_L_HIP_YAW]).abs(),
    (j[..., J_R_HIP_YAW] - j[:, :1, J_R_HIP_YAW]).abs(),
  )
  wp = j[..., J_WAIST_PITCH]
  wr = j[..., J_WAIST_ROLL]
  wp = (wp - wp.median(dim=1, keepdim=True).values).abs()
  wr = (wr - wr.median(dim=1, keepdim=True).values).abs()
  r = 180.0 / math.pi
  return {
    "root_yaw_p95_deg": float(yaw.abs().max(dim=1).values.quantile(0.95) * r),
    "hip_yaw_p95_deg": float(hip.max(dim=1).values.quantile(0.95) * r),
    "waist_pitch_p95_deg": float(wp.max(dim=1).values.quantile(0.95) * r),
    "waist_roll_p95_deg": float(wr.max(dim=1).values.quantile(0.95) * r),
  }


# --------------------------------------------------------------------------- #
# guided sampler
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _guidance_scale(step_index: int, warmup: int, w_max: float) -> float:
  if warmup <= 0:
    return w_max
  return 1.0 + (w_max - 1.0) * min(1.0, (step_index + 1) / warmup)


def sample_trajectories(
  model,
  scheduler,
  condition: torch.Tensor,
  *,
  num_steps: int = 25,
  eta: float = 0.5,
  guidance_scale: float = 2.0,
  guidance_warmup: int = 8,
  constraint_weight: float = 5.0,
  constraint_stop_frac: float = 0.3,
  q_low: torch.Tensor,
  q_high: torch.Tensor,
  window: int,
  initial_noise: torch.Tensor | None = None,
  init_clean: torch.Tensor | None = None,
  start_timestep: int = 0,
  pin_frames: int = 0,
  generator: torch.Generator | None = None,
  thresholds: dict[str, float] | None = None,
) -> torch.Tensor:
  """Return unnormalised (B, window, 38) trajectories.

  ``init_clean`` + ``start_timestep`` is GTA's partial noising: the seed trajectory
  is noised only up to ``start_timestep`` and denoised from there, which keeps its
  overall amplitude/pose while the model refines details. ``pin_frames`` additionally
  re-imposes the first frames each step for "same initial stance" pairs.
  """
  device = condition.device
  b = condition.shape[0]
  if initial_noise is not None:
    x = initial_noise.clone().to(device)
  elif init_clean is not None and start_timestep > 0:
    x0_norm = normalise(init_clean.to(device), q_low, q_high)
    noise = torch.randn(x0_norm.shape, device=device, generator=generator)
    t_mu = torch.full((x0_norm.shape[0],), start_timestep, dtype=torch.long, device=device)
    x = scheduler.add_noise(x0_norm, noise, t_mu)
  else:
    x = torch.randn(b, window, FEATURE_DIM, device=device, generator=generator)
  if init_clean is not None and pin_frames > 0:
    pin = normalise(init_clean[:, :pin_frames].to(device), q_low, q_high)

  ts = timestep_sequence(scheduler.num_timesteps, num_steps, device)
  if start_timestep > 0:
    ts = [start_timestep] + [t for t in ts if t < start_timestep]
  for i in range(len(ts) - 1):
    t_cur, t_next = ts[i], ts[i + 1]
    t = torch.full((b,), t_cur, dtype=torch.long, device=device)
    w = _guidance_scale(i, guidance_warmup, guidance_scale)

    with torch.enable_grad():
      x_req = x.detach().requires_grad_(True)
      eps_c = model(x_req, t, condition=condition)
      eps_u = model(x_req, t, condition=None)
      eps = eps_u + w * (eps_c - eps_u)
      abar = scheduler.alphas_cumprod.to(device)[t_cur]
      x0_norm = (x_req - torch.sqrt(1 - abar) * eps) / torch.sqrt(abar)
      apply_constraints = (
        constraint_weight > 0.0 and t_cur > constraint_stop_frac * scheduler.num_timesteps
      )
      if apply_constraints:
        x0_raw = denormalise(x0_norm, q_low, q_high)
        res = constraint_residual(x0_raw, thresholds)
        grad = torch.autograd.grad(res, x_req)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad = grad / (grad.flatten(1).norm(dim=1).clamp_min(1e-6)).view(-1, 1, 1)
        eps = (eps - constraint_weight * torch.sqrt(1 - abar) * grad).detach()
      else:
        eps = eps.detach()

    x = ddim_step(scheduler, eps, x, t_cur, t_next, eta=eta, generator=generator)
    if init_clean is not None and pin_frames > 0:
      x[:, :pin_frames] = pin

  return denormalise(x, q_low, q_high)


def condition_from_targets(
  *,
  waist_amp_deg: float,
  waist_speed_deg_s: float,
  sign: float,
  entry_duration_s: float,
  knee_deg: float = 18.0,
  hip_yaw_deg: float = 0.0,
  l_sh_roll_deg: float = 15.0,
  l_sh_yaw_deg: float = 0.0,
  l_sh_pitch_deg: float = 11.5,
  l_elbow_deg: float = 34.0,
  l_wr_roll_deg: float = 0.0,
  r_sh_roll_deg: float = 15.0,
  r_sh_yaw_deg: float = 0.0,
  r_sh_pitch_deg: float = 11.5,
  r_elbow_deg: float = 34.0,
  r_wr_roll_deg: float = 0.0,
  init_waist_pitch_deg: float = 0.0,
  init_waist_roll_deg: float = 0.0,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Build a (1, 18) condition from human-readable targets (degrees)."""
  from smp.pretrain.waist_aug_dataset import COND_CLIP, COND_SCALE

  raw = torch.tensor(
    [
      math.radians(waist_amp_deg), waist_speed_deg_s, sign, entry_duration_s,
      math.radians(knee_deg), math.radians(hip_yaw_deg),
      math.radians(l_sh_roll_deg), math.radians(l_sh_yaw_deg),
      math.radians(l_sh_pitch_deg), math.radians(l_elbow_deg), math.radians(l_wr_roll_deg),
      math.radians(r_sh_roll_deg), math.radians(r_sh_yaw_deg),
      math.radians(r_sh_pitch_deg), math.radians(r_elbow_deg), math.radians(r_wr_roll_deg),
      math.radians(init_waist_pitch_deg), math.radians(init_waist_roll_deg),
    ],
    dtype=torch.float32,
    device=device,
  )
  scale = torch.from_numpy(COND_SCALE).to(device)
  return (raw / scale).clamp(-COND_CLIP, COND_CLIP).unsqueeze(0)