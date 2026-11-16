"""Frozen force-estimator correction networks and feature layout."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

LOW_COLUMNS = tuple(
  index
  for start, width in ((0, 29), (290, 29), (580, 3), (610, 3), (640, 29))
  for frame in (8, 9)
  for index in range(start + frame * width, start + (frame + 1) * width)
)


class StateResidual(nn.Module):
  x_mean: torch.Tensor
  x_std: torch.Tensor
  tau_mean: torch.Tensor
  tau_std: torch.Tensor

  def __init__(
    self,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    tau_mean: torch.Tensor,
    tau_std: torch.Tensor,
    correction_cap_n: float,
    probability_floor: float,
  ) -> None:
    super().__init__()
    self.correction_cap_n = float(correction_cap_n)
    self.probability_floor = float(probability_floor)
    self.register_buffer("x_mean", x_mean.float())
    self.register_buffer("x_std", x_std.float())
    self.register_buffer("tau_mean", tau_mean.float())
    self.register_buffer("tau_std", tau_std.float())
    self.net = nn.Sequential(
      nn.Linear(1227, 128),
      nn.GELU(),
      nn.Linear(128, 64),
      nn.GELU(),
      nn.Linear(64, 3),
    )

  def forward(
    self,
    x: torch.Tensor,
    tau: torch.Tensor,
    base_force: torch.Tensor,
    base_point: torch.Tensor,
    probability: torch.Tensor,
  ) -> torch.Tensor:
    normalized_x = ((x - self.x_mean) / self.x_std).clamp(-10, 10)
    normalized_tau = ((tau - self.tau_mean) / self.tau_std).clamp(-10, 10)
    features = torch.cat(
      (
        normalized_x,
        normalized_tau.flatten(1),
        base_force / 100,
        base_point / 0.2,
        probability[:, None],
      ),
      dim=1,
    )
    correction = (
      probability.clamp(min=self.probability_floor)[:, None]
      * self.correction_cap_n
      * self.net(features).tanh()
    )
    return base_force + correction


class HighProbabilityResidual(nn.Module):
  x_mean: torch.Tensor
  x_std: torch.Tensor
  tau_mean: torch.Tensor
  tau_std: torch.Tensor
  highrate_mean: torch.Tensor
  highrate_std: torch.Tensor

  def __init__(
    self,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    tau_mean: torch.Tensor,
    tau_std: torch.Tensor,
    highrate_mean: torch.Tensor,
    highrate_std: torch.Tensor,
  ) -> None:
    super().__init__()
    self.register_buffer("x_mean", x_mean.float())
    self.register_buffer("x_std", x_std.float().clamp_min(1e-5))
    self.register_buffer("tau_mean", tau_mean.float())
    self.register_buffer("tau_std", tau_std.float().clamp_min(1e-5))
    self.register_buffer("highrate_mean", highrate_mean.float())
    self.register_buffer("highrate_std", highrate_std.float().clamp_min(1e-4))
    self.net = nn.Sequential(
      nn.Linear(430, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 3),
    )

  def forward(
    self, x: torch.Tensor, tau: torch.Tensor, base_force: torch.Tensor
  ) -> torch.Tensor:
    low_x = (
      (x[:, LOW_COLUMNS] - self.x_mean[list(LOW_COLUMNS)])
      / self.x_std[list(LOW_COLUMNS)]
    ).clamp(-8, 8)
    low_tau = ((tau[:, -2:] - self.tau_mean) / self.tau_std).flatten(1).clamp(-8, 8)
    low = torch.cat((low_x, low_tau, base_force / 100), dim=1)
    high = torch.zeros((len(x), 183), dtype=x.dtype, device=x.device)
    return base_force + 100 * self.net(torch.cat((low, high), dim=1))


class TorsoCorrection(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(252, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 3),
    )

  def forward(self, low: torch.Tensor) -> torch.Tensor:
    region = torch.zeros((len(low), 5), dtype=low.dtype, device=low.device)
    return self.net(torch.cat((low, region), dim=1))


def low_features(
  x: np.ndarray,
  tau: np.ndarray,
  base_force: np.ndarray,
  norms: dict[str, np.ndarray],
) -> np.ndarray:
  columns = list(LOW_COLUMNS)
  a = np.clip(
    (x[:, columns] - norms["x_mean"][columns]) / norms["x_std"][columns], -8, 8
  )
  b = np.clip((tau[:, -2:] - norms["tau_mean"]) / norms["tau_std"], -8, 8).reshape(
    len(x), -1
  )
  return np.concatenate((a, b, base_force / 100), axis=1).astype(np.float32)
