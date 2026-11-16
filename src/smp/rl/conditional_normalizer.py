"""Checkpointable SDS reference means per contact condition and timestep."""
from __future__ import annotations

import torch


class ConditionalDiffNormalizer:
  def __init__(self, timesteps, device, conditions=6, max_count=50_000_000):
    self.mean = torch.ones(conditions, timesteps, device=device)
    self.count = torch.zeros(conditions, timesteps, device=device, dtype=torch.long)
    self.max_count = max_count

  @torch.no_grad()
  def update_and_normalize(self, timestep, errors, condition_ids):
    number = self.mean.shape[0]
    counts = torch.bincount(condition_ids, minlength=number)
    sums = torch.zeros(number, device=errors.device).scatter_add_(0, condition_ids, errors)
    old_count = self.count[:, timestep]
    old_mean = self.mean[:, timestep]
    total = old_count + counts
    can_update = (old_count <= self.max_count) & (counts > 0)
    means = (old_mean * old_count + sums) / total.clamp_min(1)
    self.mean[:, timestep] = torch.where(can_update, means, old_mean)
    self.count[:, timestep] = torch.where(can_update, total, old_count)
    return errors / self.mean[condition_ids, timestep].clamp_min(1e-4)

  def state_dict(self):
    return {"mean": self.mean.detach().cpu(), "count": self.count.detach().cpu(),
            "max_count": self.max_count}

  def load_state_dict(self, state):
    if state["mean"].shape != self.mean.shape or state["count"].shape != self.count.shape:
      raise ValueError("Conditional normalizer dimensions changed")
    if not torch.isfinite(state["mean"]).all() or (state["count"] < 0).any():
      raise ValueError("Invalid conditional normalizer state")
    self.mean.copy_(state["mean"])
    self.count.copy_(state["count"])
    self.max_count = int(state["max_count"])
