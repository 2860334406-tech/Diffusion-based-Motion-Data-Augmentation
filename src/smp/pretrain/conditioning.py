"""Versioned conditions, exact balanced sampling, and conditional DDPM."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Sampler

CONDITION_NAMES = ("stand", "left_shoulder", "right_shoulder", "chest", "back")
NULL_CONDITION = 5
CONDITION_SCHEMA = {
  "version": 1, "names": list(CONDITION_NAMES), "null_id": 5,
  "mirror_ids": [0, 2, 1, 3, 4],
  "meaning": "motion-response labels; LAFAN has no measured contact annotations",
}


def sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def condition_index(value: int | str) -> int:
  if isinstance(value, str):
    if value == "null":
      return NULL_CONDITION
    if value in CONDITION_NAMES:
      return CONDITION_NAMES.index(value)
    try:
      value = int(value)
    except ValueError as error:
      raise ValueError(f"Unknown condition: {value}") from error
  if not 0 <= value <= NULL_CONDITION:
    raise ValueError(f"Invalid condition: {value}")
  return value


class BalancedConditionSampler(Sampler[int]):
  def __init__(self, labels: torch.Tensor, seed: int = 42, samples: int = 3360):
    if samples <= 0 or samples % 5:
      raise ValueError("samples must be positive and divisible by five")
    self.pools = [(labels == i).nonzero().flatten().cpu() for i in range(5)]
    if any(len(pool) == 0 for pool in self.pools):
      raise ValueError("Every condition must have training examples")
    self.seed, self.samples, self.epoch = seed, samples, 0

  def __len__(self):
    return self.samples

  def __iter__(self):
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    indices = torch.cat([
      pool[torch.randint(len(pool), (self.samples // 5,), generator=generator)]
      for pool in self.pools
    ])
    self.epoch += 1
    return iter(indices[torch.randperm(len(indices), generator=generator)].tolist())


class ShoulderPhaseBalancedSampler(BalancedConditionSampler):
  """Keep five classes balanced, with 50/50 entering/holding in each shoulder."""

  def __init__(self, labels, phase_ids, seed=42, samples=3360):
    super().__init__(labels, seed, samples)
    if samples % 10 or phase_ids.shape != labels.shape:
      raise ValueError("Phase sampling needs aligned labels and samples divisible by ten")
    self.groups = []
    for condition in range(5):
      mask = labels == condition
      if condition in (1, 2):
        if not ((phase_ids[mask] == 1) | (phase_ids[mask] == 2)).all():
          raise ValueError("Every shoulder window needs an entering or holding label")
        for phase in (1, 2):
          pool = (mask & (phase_ids == phase)).nonzero().flatten().cpu()
          if not len(pool):
            raise ValueError("Missing shoulder phase sampling pool")
          self.groups.append((pool, samples // 10))
      else:
        self.groups.append((self.pools[condition], samples // 5))

  def __iter__(self):
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    indices = torch.cat([
      pool[torch.randint(len(pool), (count,), generator=generator)]
      for pool, count in self.groups
    ])
    self.epoch += 1
    return iter(indices[torch.randperm(len(indices), generator=generator)].tolist())


@torch.no_grad()
def sample_condition(
  bundle, condition: int | str, count: int, *,
  initial_noise: torch.Tensor | None = None, guidance_scale: float = 1.0,
) -> torch.Tensor:
  model, scheduler, q_low, q_high, feature_dim, window_size = bundle
  from smp.pretrain.schema import resolve_condition, validate_schema
  schema = getattr(model, "condition_schema", None)
  if schema is None and getattr(model, "num_conditions", 0) == 5:
    schema = CONDITION_SCHEMA
  validate_schema(schema, getattr(model, "num_conditions", 0))
  device = q_low.device
  condition = resolve_condition(condition, schema)
  x = (
    torch.randn(count, window_size, feature_dim, device=device)
    if initial_noise is None else initial_noise.clone().to(device)
  )
  if x.shape != (count, window_size, feature_dim):
    raise ValueError("initial_noise shape mismatch")
  labels = torch.full((count,), condition, device=device, dtype=torch.long)
  null = torch.full_like(labels, schema["null_id"])
  for timestep in reversed(range(scheduler.num_timesteps)):
    t = torch.full_like(labels, timestep)
    eps = model(x, t, labels)
    if guidance_scale != 1.0:
      unconditional = model(x, t, null)
      eps = unconditional + guidance_scale * (eps - unconditional)
    if not torch.isfinite(eps).all():
      raise FloatingPointError("Nonfinite conditional DDPM prediction")
    x = scheduler.step(eps, x, timestep)
  windows = (x + 1.0) * 0.5 * (q_high - q_low) + q_low
  if not torch.isfinite(windows).all():
    raise FloatingPointError("Nonfinite conditional DDPM sample")
  return windows
