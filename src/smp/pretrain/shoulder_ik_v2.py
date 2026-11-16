"""Three-condition schema and clip-first sampler for the G1 IK-v2 dataset."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler

CONDITION_NAMES = ("stand", "left_shoulder", "right_shoulder")
NULL_CONDITION = 3
CONDITION_SCHEMA = {
  "version": 3,
  "name": "g1_shoulder_ik_v2_conditional_v1",
  "names": list(CONDITION_NAMES),
  "null_id": NULL_CONDITION,
  "mirror_ids": [0, 2, 1],
  "meaning": "synthetic G1 motion-response labels; NULL is classifier-free only",
}
AMPLITUDE_EDGES_DEG = tuple(float(value) for value in range(15, 75, 5))
SAMPLING = {
  "algorithm": "condition_clip_first_amplitude_stratified_v1",
  "samples_per_epoch": 3360,
  "per_condition": [672, 1344, 1344],
  "amplitude_edges_deg": list(AMPLITUDE_EDGES_DEG),
  "shoulder_bins": 11,
  "within_group": "uniform_clip_then_uniform_window_with_replacement",
  "shoulder_bin_remainder": "two rotating extra samples per condition and epoch",
  "seed": 42,
  "proportions_apply_before_null_dropout": True,
}


def condition_index(value: int | str) -> int:
  if isinstance(value, str):
    if value == "null":
      return NULL_CONDITION
    if value in CONDITION_NAMES:
      return CONDITION_NAMES.index(value)
    try:
      value = int(value)
    except ValueError as error:
      raise ValueError(f"Unsupported IK-v2 condition: {value}") from error
  if type(value) is not int or not 0 <= value <= NULL_CONDITION:
    raise ValueError(f"Invalid IK-v2 condition: {value}")
  return value


class ShoulderIkV2Sampler(Sampler[int]):
  """Draw exact 20/40/40 quotas without weighting long clips more heavily."""

  def __init__(
    self,
    labels,
    clip_indices,
    target_amplitudes_deg,
    *,
    seed: int = 42,
    samples: int = 3360,
  ) -> None:
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    clip_indices = torch.as_tensor(clip_indices, dtype=torch.long).cpu()
    amplitudes = torch.as_tensor(target_amplitudes_deg, dtype=torch.float32).cpu()
    if (
      labels.ndim != 1
      or clip_indices.shape != labels.shape
      or amplitudes.shape != labels.shape
    ):
      raise ValueError("Labels, clip indices and amplitudes must be aligned vectors")
    if samples <= 0 or samples % 10:
      raise ValueError("samples must be positive and divisible by ten")
    if not ((labels >= 0) & (labels < 3)).all():
      raise ValueError("Dataset conditions must contain only stand/left/right")
    if not torch.isfinite(amplitudes).all():
      raise ValueError("Target amplitudes contain NaN/Inf")
    self.seed = int(seed)
    self.samples = int(samples)
    self.epoch = 0
    scale = samples // 3360
    self.condition_quotas = (672 * scale, 1344 * scale, 1344 * scale)

    self.stand = self._clip_pools((labels == 0).nonzero().flatten(), clip_indices)
    if not self.stand:
      raise ValueError("Missing stand sampling pool")

    edges = torch.tensor(AMPLITUDE_EDGES_DEG[1:-1])
    self.shoulder: dict[int, list[list[torch.Tensor]]] = {}
    for condition in (1, 2):
      selected = (labels == condition).nonzero().flatten()
      if not len(selected):
        raise ValueError(f"Missing shoulder condition {condition}")
      values = amplitudes[selected]
      if bool(((values < 15.0 - 1.0e-3) | (values > 70.0 + 1.0e-3)).any()):
        raise ValueError("Shoulder target amplitude is outside [15, 70] degrees")
      bin_ids = torch.bucketize(values, edges, right=False)
      bins = []
      for bin_id in range(11):
        pools = self._clip_pools(selected[bin_ids == bin_id], clip_indices)
        if not pools:
          raise ValueError(
            f"Missing amplitude bin: condition={condition}, bin={bin_id}"
          )
        bins.append(pools)
      self.shoulder[condition] = bins

  @staticmethod
  def _clip_pools(indices: torch.Tensor, clip_indices: torch.Tensor):
    pools = []
    for clip in torch.unique(clip_indices[indices], sorted=True):
      pool = indices[clip_indices[indices] == clip]
      if len(pool):
        pools.append(pool)
    return pools

  @staticmethod
  def _draw_clip_first(
    pools: list[torch.Tensor], count: int, generator: torch.Generator
  ) -> torch.Tensor:
    selected_clips = torch.randint(len(pools), (count,), generator=generator)
    result = []
    for clip in selected_clips.tolist():
      pool = pools[clip]
      offset = int(torch.randint(len(pool), (1,), generator=generator))
      result.append(pool[offset])
    return torch.stack(result)

  def __len__(self) -> int:
    return self.samples

  def __iter__(self):
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    draws = [self._draw_clip_first(self.stand, self.condition_quotas[0], generator)]
    for condition in (1, 2):
      quota = self.condition_quotas[condition]
      base, remainder = divmod(quota, 11)
      extras = {(2 * self.epoch + offset) % 11 for offset in range(remainder)}
      for bin_id, pools in enumerate(self.shoulder[condition]):
        draws.append(
          self._draw_clip_first(pools, base + int(bin_id in extras), generator)
        )
    selected = torch.cat(draws)
    if len(selected) != self.samples:
      raise AssertionError("Sampler quota implementation drifted")
    self.epoch += 1
    order = torch.randperm(len(selected), generator=generator)
    return iter(selected[order].tolist())

  def state_dict(self) -> dict[str, int]:
    return {"seed": self.seed, "samples": self.samples, "epoch": self.epoch}

  def load_state_dict(self, state: dict[str, int]) -> None:
    if state.get("seed") != self.seed or state.get("samples") != self.samples:
      raise ValueError("Sampler resume identity mismatch")
    epoch = state.get("epoch")
    if type(epoch) is not int or epoch < 0:
      raise ValueError("Invalid sampler epoch")
    self.epoch = epoch
