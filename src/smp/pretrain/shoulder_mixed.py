"""Frozen EXP-035 schema, source quotas and physically scaled normalization."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler

CONDITION_NAMES = ("stand", "left_shoulder", "right_shoulder")
CONDITION_SCHEMA = {
  "version": 4,
  "name": "g1_shoulder_mixed_conditional_v1",
  "names": list(CONDITION_NAMES),
  "null_id": 3,
  "mirror_ids": [0, 2, 1],
  "meaning": "motion-response classes; pelvis participation permitted; NULL is not stand",
}
SOURCES = ("ik", "personal", "lafan", "g1_moves", "amass_g1")
QUOTAS = {
  0: {"ik": 168, "personal": 168, "lafan": 112, "g1_moves": 112, "amass_g1": 112},
  1: {"ik": 672, "personal": 672},
  2: {"ik": 672, "personal": 672},
}
SAMPLING = {
  "algorithm": "mixed_source_clip_amplitude_v1",
  "samples_per_epoch": 3360,
  "per_condition": [672, 1344, 1344],
  "source_quotas": {str(k): v for k, v in QUOTAS.items()},
  "ik_amplitude_bins": 11,
  "legacy_hierarchy": "source, actor, original clip, window; mirrors share clip mass",
  "personal_hierarchy": "clip, window",
  "seed": 42,
  "proportions_apply_before_null_dropout": True,
}
MIN_SPAN = np.array(
  [0.02] * 3 + [0.05] * 6 + [0.10] * 29 + [0.02] * 15 + [0.20] * 3 + [0.40] * 3,
  dtype=np.float64,
)
NORM_SCHEMA = "mixed_train_weighted_q01_q99_physical_span_v1"


def normalization(raw: np.ndarray, weights: np.ndarray):
  """Exact weighted train quantiles; each window gives equal mass to its frames."""
  raw = np.asarray(raw)
  weights = np.asarray(weights, dtype=np.float64)
  if raw.ndim != 3 or raw.shape[1:] != (10, 59) or weights.shape != (len(raw),):
    raise ValueError("Invalid normalization inputs")
  if not np.isfinite(raw).all() or not np.isfinite(weights).all():
    raise ValueError("Nonfinite normalization inputs")
  if (weights < 0).any() or weights.sum() <= 0:
    raise ValueError("Invalid normalization weights")
  flat = raw.reshape(-1, 59)
  frame_weights = np.repeat(weights / weights.sum() / 10, 10)
  quantiles = np.empty((2, 59), dtype=np.float64)
  for feature in range(59):
    order = np.argsort(flat[:, feature], kind="stable")
    cdf = np.cumsum(frame_weights[order])
    cdf /= cdf[-1]
    quantiles[:, feature] = flat[
      order[np.searchsorted(cdf, [0.01, 0.99], side="left")], feature
    ]
  center = quantiles.mean(0)
  span = np.maximum(quantiles[1] - quantiles[0], MIN_SPAN)
  # Float32 endpoints are widened slightly so roundoff cannot violate a floor.
  low = np.nextafter((center - span / 2).astype(np.float32), -np.inf)
  high = np.nextafter((center + span / 2).astype(np.float32), np.inf)
  return low, high, quantiles


def condition_weights(rows: list[dict], condition: int) -> np.ndarray:
  """Expected source/clip weights, including IK amplitude and legacy actor balance."""
  selected = [i for i, row in enumerate(rows) if int(row["condition_id"]) == condition]
  sources = {rows[i]["dataset"] for i in selected}
  if not {"ik", "personal"}.issubset(sources):
    raise ValueError(f"Missing primary source for condition {condition}")
  source_mass = {
    "ik": 0.25 if condition == 0 else 0.5,
    "personal": 0.25 if condition == 0 else 0.5,
  }
  if condition == 0:
    old = sorted(sources - {"ik", "personal"})
    if not old:
      raise ValueError("No held-out legacy stand pool")
    source_mass.update({source: 0.5 / len(old) for source in old})
  elif sources != {"ik", "personal"}:
    raise ValueError("Legacy shoulder labels are forbidden")
  result = np.zeros(len(rows), dtype=np.float64)
  for source, mass in source_mass.items():
    ids = [i for i in selected if rows[i]["dataset"] == source]
    groups = {}
    for i in ids:
      row = rows[i]
      if source == "ik" and condition:
        amplitude = float(row["target_yaw_amplitude_deg"])
        if not 15 - 1e-3 <= amplitude <= 70 + 1e-3:
          raise ValueError("Invalid IK target amplitude")
        group = min(int((amplitude - 15) // 5), 10)
      elif source not in {"ik", "personal"}:
        group = row["source_group"]
      else:
        group = "all"
      groups.setdefault(group, {}).setdefault(row["clip_id"], []).append(i)
    if source == "ik" and condition and len(groups) != 11:
      raise ValueError(f"Missing IK amplitude bin in condition {condition}")
    for clips in groups.values():
      for indices in clips.values():
        result[indices] = mass / len(groups) / len(clips) / len(indices)
  if not np.isclose(result.sum(), 1):
    raise ValueError("Conditional sampling mass does not sum to one")
  return result


class ShoulderMixedSampler(Sampler[int]):
  """Exact source quotas, with one rotating extra IK sample across eleven bins."""

  def __init__(self, dataset, *, seed=42):
    self.dataset = dataset
    self.seed, self.samples, self.epoch = int(seed), 3360, 0
    self.rows = dataset.rows
    self.weights = [torch.from_numpy(condition_weights(self.rows, c)) for c in range(3)]
    for condition, sources in QUOTAS.items():
      for source in sources:
        if not any(
          int(r["condition_id"]) == condition and r["dataset"] == source
          for r in self.rows
        ):
          raise ValueError(f"Missing required training pool: {condition}/{source}")
    self.pools = {}
    for condition, sources in QUOTAS.items():
      for source in sources:
        ids = torch.tensor(
          [
            i
            for i, row in enumerate(self.rows)
            if int(row["condition_id"]) == condition and row["dataset"] == source
          ],
          dtype=torch.long,
        )
        if source == "ik" and condition:
          values = torch.tensor(
            [float(self.rows[i]["target_yaw_amplitude_deg"]) for i in ids.tolist()]
          )
          bins = ((values - 15) / 5).floor().long().clamp(0, 10)
          self.pools[condition, source] = [ids[bins == b] for b in range(11)]
        else:
          self.pools[condition, source] = [ids]

  def __len__(self):
    return self.samples

  def __iter__(self):
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    draws = []
    for condition, sources in QUOTAS.items():
      for source, quota in sources.items():
        pools = self.pools[condition, source]
        base, remainder = divmod(quota, len(pools))
        extras = {(self.epoch + i) % len(pools) for i in range(remainder)}
        for b, ids in enumerate(pools):
          count = base + int(b in extras)
          offsets = torch.multinomial(
            self.weights[condition][ids], count, True, generator=generator
          )
          draws.append(ids[offsets])
    indices = torch.cat(draws)
    self.epoch += 1
    return iter(indices[torch.randperm(len(indices), generator=generator)].tolist())

  def state_dict(self):
    return {
      "seed": self.seed,
      "samples": self.samples,
      "epoch": self.epoch,
      "sampling": SAMPLING,
    }

  def load_state_dict(self, state):
    if any(state.get(k) != v for k, v in self.state_dict().items() if k != "epoch"):
      raise ValueError("Mixed sampler resume identity mismatch")
    if type(state.get("epoch")) is not int or state["epoch"] < 0:
      raise ValueError("Invalid sampler epoch")
    self.epoch = state["epoch"]
