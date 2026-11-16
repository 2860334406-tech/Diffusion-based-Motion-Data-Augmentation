"""Independent three-condition shoulder prior schema and exact epoch sampler.

This module deliberately does not replace the historical five-condition schema.
Condition IDs describe desired motion, not measured contact annotations.
"""

from __future__ import annotations

import re

import torch
from torch.utils.data import Sampler

CONDITION_NAMES = ("stand", "left_shoulder", "right_shoulder")
NULL_CONDITION = 3
CONDITION_SCHEMA = {
  "version": 2,
  "name": "shoulder80_v1",
  "names": list(CONDITION_NAMES),
  "null_id": NULL_CONDITION,
  "mirror_ids": [0, 2, 1],
  "meaning": "motion-response labels; LAFAN has no measured contact annotations",
}
PHASE_NAMES = ("stand", "early_entry", "strong_entry", "hold")
SPLIT_SUBJECTS = {"train": [1, 2, 3], "validation": [4], "test": [5]}
SAMPLING = {
  "algorithm": "shoulder80_three_phase_v1",
  "samples_per_epoch": 3360,
  "per_condition": [672, 1344, 1344],
  "per_condition_phase": [[672, 0, 0, 0], [0, 336, 336, 672], [0, 336, 336, 672]],
  "phase_names": list(PHASE_NAMES),
  "within_phase": "uniform_with_replacement",
  "seed": 42,
  "proportions_apply_before_null_dropout": True,
}


def subject_id(source: str) -> int:
  match = re.search(r"subject([1-5])(?:\D|$)", source)
  if not match:
    raise ValueError(f"Missing supported subject ID: {source}")
  return int(match[1])


def subject_split(source: str) -> str:
  subject = subject_id(source)
  return next(name for name, ids in SPLIT_SUBJECTS.items() if subject in ids)


def condition_index(value: int | str) -> int:
  """Resolve only this schema; ID 3 means NULL, never historical chest."""
  if isinstance(value, str):
    if value == "null":
      return NULL_CONDITION
    if value in CONDITION_NAMES:
      return CONDITION_NAMES.index(value)
    try:
      value = int(value)
    except ValueError as error:
      raise ValueError(f"Unsupported shoulder80 condition: {value}") from error
  if not isinstance(value, int) or not 0 <= value <= NULL_CONDITION:
    raise ValueError(f"Invalid shoulder80 condition: {value}")
  return value


class Shoulder80Sampler(Sampler[int]):
  """Exactly 20/40/40, with 25/25/50 early/strong/hold in each shoulder."""

  def __init__(self, labels, phase_ids, seed=42, samples=3360):
    labels = torch.as_tensor(labels).cpu()
    phase_ids = torch.as_tensor(phase_ids).cpu()
    if labels.ndim != 1 or phase_ids.shape != labels.shape:
      raise ValueError("Labels and phases must be aligned one-dimensional tensors")
    if labels.dtype not in (torch.int32, torch.int64) or phase_ids.dtype not in (
      torch.int32,
      torch.int64,
    ):
      raise ValueError("Condition and phase IDs must be integers")
    if samples <= 0 or samples % 10:
      raise ValueError("Epoch samples must be positive and divisible by ten")
    if not ((labels >= 0) & (labels < 3)).all():
      raise ValueError("Chest, back and NULL are not dataset conditions")
    if (phase_ids[labels == 0] != 0).any() or not (
      (phase_ids[labels != 0] >= 1) & (phase_ids[labels != 0] <= 3)
    ).all():
      raise ValueError("Condition/phase mismatch")
    self.seed, self.samples, self.epoch = int(seed), int(samples), 0
    self.groups = []
    for condition, quotas in enumerate(SAMPLING["per_condition_phase"]):
      for phase, quota in enumerate(quotas):
        if not quota:
          continue
        pool = ((labels == condition) & (phase_ids == phase)).nonzero().flatten()
        if not len(pool):
          raise ValueError(
            f"Missing sampling pool: condition={condition}, phase={phase}"
          )
        self.groups.append((pool, quota * samples // 3360))

  def __len__(self):
    return self.samples

  def __iter__(self):
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    selected = torch.cat(
      [
        pool[torch.randint(len(pool), (count,), generator=generator)]
        for pool, count in self.groups
      ]
    )
    self.epoch += 1
    return iter(selected[torch.randperm(len(selected), generator=generator)].tolist())

  def state_dict(self):
    return {"seed": self.seed, "samples": self.samples, "epoch": self.epoch}

  def load_state_dict(self, state):
    if state["seed"] != self.seed or state["samples"] != self.samples:
      raise ValueError("Sampler resume identity mismatch")
    if not isinstance(state["epoch"], int) or state["epoch"] < 0:
      raise ValueError("Invalid sampler epoch")
    self.epoch = state["epoch"]
