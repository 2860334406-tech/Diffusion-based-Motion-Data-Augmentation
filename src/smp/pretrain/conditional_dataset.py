"""Manifest-verified, subject-disjoint conditional motion windows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smp.pretrain.conditioning import CONDITION_SCHEMA, sha256


class ConditionalMotionDataset(Dataset):
  def __init__(self, root: str | Path, split: str):
    root = Path(root)
    self.manifest_sha256 = sha256(root / "manifest.json")
    self.manifest = json.loads((root / "manifest.json").read_text())
    if self.manifest["condition_schema"] != CONDITION_SCHEMA:
      raise ValueError("Condition schema mismatch")
    if sha256(root / "norm_stats.npz") != self.manifest["norm_stats_sha256"]:
      raise ValueError("Normalization checksum mismatch")
    chunks, labels, phases = [], [], []
    self.sampling = self.manifest["sampling"]
    phase_balanced = self.sampling.get("algorithm") == "shoulder_phase_balanced_v1"
    for item in self.manifest["files"]:
      if item["split"] != split:
        continue
      if ("subject5" in item["source"]) != (split == "validation"):
        raise ValueError("Subject leakage in curated manifest")
      path = root / item["path"]
      if sha256(path) != item["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {path}")
      with np.load(path, allow_pickle=False) as shard:
        windows, ids = shard["windows"], shard["condition_id"]
        if phase_balanced:
          phase = shard["phase_id"]
          if phase.shape != ids.shape or phase.dtype.kind not in "iu":
            raise ValueError("Invalid shoulder phase labels")
          shoulder = np.isin(ids, (1, 2))
          if not np.isin(phase[shoulder], (1, 2)).all() or np.any(phase[~shoulder] != 0):
            raise ValueError("Shoulder phase/condition mismatch")
          phases.append(phase)
      if windows.shape[1:] != (10, 59) or ids.shape != (len(windows),):
        raise ValueError("Invalid conditional shard shapes")
      if not np.isfinite(windows).all() or not ((ids >= 0) & (ids < 5)).all():
        raise ValueError("Invalid conditional data values")
      chunks.append(windows)
      labels.append(ids)
    if not chunks:
      raise ValueError(f"No shards for split {split}")
    with np.load(root / "norm_stats.npz") as stats:
      self.q_low = stats["q_low"].astype(np.float32)
      self.q_high = stats["q_high"].astype(np.float32)
    raw = np.concatenate(chunks).astype(np.float32)
    self.raw = torch.from_numpy(raw)
    self.windows = torch.from_numpy(2*(raw-self.q_low)/(self.q_high-self.q_low)-1)
    self.labels = torch.from_numpy(np.concatenate(labels).astype(np.int64))
    self.phase_ids = torch.from_numpy(np.concatenate(phases).astype(np.int64)) if phases else None
    if any(not (self.labels == i).any() for i in range(5)):
      raise ValueError(f"Missing conditions in {split}")

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.windows[index], self.labels[index]
