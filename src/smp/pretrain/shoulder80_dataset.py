"""Read the combined EXP-028 index, not just its external-only NPZ directory."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder80 import CONDITION_SCHEMA, SAMPLING, Shoulder80Sampler

DATA_DIR = Path("logs/reproduction/external_shoulder_data_review_v1/review")
DATA_SHA = "d5616d77a051266bde3b075534ce434d6159b1a1dd790ab0a56057d63f29eb05"


class Shoulder80Dataset(Dataset):
  """SHA-verified original LAFAN + G1 Moves + AMASS, with immutable lineage."""

  def __init__(self, root=DATA_DIR, split="train", expected_sha=DATA_SHA):
    root = Path(root)
    if split not in ("train", "validation", "test"):
      raise ValueError("Unknown split")
    self.manifest_sha256 = sha256(root / "manifest.json")
    if self.manifest_sha256 != expected_sha:
      raise ValueError("Expanded data identity mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["gate"]["passed"]:
      raise ValueError("Data gate failed; training prohibited")
    if (
      manifest["condition_schema"] != CONDITION_SCHEMA
      or manifest["sampling"] != SAMPLING
    ):
      raise ValueError("Schema/sampling mismatch")
    for filename, key in (
      ("windows.csv", "windows_manifest_sha256"),
      ("norm_stats.npz", "norm_stats_sha256"),
    ):
      if sha256(root / filename) != manifest[key]:
        raise ValueError(f"Data identity mismatch: {filename}")
    base_path = Path(manifest["historical_data_manifest"])
    if sha256(base_path) != manifest["historical_data_sha256"]:
      raise ValueError("Original LAFAN manifest identity mismatch")
    base = json.loads(base_path.read_text())
    inventory = {}
    for directory, entries in (
      (base_path.parent, base["files"]),
      (root, manifest["external_files"]),
    ):
      for item in entries:
        path = str((directory / item["path"]).resolve())
        if path in inventory:
          raise ValueError("Duplicate shard identity")
        inventory[path] = item
    with (root / "windows.csv").open(newline="") as stream:
      rows = list(csv.DictReader(stream))
    actors, sources, seen, by_shard = {}, {}, set(), defaultdict(list)
    for row in rows:
      for key in ("index", "condition_id", "phase_id", "frame50_start"):
        row[key] = int(row[key])
      for mapping, key in ((actors, "source_group"), (sources, "source_id")):
        if mapping.setdefault(row[key], row["split"]) != row["split"]:
          raise ValueError("Actor/source leakage across splits")
      path = str(Path(row["shard"]).resolve())
      if path not in inventory or inventory[path]["split"] != row["split"]:
        raise ValueError("Unregistered shard/split")
      identity = (path, row["index"])
      if identity in seen:
        raise ValueError("Duplicated window index")
      seen.add(identity)
      by_shard[path].append(row)
    if set(inventory) != set(by_shard):
      raise ValueError("Combined index omits LAFAN or external shards")
    chunks, labels, phases, lineage = [], [], [], []
    for path in sorted(by_shard):
      item = inventory[path]
      if item["split"] != split:
        continue
      if sha256(path) != item["sha256"]:
        raise ValueError(f"Shard checksum mismatch: {path}")
      selected = sorted(by_shard[path], key=lambda row: row["index"])
      with np.load(path, allow_pickle=False) as data:
        raw, ids, phase = data["windows"], data["condition_id"], data["phase_id"]
        if raw.shape != (len(ids), 10, 59) or phase.shape != ids.shape:
          raise ValueError("Shard shape mismatch")
        if not np.isfinite(raw).all() or ids.dtype.kind not in "iu":
          raise ValueError("Invalid shard values/labels")
        if [row["index"] for row in selected] != list(range(len(raw))):
          raise ValueError("Incomplete shard index")
        for i, row in enumerate(selected):
          if (row["condition_id"], row["phase_id"], row["frame50_start"]) != (
            int(ids[i]),
            int(phase[i]),
            int(data["frame50_start"][i]),
          ):
            raise ValueError("Window lineage mismatch")
        chunks.append(raw.astype(np.float32))
        labels.append(ids.astype(np.int64))
        phases.append(phase.astype(np.int64))
      lineage.extend(selected)
    with np.load(root / "norm_stats.npz", allow_pickle=False) as stats:
      self.q_low = stats["q_low"].astype(np.float32)
      self.q_high = stats["q_high"].astype(np.float32)
    if self.q_low.shape != (59,) or self.q_high.shape != (59,):
      raise ValueError("Normalization shape mismatch")
    if (
      not np.isfinite([self.q_low, self.q_high]).all()
      or not (self.q_high > self.q_low).all()
    ):
      raise ValueError("Invalid normalization")
    self.raw = torch.from_numpy(np.concatenate(chunks))
    low, high = torch.from_numpy(self.q_low), torch.from_numpy(self.q_high)
    self.windows = 2 * (self.raw - low) / (high - low) - 1
    self.labels = torch.from_numpy(np.concatenate(labels))
    self.phase_ids = torch.from_numpy(np.concatenate(phases))
    self.rows, self.manifest, self.split = lineage, manifest, split
    self.sampling = SAMPLING
    if torch.bincount(self.labels, minlength=3).tolist() != manifest["counts"][split]:
      raise ValueError("Combined sample counts mismatch")
    if not torch.isfinite(self.windows).all():
      raise ValueError("Nonfinite normalized data")
    Shoulder80Sampler(self.labels, self.phase_ids)  # Also checks every phase pool.

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.windows[index], self.labels[index]

  def mirror_indices(self):
    def key(row, mirror=None):
      value = row["mirrored"].lower() == "true" if mirror is None else mirror
      return row["source_id"], row["frame50_start"], value

    lookup = {key(row): index for index, row in enumerate(self.rows)}
    result = []
    for i, row in enumerate(self.rows):
      j = lookup[key(row, row["mirrored"].lower() != "true")]
      if int(self.labels[j]) != (0, 2, 1)[int(self.labels[i])]:
        raise ValueError("Mirrored label mismatch")
      result.append(j)
    return torch.tensor(result)
