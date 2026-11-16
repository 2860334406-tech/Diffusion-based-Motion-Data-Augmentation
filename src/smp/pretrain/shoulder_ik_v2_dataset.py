"""SHA-verified loader for the phase-free G1 shoulder IK-v2 dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_ik_v2 import ShoulderIkV2Sampler

DATA_DIR = Path("logs/reproduction/g1_shoulder_ik_v2/formal/dataset")
DATA_SHA = "d9ad8ef6d8d8b9680709d113556a5efd50ae4812c1138b6fb832dd20f7ec486e"
SOURCE_MANIFEST_SHA = "a2ed23e37ac772c4a1379f85f54bbf9a764320264580a33bc23e684be6a70d2a"
EXPECTED_COUNTS = {
  "train": [21840, 29169, 29169],
  "validation": [4680, 6241, 6241],
  "test": [4680, 6214, 6214],
}


class ShoulderIkV2Dataset(Dataset):
  """Normalized windows plus immutable clip/amplitude lineage."""

  def __init__(self, root=DATA_DIR, split="train", expected_sha=DATA_SHA):
    root = Path(root)
    if split not in EXPECTED_COUNTS:
      raise ValueError(f"Unknown split: {split}")
    self.manifest_sha256 = sha256(root / "manifest.json")
    if self.manifest_sha256 != expected_sha:
      raise ValueError("IK-v2 dataset manifest identity mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("gate_passed") is not True:
      raise ValueError("IK-v2 dataset was not accepted")
    identity = manifest.get("identity", {})
    if identity.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA:
      raise ValueError("IK-v2 source manifest identity mismatch")
    if identity.get("feature_contract") != "smp_g1_window_10x59_continuous_yaw_v3":
      raise ValueError("IK-v2 feature contract mismatch")
    if (identity.get("fps"), identity.get("window_size"), identity.get("stride")) != (
      50,
      10,
      5,
    ):
      raise ValueError("IK-v2 temporal contract mismatch")

    for filename, key in (
      (manifest["clip_table"], "clip_table_sha256"),
      (manifest["windows_index"], "windows_index_sha256"),
      (manifest["norm_stats"], "norm_stats_sha256"),
    ):
      if sha256(root / filename) != manifest[key]:
        raise ValueError(f"IK-v2 auxiliary identity mismatch: {filename}")
    shard_record = next(item for item in manifest["shards"] if item["split"] == split)
    shard_path = root / shard_record["path"]
    if sha256(shard_path) != shard_record["sha256"]:
      raise ValueError(f"IK-v2 shard checksum mismatch: {split}")

    required = {
      "windows",
      "condition_id",
      "clip_index",
      "frame50_start",
      "target_yaw_amplitude_deg",
      "actual_waist_amplitude_deg",
      "target_mean_yaw_speed_deg_s",
      "clip_balanced_weight",
    }
    with np.load(shard_path, allow_pickle=False) as data:
      if required.difference(data.files):
        raise ValueError(
          f"IK-v2 shard is missing fields: {required.difference(data.files)}"
        )
      if "phase_id" in data.files:
        raise ValueError("IK-v2 must remain phase-free")
      arrays = {name: data[name].copy() for name in required}
    windows = arrays["windows"]
    labels = arrays["condition_id"]
    count = len(labels)
    if windows.shape != (count, 10, 59):
      raise ValueError(f"Unexpected IK-v2 window shape: {windows.shape}")
    if any(
      np.asarray(value).shape != (count,)
      for key, value in arrays.items()
      if key != "windows"
    ):
      raise ValueError("IK-v2 window lineage arrays are misaligned")
    if not all(np.isfinite(value).all() for value in arrays.values()):
      raise ValueError("IK-v2 shard contains NaN/Inf")
    if labels.dtype.kind not in "iu" or not np.isin(labels, [0, 1, 2]).all():
      raise ValueError("IK-v2 condition IDs are invalid")
    counts = np.bincount(labels, minlength=3).tolist()
    if (
      counts != EXPECTED_COUNTS[split] or counts != manifest["condition_counts"][split]
    ):
      raise ValueError("IK-v2 condition counts mismatch")

    clips = json.loads((root / manifest["clip_table"]).read_text())
    clip_by_index = {int(item["clip_index"]): item for item in clips}
    if len(clip_by_index) != manifest["clips"]:
      raise ValueError("IK-v2 clip table contains duplicate indices")
    rows = []
    for clip_index, frame, label, target, actual, speed in zip(
      arrays["clip_index"],
      arrays["frame50_start"],
      labels,
      arrays["target_yaw_amplitude_deg"],
      arrays["actual_waist_amplitude_deg"],
      arrays["target_mean_yaw_speed_deg_s"],
      strict=True,
    ):
      clip = clip_by_index.get(int(clip_index))
      if clip is None or clip["split"] != split:
        raise ValueError("IK-v2 window references an invalid clip")
      rows.append(
        {
          "source_id": clip["clip_id"],
          "clip_id": clip["clip_id"],
          "family_id": clip["family_id"],
          "clip_condition_id": int(clip["clip_condition_id"]),
          "condition_id": int(label),
          "frame50_start": int(frame),
          "target_yaw_amplitude_deg": float(target),
          "actual_waist_amplitude_deg": float(actual),
          "target_mean_yaw_speed_deg_s": float(speed),
        }
      )

    with np.load(root / manifest["norm_stats"], allow_pickle=False) as stats:
      self.q_low = stats["q_low"].astype(np.float32)
      self.q_high = stats["q_high"].astype(np.float32)
    if self.q_low.shape != (59,) or self.q_high.shape != (59,):
      raise ValueError("IK-v2 normalization shape mismatch")
    if (
      not np.isfinite([self.q_low, self.q_high]).all()
      or not (self.q_high > self.q_low).all()
    ):
      raise ValueError("IK-v2 normalization is invalid")

    self.raw = torch.from_numpy(windows.astype(np.float32))
    low, high = torch.from_numpy(self.q_low), torch.from_numpy(self.q_high)
    self.windows = 2.0 * (self.raw - low) / (high - low) - 1.0
    self.labels = torch.from_numpy(labels.astype(np.int64))
    self.clip_indices = torch.from_numpy(arrays["clip_index"].astype(np.int64))
    self.frame_starts = torch.from_numpy(arrays["frame50_start"].astype(np.int64))
    self.target_amplitudes_deg = torch.from_numpy(
      arrays["target_yaw_amplitude_deg"].astype(np.float32)
    )
    self.actual_amplitudes_deg = torch.from_numpy(
      arrays["actual_waist_amplitude_deg"].astype(np.float32)
    )
    self.clip_balanced_weight = torch.from_numpy(
      arrays["clip_balanced_weight"].astype(np.float32)
    )
    self.rows = rows
    self.clips = clip_by_index
    self.manifest = manifest
    self.split = split
    self.root = root
    if not torch.isfinite(self.windows).all():
      raise ValueError("Normalized IK-v2 data contain NaN/Inf")
    if split == "train":
      ShoulderIkV2Sampler(self.labels, self.clip_indices, self.target_amplitudes_deg)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.windows[index], self.labels[index]

  def mirror_indices(self) -> torch.Tensor:
    """Pair separately solved left/right clips by family and 50-Hz frame."""
    lookup = {
      (row["family_id"], row["clip_condition_id"], row["frame50_start"]): index
      for index, row in enumerate(self.rows)
    }
    result = torch.arange(len(self.rows))
    for index, row in enumerate(self.rows):
      condition = int(self.labels[index])
      if condition == 0:
        continue
      opposite = 2 if condition == 1 else 1
      key = (row["family_id"], opposite, row["frame50_start"])
      if key not in lookup:
        raise ValueError(f"Missing left/right paired IK-v2 window: {key}")
      other = lookup[key]
      if int(self.labels[other]) != opposite:
        raise ValueError("IK-v2 paired label mismatch")
      result[index] = other
    return result

  def clip_equal_weights(self, condition: int) -> torch.Tensor:
    """Return weights summing to one with equal mass for each eligible clip."""
    selected = (self.labels == condition).nonzero().flatten()
    clips = self.clip_indices[selected]
    unique = torch.unique(clips)
    weights = torch.zeros(len(self), dtype=torch.float64)
    for clip in unique:
      indices = selected[clips == clip]
      weights[indices] = 1.0 / (len(unique) * len(indices))
    return weights

  def clip_groups(self, indices: torch.Tensor) -> dict[int, torch.Tensor]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index in indices.tolist():
      groups[int(self.clip_indices[index])].append(index)
    return {clip: torch.tensor(values) for clip, values in groups.items()}
