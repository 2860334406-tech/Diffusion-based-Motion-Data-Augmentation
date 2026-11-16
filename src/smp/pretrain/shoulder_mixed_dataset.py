"""Audited raw-feature assembly and source-balanced EXP-035 dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from smp.data.shoulder_ik import JOINT_NAMES, _build_model, _qpos_ids, atomic_json
from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_mixed import (
  CONDITION_SCHEMA,
  MIN_SPAN,
  NORM_SCHEMA,
  SAMPLING,
  SOURCES,
  ShoulderMixedSampler,
  condition_weights,
  normalization,
)

DATA_DIR = Path("logs/reproduction/g1_shoulder_mixed_prior_ppo_v1/data")
INPUTS = {
  "ik": (
    Path("logs/reproduction/g1_shoulder_ik_v2/formal/dataset"),
    "d9ad8ef6d8d8b9680709d113556a5efd50ae4812c1138b6fb832dd20f7ec486e",
  ),
  "personal": (
    Path("logs/reproduction/g1_personal_retarget_v1/dataset"),
    "616964e09b235db14e69a68db68d0d2224080020fbc78f9c514cb9f1c08157bf",
  ),
  "legacy": (
    Path("logs/reproduction/external_shoulder_data_review_v1/review"),
    "d5616d77a051266bde3b075534ce434d6159b1a1dd790ab0a56057d63f29eb05",
  ),
}
SPLITS = ("train", "validation", "test")


def data_sha(root=DATA_DIR):
  return sha256(Path(root) / "manifest.json")


def _verify(path, expected):
  if sha256(path) != expected:
    raise ValueError(f"Dataset identity mismatch: {path}")


class LegacySafety:
  """Full-current-collision FK screening, never a claim of dynamic feasibility."""

  def __init__(self):
    self.model = _build_model(soft_limits=False)
    self.data = mujoco.MjData(self.model)
    self.qids = _qpos_ids(self.model)
    self.limits = np.array([self.model.joint(n).range for n in JOINT_NAMES])
    self.floor = self.model.geom("shoulder_ik_floor").id
    self.feet = {
      i
      for i in range(self.model.ngeom)
      if "foot" in (self.model.geom(i).name or "")
      and "collision" in (self.model.geom(i).name or "")
    }
    self.cache = {}
    self.rejected = Counter()

  def check(self, window):
    if np.linalg.norm(window[:, 53:55], axis=-1).max() > 0.08 + 1e-5:
      return "stand_speed"
    if np.abs(window[:, 21]).max() > 0.06 + 1e-5:
      return "neutral_waist"
    if abs(float(window[-1, 21] - window[0, 21])) > 0.005 + 1e-5:
      return "waist_change"
    if np.abs(window[:, 58]).max() > 0.15 + 1e-5:
      return "pelvis_speed"
    for frame in window:
      key = frame[:38].tobytes()
      if key in self.cache:
        reason = self.cache[key]
      else:
        reason = self._frame(frame)
        self.cache[key] = reason
      if reason:
        return reason
    return None

  def _frame(self, frame):
    joints = frame[9:38]
    if (
      (joints < self.limits[:, 0] + 1e-4) | (joints > self.limits[:, 1] - 1e-4)
    ).any():
      return "joint_limit"
    x, z = frame[3:6].astype(float), frame[6:9].astype(float)
    x /= np.linalg.norm(x)
    z -= np.dot(z, x) * x
    z /= np.linalg.norm(z)
    rotation = Rotation.from_matrix(np.stack([x, np.cross(z, x), z], axis=-1))
    if frame[2] < 0.60 or np.abs(rotation.as_euler("xyz")[:2]).max() > np.deg2rad(20):
      return "root_pose"
    self.data.qpos[:] = self.model.qpos0
    self.data.qpos[:3] = frame[:3]
    self.data.qpos[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
    self.data.qpos[self.qids] = joints
    mujoco.mj_forward(self.model, self.data)
    ids = np.array(sorted(self.feet))
    bottoms = self.data.geom_xpos[ids, 2] - self.model.geom_size[ids, 0]
    if bottoms.min() < -0.02 or bottoms.min() > 0.03:
      return "support_or_floor"
    for contact in self.data.contact:
      a, b = map(int, contact.geom)
      if self.floor not in (a, b) and contact.dist < -1e-4:
        return "self_collision"
      if self.floor in (a, b) and contact.dist < -1e-4:
        other = b if a == self.floor else a
        if other not in self.feet:
          return "nonfoot_ground"
    return None


def _primary(source, root, manifest, split):
  for field, digest in (
    ("clip_table", "clip_table_sha256"),
    ("windows_index", "windows_index_sha256"),
  ):
    _verify(root / manifest[field], manifest[digest])
  clips = {
    int(c["clip_index"]): c
    for c in json.loads((root / manifest["clip_table"]).read_text())
  }
  item = next(s for s in manifest["shards"] if s["split"] == split)
  _verify(root / item["path"], item["sha256"])
  rows = []
  with np.load(root / item["path"], allow_pickle=False) as z:
    raw = z["windows"].copy()
    for i, (clip_index, label, frame) in enumerate(
      zip(z["clip_index"], z["condition_id"], z["frame50_start"], strict=True)
    ):
      c = clips[int(clip_index)]
      if c["split"] != split:
        raise ValueError("Primary clip split leakage")
      actor = c.get("subject_id", c.get("family_id"))
      row = {
        "dataset": source,
        "source_id": f"{source}/{c['clip_id']}",
        "clip_id": f"{source}/{c['clip_id']}",
        "source_group": f"{source}/{actor}",
        "family_id": f"{source}/{c.get('family_id', actor)}",
        "split": split,
        "condition_id": int(label),
        "frame50_start": int(frame),
        "target_yaw_amplitude_deg": float(z["target_yaw_amplitude_deg"][i])
        if source == "ik"
        else None,
        "actual_waist_delta_rad": float(z["actual_waist_amplitude_deg"][i])
        * np.pi
        / 180
        if source == "ik"
        else float(z["actual_waist_delta_rad"][i]),
        "actual_pelvis_delta_rad": None
        if source == "ik"
        else float(z["actual_pelvis_delta_rad"][i]),
        "parent_shard_sha256": item["sha256"],
        "parent_index": i,
        "mirrored": False,
      }
      rows.append(row)
  return raw, rows


def prepare(root=DATA_DIR):
  root = Path(root)
  code = {
    str(Path(__file__)): sha256(Path(__file__)),
    "src/smp/pretrain/shoulder_mixed.py": sha256("src/smp/pretrain/shoulder_mixed.py"),
  }
  identity = {
    "parents": {k: v[1] for k, v in INPUTS.items()},
    "schema": CONDITION_SCHEMA,
    "sampling": SAMPLING,
    "normalization": NORM_SCHEMA,
    "min_span": MIN_SPAN.tolist(),
    "code": code,
  }
  manifests = {}
  for source, (path, digest) in INPUTS.items():
    _verify(path / "manifest.json", digest)
    manifests[source] = json.loads((path / "manifest.json").read_text())
  if not manifests["ik"]["gate_passed"] or not manifests["personal"]["gate_passed"]:
    raise RuntimeError("Primary source was not accepted")
  if not manifests["personal"]["raw_unnormalized_features"]:
    raise ValueError("Personal source must contain raw features")
  if (root / "manifest.json").exists():
    m = json.loads((root / "manifest.json").read_text())
    if m["identity"] != identity:
      raise ValueError("Existing mixed dataset has another identity")
    for split in SPLITS:
      ShoulderMixedDataset(root, split)
    return m
  root.mkdir(parents=True, exist_ok=True)
  buckets = {s: {"raw": [], "rows": []} for s in SPLITS}
  for source in ("ik", "personal"):
    for split in SPLITS:
      raw, rows = _primary(source, INPUTS[source][0], manifests[source], split)
      buckets[split]["raw"].append(raw)
      buckets[split]["rows"].extend(rows)
  legacy_root = INPUTS["legacy"][0]
  old = manifests["legacy"]
  _verify(legacy_root / "windows.csv", old["windows_manifest_sha256"])
  historical_path = Path(old["historical_data_manifest"])
  _verify(historical_path, old["historical_data_sha256"])
  historical = json.loads(historical_path.read_text())
  # Basenames plus the recorded suffix identify every referenced curated shard.
  approved = {}
  for entry in old["external_files"] + historical["files"]:
    key = Path(entry["path"]).name
    if key in approved and approved[key]["sha256"] != entry["sha256"]:
      raise ValueError("Ambiguous legacy shard basename")
    approved[key] = entry
  safety, cache, rejections, legacy_windows = LegacySafety(), {}, [], Counter()
  with (legacy_root / "windows.csv").open() as stream:
    for original in csv.DictReader(stream):
      if original["condition_id"] != "0":
        continue
      path = Path(original["shard"])
      if str(path) not in cache:
        entry = approved[path.name]
        if not str(path).endswith(entry["path"]):
          raise ValueError("Unapproved legacy shard path")
        _verify(path, entry["sha256"])
        with np.load(path, allow_pickle=False) as z:
          cache[str(path)] = (z["windows"].copy(), z["condition_id"].copy())
      raw, labels = cache[str(path)]
      index = int(original["index"])
      if int(labels[index]) != 0:
        raise ValueError("Legacy window label identity mismatch")
      window = raw[index]
      reason = safety.check(window)
      if reason:
        rejections.append({**original, "reason": reason})
        continue
      split, source = original["split"], original["dataset"]
      row = {
        "dataset": source,
        "source_id": original["source_id"],
        "clip_id": original["source_id"],
        "source_group": original["source_group"],
        "family_id": original["source_id"],
        "split": split,
        "condition_id": 0,
        "frame50_start": int(original["frame50_start"]),
        "target_yaw_amplitude_deg": None,
        "actual_waist_delta_rad": None,
        "actual_pelvis_delta_rad": None,
        "parent_shard_sha256": approved[path.name]["sha256"],
        "parent_index": index,
        "mirrored": original["mirrored"] == "True",
      }
      buckets[split]["raw"].append(window[None])
      buckets[split]["rows"].append(row)
      legacy_windows[split, source] += 1
  # Reject cross-split exact duplicates, retaining held-out ownership.
  seen, duplicates = {}, Counter()
  arrays = {}
  all_rows = []
  split_groups = {}
  for split in reversed(SPLITS):
    bucket = buckets[split]
    raw = np.concatenate(bucket["raw"])
    if raw.shape != (len(bucket["rows"]), 10, 59) or not np.isfinite(raw).all():
      raise ValueError("Invalid mixed raw features")
    keep = []
    for i, row in enumerate(bucket["rows"]):
      digest = hashlib.sha256(raw[i].tobytes()).hexdigest()
      if digest in seen and seen[digest] != split:
        duplicates[split] += 1
        continue
      seen[digest] = split
      group = row["source_group"]
      if group in split_groups and split_groups[group] != split:
        raise ValueError(f"Source actor/family crosses splits: {group}")
      split_groups[group] = split
      keep.append(i)
    rows = [bucket["rows"][i] for i in keep]
    raw = raw[keep]
    clip_keys = sorted({r["clip_id"] for r in rows})
    clips = {c: i for i, c in enumerate(clip_keys)}
    arrays[split] = (raw, rows, clips)
  train_raw, train_rows, _ = arrays["train"]
  class_weights = [condition_weights(train_rows, c) for c in range(3)]
  weights = sum(
    w * mass for w, mass in zip(class_weights, (0.2, 0.4, 0.4), strict=True)
  )
  low, high, original_quantiles = normalization(train_raw, weights)
  norm_path = root / "norm_stats.npz"
  np.savez(
    norm_path,
    q_low=low,
    q_high=high,
    min_span=MIN_SPAN,
    original_quantiles=original_quantiles,
    schema=np.str_(NORM_SCHEMA),
  )
  shards = []
  for split in SPLITS:
    raw, rows, clips = arrays[split]
    labels = np.array([int(r["condition_id"]) for r in rows], dtype=np.int64)
    path = root / f"{split}.npz"
    np.savez_compressed(
      path,
      windows=raw,
      condition_id=labels,
      clip_index=np.array([clips[r["clip_id"]] for r in rows]),
      frame50_start=np.array([r["frame50_start"] for r in rows]),
    )
    row_path = root / f"{split}.rows.json"
    atomic_json(row_path, rows)
    all_rows.extend(rows)
    shards.append(
      {
        "split": split,
        "path": path.name,
        "sha256": sha256(path),
        "rows": row_path.name,
        "rows_sha256": sha256(row_path),
        "windows": len(rows),
        "clips": len(clips),
        "condition_counts": np.bincount(labels, minlength=3).tolist(),
        "source_counts": dict(Counter(r["dataset"] for r in rows)),
      }
    )
  with (root / "windows.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
  atomic_json(root / "legacy_rejected.json", rejections)
  atomic_json(
    root / "normalization_report.json",
    {
      "schema": NORM_SCHEMA,
      "train_only": True,
      "source_clip_weighted": True,
      "original_spans": (original_quantiles[1] - original_quantiles[0]).tolist(),
      "effective_spans": (high - low).tolist(),
      "floored_features": np.flatnonzero(
        original_quantiles[1] - original_quantiles[0] < MIN_SPAN
      ).tolist(),
      "training_and_runtime_input_clipping": False,
    },
  )
  manifest = {
    "experiment": "EXP-035",
    "status": "complete",
    "gate_passed": True,
    "identity": identity,
    "condition_schema": CONDITION_SCHEMA,
    "feature_contract": "smp_g1_last_heading_50hz_10x59_mixed_v1",
    "norm_stats_sha256": sha256(norm_path),
    "shards": shards,
    "windows_index_sha256": sha256(root / "windows.csv"),
    "legacy_fk_only_not_dynamic_validation": True,
    "legacy_rejected": dict(Counter(r["reason"] for r in rejections)),
    "cross_split_exact_duplicates_removed": dict(duplicates),
    "independent_sample_unit": "original clip; personal participant count remains three",
  }
  atomic_json(root / "manifest.json", manifest)
  for split in SPLITS:
    dataset = ShoulderMixedDataset(root, split)
    if split == "train":
      ShoulderMixedSampler(dataset)
  return manifest


class ShoulderMixedDataset(Dataset):
  def __init__(self, root=DATA_DIR, split="train", expected_sha=None):
    self.root, self.split = Path(root), split
    self.manifest_sha256 = data_sha(root)
    if expected_sha and self.manifest_sha256 != expected_sha:
      raise ValueError("Mixed manifest checksum mismatch")
    self.manifest = json.loads((self.root / "manifest.json").read_text())
    m = self.manifest
    if not m["gate_passed"] or m["condition_schema"] != CONDITION_SCHEMA:
      raise ValueError("Unaccepted mixed schema")
    item = next(s for s in m["shards"] if s["split"] == split)
    _verify(self.root / item["path"], item["sha256"])
    _verify(self.root / item["rows"], item["rows_sha256"])
    _verify(self.root / "norm_stats.npz", m["norm_stats_sha256"])
    self.rows = json.loads((self.root / item["rows"]).read_text())
    with np.load(self.root / item["path"], allow_pickle=False) as z:
      self.raw = torch.from_numpy(z["windows"].copy())
      self.labels = torch.from_numpy(z["condition_id"].astype(np.int64))
      self.clip_indices = torch.from_numpy(z["clip_index"].astype(np.int64))
      self.frame_starts = torch.from_numpy(z["frame50_start"].astype(np.int64))
    if (
      self.raw.shape != (len(self.rows), 10, 59)
      or not np.isin(self.labels, [0, 1, 2]).all()
    ):
      raise ValueError("Invalid mixed shard shape/labels")
    for i, row in enumerate(self.rows):
      if row["split"] != split or int(row["condition_id"]) != int(self.labels[i]):
        raise ValueError("Mixed row/shard identity mismatch")
      if row["dataset"] not in SOURCES or (
        int(self.labels[i]) and row["dataset"] not in ("ik", "personal")
      ):
        raise ValueError("Forbidden mixed source/label")
    with np.load(self.root / "norm_stats.npz", allow_pickle=False) as z:
      self.q_low, self.q_high = z["q_low"].copy(), z["q_high"].copy()
    if not np.isfinite([self.q_low, self.q_high]).all() or np.any(
      self.q_high - self.q_low < MIN_SPAN - 1e-6
    ):
      raise ValueError("Invalid robust normalization spans")
    self.windows = (
      2
      * (self.raw - torch.from_numpy(self.q_low))
      / torch.from_numpy(self.q_high - self.q_low)
      - 1
    )
    if not torch.isfinite(self.windows).all():
      raise ValueError("Nonfinite normalized data")
    self.source_ids = torch.tensor([SOURCES.index(r["dataset"]) for r in self.rows])
    self.target_amplitudes_deg = torch.tensor(
      [r["target_yaw_amplitude_deg"] or 0 for r in self.rows]
    )
    self._weights = [
      torch.from_numpy(condition_weights(self.rows, c)) for c in range(3)
    ]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, i):
    return self.windows[i], self.labels[i]

  def clip_equal_weights(self, condition):
    return self._weights[condition].clone()

  def sample_indices(self, condition, count, seed=42, source=None):
    weights = self._weights[condition].clone()
    if source is not None:
      weights *= self.source_ids == SOURCES.index(source)
    if weights.sum() <= 0:
      raise ValueError("Empty conditional/source sample pool")
    return torch.multinomial(
      weights, count, replacement=True, generator=torch.Generator().manual_seed(seed)
    )

  def mirror_indices(self):
    """IK uses its independently solved pair; personal uses real opposite clips.

    Personal takes have different durations; match normalized window progress
    within a seeded opposite clip rather than inventing mirrored joint angles.
    """
    result = torch.arange(len(self))
    ik_lookup = {
      (r["family_id"], int(r["condition_id"]), r["frame50_start"]): i
      for i, r in enumerate(self.rows)
      if r["dataset"] == "ik"
    }
    generator = np.random.default_rng(42)
    personal = {}
    for c in (1, 2):
      groups = {}
      for i, row in enumerate(self.rows):
        if row["dataset"] == "personal" and int(row["condition_id"]) == c:
          groups.setdefault(row["clip_id"], []).append(i)
      personal[c] = [
        sorted(ids, key=lambda j: self.rows[j]["frame50_start"])
        for _, ids in sorted(groups.items())
      ]
    personal_match = {}
    for c in (1, 2):
      for ids in personal[c]:
        opposite = personal[3 - c][generator.integers(len(personal[3 - c]))]
        for j, i in enumerate(ids):
          personal_match[i] = opposite[
            round(j * (len(opposite) - 1) / max(len(ids) - 1, 1))
          ]
    for i, row in enumerate(self.rows):
      c = int(row["condition_id"])
      if not c:
        continue
      if row["dataset"] == "ik":
        key = row["family_id"], 3 - c, row["frame50_start"]
        if key not in ik_lookup:
          raise ValueError("Missing independent IK opposite window")
        result[i] = ik_lookup[key]
      else:
        result[i] = personal_match[i]
    return result
