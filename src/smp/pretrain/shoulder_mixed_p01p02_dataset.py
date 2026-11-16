"""EXP-036: train on P01 plus P02, holding out complete P02 takes."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from smp.data.shoulder_ik import atomic_json
from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_mixed import (
  CONDITION_SCHEMA,
  MIN_SPAN,
  NORM_SCHEMA,
  SAMPLING,
  ShoulderMixedSampler,
  condition_weights,
  normalization,
)
from smp.pretrain.shoulder_mixed_dataset import (
  ShoulderMixedDataset,
  _verify,
)

DATA_DIR = Path("logs/reproduction/g1_shoulder_mixed_p01p02_prior_v1/data")
PARENT = Path("logs/reproduction/g1_shoulder_mixed_prior_ppo_v1/data")
PARENT_SHA = "3504b7222e50e08f15486b52aa77c000d47cd7b618c83e4becea99684af9a801"
PERSONAL = Path("logs/reproduction/g1_personal_retarget_v1/dataset")
PERSONAL_SHA = "616964e09b235db14e69a68db68d0d2224080020fbc78f9c514cb9f1c08157bf"
SPLITS = ("train", "validation", "test")
POLICY = {
  "version": 1,
  "name": "p01_p02_train_p02_complete_take_holdout",
  "seed": 42,
  "holdout": "one accepted P02 take per condition, sorted IDs and NumPy PCG64",
  "personal_pool_clips": 17,
  "personal_training_clips": 14,
  "personal_validation_clips": 3,
  "p03_role": "reserved independent subject; not evaluated by training driver",
  "validation_scope": "unseen takes from an already covered participant",
  "cross_subject_generalization_claim": False,
  "model_or_score_metrics_used_for_split": False,
}


def data_sha(root=DATA_DIR):
  return sha256(Path(root) / "manifest.json")


def split_personal_clips(clips, seed=42):
  """Choose whole videos before window sampling; never choose by model scores."""
  by_id = {c["clip_id"]: c for c in clips}
  if len(by_id) != len(clips):
    raise ValueError("Duplicate personal clip IDs")
  counts = Counter(c["subject_id"] for c in clips)
  if counts != {"P01": 9, "P02": 8, "P03": 9}:
    raise ValueError(f"Unexpected accepted personal inventory: {counts}")
  generator = np.random.default_rng(seed)
  held = set()
  for condition in range(3):
    pool = sorted(
      c["clip_id"]
      for c in clips
      if c["subject_id"] == "P02" and c["condition_id"] == condition
    )
    if len(pool) < 2:
      raise ValueError("P02 needs both training and held-out takes per condition")
    held.add(pool[int(generator.integers(len(pool)))])
  records = []
  for clip_id, clip in sorted(by_id.items()):
    subject = clip["subject_id"]
    split = "test" if subject == "P03" else "validation" if clip_id in held else "train"
    records.append(
      {
        "clip_id": clip_id,
        "subject_id": subject,
        "condition_id": int(clip["condition_id"]),
        "original_split": clip["split"],
        "split": split,
        "source_video_sha256": clip["source_identity"]["video_sha256"],
      }
    )
  validate_split_records(records)
  return records


def validate_split_records(records):
  videos, clips, coverage = {}, {}, Counter()
  for row in records:
    clip, video, split = row["clip_id"], row["source_video_sha256"], row["split"]
    if clip in clips and clips[clip] != split:
      raise ValueError("Personal clip crosses splits")
    if video in videos and videos[video] != split:
      raise ValueError("Original video crosses splits")
    clips[clip], videos[video] = split, split
    if row["subject_id"] == "P03" and split != "test":
      raise ValueError("P03 must remain independently reserved")
    if split == "train":
      coverage[row["subject_id"], int(row["condition_id"])] += 1
  if any(coverage[s, c] < 1 for s in ("P01", "P02") for c in range(3)):
    raise ValueError("Training must cover both participants and every condition")
  count = Counter(r["split"] for r in records)
  if count != {"train": 14, "validation": 3, "test": 9}:
    raise ValueError(f"Unexpected complete-clip quota: {count}")


def prepare(root=DATA_DIR):
  root = Path(root)
  _verify(PARENT / "manifest.json", PARENT_SHA)
  _verify(PERSONAL / "manifest.json", PERSONAL_SHA)
  parent = json.loads((PARENT / "manifest.json").read_text())
  personal = json.loads((PERSONAL / "manifest.json").read_text())
  _verify(PERSONAL / personal["clip_table"], personal["clip_table_sha256"])
  clips = json.loads((PERSONAL / personal["clip_table"]).read_text())
  allocation = split_personal_clips(clips)
  mapping = {r["clip_id"]: r for r in allocation}
  code_paths = [
    Path(__file__),
    Path("src/smp/pretrain/shoulder_mixed.py"),
    Path("src/smp/pretrain/shoulder_mixed_dataset.py"),
  ]
  identity = {
    "experiment": "EXP-036",
    "parent_manifest_sha256": PARENT_SHA,
    "personal_manifest_sha256": PERSONAL_SHA,
    "policy": POLICY,
    "condition_schema": CONDITION_SCHEMA,
    "sampling": SAMPLING,
    "normalization_schema": NORM_SCHEMA,
    "min_span": MIN_SPAN.tolist(),
    "code": {str(p.resolve()): sha256(p) for p in code_paths},
  }
  manifest_path = root / "manifest.json"
  if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    if existing["identity"] != identity:
      raise ValueError("Existing P01/P02 dataset has a different identity")
    for split in SPLITS:
      loaded = ShoulderMixedP01P02Dataset(root, split)
      if split == "train":
        ShoulderMixedSampler(loaded)
    return existing
  root.mkdir(parents=True, exist_ok=True)
  raw_parts, row_parts = [], []
  for item in parent["shards"]:
    _verify(PARENT / item["path"], item["sha256"])
    _verify(PARENT / item["rows"], item["rows_sha256"])
    rows = json.loads((PARENT / item["rows"]).read_text())
    with np.load(PARENT / item["path"], allow_pickle=False) as z:
      raw = z["windows"].copy()
      labels = z["condition_id"]
      if raw.shape != (len(rows), 10, 59):
        raise ValueError("Parent rows and windows disagree")
      for index, old in enumerate(rows):
        row = dict(old)
        if int(row["condition_id"]) != int(labels[index]):
          raise ValueError("Parent label integrity mismatch")
        row["parent_mixed_split"] = item["split"]
        row["parent_mixed_shard_sha256"] = item["sha256"]
        row["parent_mixed_index"] = index
        row["subject_id"] = ""
        row["source_video_sha256"] = ""
        if row["dataset"] == "personal":
          clip_id = row["clip_id"].removeprefix("personal/")
          chosen = mapping[clip_id]
          row.update(
            split=chosen["split"],
            subject_id=chosen["subject_id"],
            source_video_sha256=chosen["source_video_sha256"],
            split_group_id="personal_video/" + chosen["source_video_sha256"],
          )
        else:
          row["split_group_id"] = row["source_group"]
        row_parts.append(row)
      raw_parts.append(raw)
  raw = np.concatenate(raw_parts)
  if not np.isfinite(raw).all():
    raise ValueError("Nonfinite parent raw features")
  buckets, seen, removed, group_owners = {}, {}, Counter(), {}
  # Preserve independent held-out ownership if identical windows appear after resplitting.
  for split in reversed(SPLITS):
    ids = []
    rows = []
    for i, row in enumerate(row_parts):
      if row["split"] != split:
        continue
      group = row["split_group_id"]
      if group in group_owners and group_owners[group] != split:
        raise ValueError(f"Original source group crosses splits: {group}")
      group_owners[group] = split
      digest = hashlib.sha256(raw[i].tobytes()).hexdigest()
      if digest in seen and seen[digest] != split:
        removed[split] += 1
        continue
      seen[digest] = split
      ids.append(i)
      rows.append(row)
    buckets[split] = (raw[ids], rows)
  training, train_rows = buckets["train"]
  probabilities = sum(
    mass * condition_weights(train_rows, c) for c, mass in enumerate((0.2, 0.4, 0.4))
  )
  low, high, quantiles = normalization(training, probabilities)
  norm = root / "norm_stats.npz"
  np.savez(
    norm,
    q_low=low,
    q_high=high,
    min_span=MIN_SPAN,
    original_quantiles=quantiles,
    schema=np.str_(NORM_SCHEMA),
  )
  shards, all_rows = [], []
  for split in SPLITS:
    values, rows = buckets[split]
    clip_ids = {
      key: index for index, key in enumerate(sorted({r["clip_id"] for r in rows}))
    }
    labels = np.array([int(r["condition_id"]) for r in rows], dtype=np.int64)
    path = root / f"{split}.npz"
    np.savez_compressed(
      path,
      windows=values,
      condition_id=labels,
      clip_index=np.array([clip_ids[r["clip_id"]] for r in rows]),
      frame50_start=np.array([r["frame50_start"] for r in rows]),
    )
    row_path = root / f"{split}.rows.json"
    atomic_json(row_path, rows)
    shards.append(
      {
        "split": split,
        "path": path.name,
        "sha256": sha256(path),
        "rows": row_path.name,
        "rows_sha256": sha256(row_path),
        "windows": len(rows),
        "condition_counts": np.bincount(labels, minlength=3).tolist(),
        "source_counts": dict(Counter(r["dataset"] for r in rows)),
        "personal_clip_count": len(
          {r["clip_id"] for r in rows if r["dataset"] == "personal"}
        ),
        "personal_subject_window_counts": dict(
          Counter(r["subject_id"] for r in rows if r["dataset"] == "personal")
        ),
      }
    )
    all_rows.extend(rows)
  with (root / "windows.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
  atomic_json(root / "personal_split.json", {"policy": POLICY, "clips": allocation})
  atomic_json(
    root / "normalization_report.json",
    {
      "schema": NORM_SCHEMA,
      "train_only": True,
      "original_spans": (quantiles[1] - quantiles[0]).tolist(),
      "effective_spans": (high - low).tolist(),
      "floored_features": np.flatnonzero(
        quantiles[1] - quantiles[0] < MIN_SPAN
      ).tolist(),
      "old_normalization_reused": False,
    },
  )
  manifest = {
    "experiment": "EXP-036",
    "status": "complete",
    "gate_passed": True,
    "identity": identity,
    "condition_schema": CONDITION_SCHEMA,
    "feature_contract": parent["feature_contract"],
    "shards": shards,
    "norm_stats_sha256": sha256(norm),
    "windows_index_sha256": sha256(root / "windows.csv"),
    "personal_split_sha256": sha256(root / "personal_split.json"),
    "cross_split_exact_duplicates_removed": dict(removed),
    "personal_validation_scope": POLICY["validation_scope"],
    "p03_model_evaluation_performed": False,
    "score_formula_changed": False,
    "raw_features_changed": False,
  }
  atomic_json(manifest_path, manifest)
  for split in SPLITS:
    loaded = ShoulderMixedP01P02Dataset(root, split)
    if split == "train":
      ShoulderMixedSampler(loaded)
    # Every opposite comparison remains a real opposite-direction clip.
    opposite = loaded.mirror_indices()
    active = loaded.labels > 0
    if not np.array_equal(
      loaded.labels[opposite[active]].numpy(), (3 - loaded.labels[active]).numpy()
    ):
      raise ValueError("Opposite-motion pairing failed")
  return manifest


class ShoulderMixedP01P02Dataset(ShoulderMixedDataset):
  def __init__(self, root=DATA_DIR, split="train", expected_sha=None):
    header = json.loads((Path(root) / "manifest.json").read_text())
    if header.get("identity", {}).get("policy") != POLICY:
      raise ValueError("Personal clip-split policy identity mismatch")
    super().__init__(root, split, expected_sha)
    _verify(self.root / "personal_split.json", self.manifest["personal_split_sha256"])
    records = json.loads((self.root / "personal_split.json").read_text())["clips"]
    validate_split_records(records)
    allocation = {"personal/" + r["clip_id"]: r for r in records}
    for row in self.rows:
      if row["dataset"] == "personal":
        expected = allocation[row["clip_id"]]
        if (
          row["split"] != expected["split"]
          or row["subject_id"] != expected["subject_id"]
        ):
          raise ValueError("Personal row does not match frozen complete-video split")
