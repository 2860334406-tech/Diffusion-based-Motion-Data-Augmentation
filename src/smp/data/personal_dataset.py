"""Export physics-approved personal clips without reusing an old Prior norm."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from smp.data.personal_retarget import (
  CONDITIONS,
  SCHEMA,
  SUBJECT_SPLITS,
  coverage,
  verify_attempt,
)
from smp.data.shoulder_ik import FPS, atomic_json, sha256
from smp.data.shoulder_ik_dataset import ExecutedFk, compute_window_features


def window_labels(root, quat, joints, condition, onset):
  starts = np.arange(0, len(root) - 9, 5)
  labels = np.full(len(starts), -1, dtype=np.int64)
  if condition == 0:
    labels[:] = 0
  else:
    labels[starts + 10 <= onset] = 0
    labels[starts >= onset] = condition
  # An initial shoulder recording is stand only if its measured motion is neutral.
  velocity = np.gradient(root, 1 / FPS, axis=0)
  index = starts[:, None] + np.arange(10)
  neutral = (np.abs(joints[index, 12]).max(axis=1) <= 0.06) & (
    np.linalg.norm(velocity[index, :2], axis=-1).max(axis=1) <= 0.08
  )
  labels[(labels == 0) & ~neutral] = -1
  return starts, labels


def publication_gate(records: dict, review: dict) -> dict:
  cover = coverage(records)
  accepted = [
    clip for clip, record in records.items() if record["status"] == "accepted"
  ]
  reviewed = all(
    review.get("clips", {}).get(clip, {}).get("decision") == "pass" for clip in accepted
  )
  return {
    **cover,
    "all_accepted_clips_reviewed": reviewed,
    "passed": bool(len(records) == 27 and cover["passed"] and reviewed),
  }


def export_dataset(output: Path) -> dict:
  manifest = json.loads((output / "manifest.json").read_text())
  review_path = output / "replay" / "review.json"
  review = json.loads(review_path.read_text()) if review_path.exists() else {}
  if review.get("identity_sha256") != manifest["identity_sha256"]:
    raise ValueError("Visual review identity mismatch or missing review")
  gate = publication_gate(manifest["clips"], review)
  if not gate["passed"]:
    raise RuntimeError(f"Personal dataset publication blocked: {gate}")
  replay_index = json.loads((output / "replay" / "index.json").read_text())
  if replay_index["identity_sha256"] != manifest["identity_sha256"]:
    raise ValueError("Replay identity mismatch")
  for clip_id, record in manifest["clips"].items():
    if record["status"] != "accepted":
      continue
    replay = replay_index["clips"][clip_id]
    if (
      replay["metadata_sha256"] != record["accepted"]["metadata_sha256"]
      or replay["video_sha256"] != sha256(output / replay["video"])
      or review["clips"][clip_id]["video_sha256"] != replay["video_sha256"]
    ):
      raise ValueError(f"Reviewed replay changed: {clip_id}")
  destination = output / "dataset"
  destination.mkdir(exist_ok=True)
  fk = ExecutedFk()
  buckets = defaultdict(lambda: defaultdict(list))
  clips, rows = [], []
  for clip_id, record in sorted(manifest["clips"].items()):
    if record["status"] != "accepted":
      continue
    metadata = verify_attempt(output, record["accepted"])
    with np.load(
      output / metadata["files"]["execution.npz"]["path"], allow_pickle=False
    ) as z:
      root, quat, joints = z["root_pos"], z["root_rot"], z["dof_pos"]
    starts, features = compute_window_features(
      root, quat, joints, fk.positions(root, quat, joints)
    )
    label_starts, labels = window_labels(
      root, quat, joints, metadata["condition_id"], metadata["turn_start_frame"]
    )
    if not np.array_equal(starts, label_starts):
      raise AssertionError("Window starts differ")
    keep = labels >= 0
    if not keep.any() or (
      metadata["condition_id"] and not np.any(labels == metadata["condition_id"])
    ):
      raise RuntimeError(f"Accepted clip has no correctly labeled windows: {clip_id}")
    split = metadata["split"]
    if SUBJECT_SPLITS[metadata["subject_id"]] != split:
      raise ValueError("Subject split leakage")
    clip_index = len(clips)
    count = int(keep.sum())
    bucket = buckets[split]
    bucket["windows"].append(features[keep])
    bucket["condition_id"].append(labels[keep])
    bucket["clip_index"].append(np.full(count, clip_index, dtype=np.int32))
    bucket["frame50_start"].append(starts[keep].astype(np.int32))
    bucket["clip_balanced_weight"].append(np.full(count, 1 / count, dtype=np.float64))
    for field in (
      "actual_waist_delta_rad",
      "actual_pelvis_delta_rad",
      "actual_torso_delta_rad",
    ):
      bucket[field].append(np.full(count, metadata["metrics"][field], dtype=np.float32))
    clips.append(
      {
        "clip_index": clip_index,
        "clip_id": clip_id,
        "subject_id": metadata["subject_id"],
        "split": split,
        "condition_id": metadata["condition_id"],
        "frames": len(root),
        "turn_start_frame": metadata["turn_start_frame"],
        "source_identity": metadata["source_identity"],
        "accepted": record["accepted"],
        "windows": count,
      }
    )
    for start, label in zip(starts[keep], labels[keep], strict=True):
      rows.append(
        {
          "clip_index": clip_index,
          "source_clip_id": clip_id,
          "subject_id": metadata["subject_id"],
          "split": split,
          "condition_id": int(label),
          "frame50_start": int(start),
          "frame50_end_exclusive": int(start + 10),
          "source_time_start_s": float(start / FPS),
          "clip_balanced_weight": 1 / count,
          "actual_waist_delta_rad": metadata["metrics"]["actual_waist_delta_rad"],
          "actual_pelvis_delta_rad": metadata["metrics"]["actual_pelvis_delta_rad"],
          "actual_torso_delta_rad": metadata["metrics"]["actual_torso_delta_rad"],
        }
      )
  shard_records, arrays = [], {}
  for split in SUBJECT_SPLITS.values():
    arrays[split] = {
      key: np.concatenate(value) for key, value in buckets[split].items()
    }
    a = arrays[split]
    if (
      a["windows"].shape != (len(a["condition_id"]), 10, 59)
      or not np.isfinite(a["windows"]).all()
    ):
      raise ValueError("Invalid personal feature shard")
    counts = np.bincount(a["condition_id"], minlength=3).tolist()
    if min(counts) == 0:
      raise RuntimeError(f"Empty personal condition: {split}")
    path = destination / f"{split}.npz"
    np.savez_compressed(
      path, **a, fps=np.float32(FPS), window_size=np.int32(10), stride=np.int32(5)
    )
    shard_records.append(
      {
        "path": path.name,
        "split": split,
        "windows": len(a["windows"]),
        "condition_counts": counts,
        "sha256": sha256(path),
      }
    )
  train = arrays["train"]["windows"].reshape(-1, 59)
  lo, hi = np.quantile(train, [0.01, 0.99], axis=0)
  norm_path = destination / "norm_stats_diagnostic.npz"
  np.savez(
    norm_path,
    q_low=lo.astype(np.float32),
    q_high=hi.astype(np.float32),
    diagnostic_only=np.bool_(True),
  )
  atomic_json(
    destination / "normalization_report.json",
    {
      "source_split": "train",
      "diagnostic_only": True,
      "do_not_load_as_old_prior_normalization": True,
      "ranges": (hi - lo).tolist(),
      "dimensions_below_1e-3": np.flatnonzero(hi - lo < 1e-3).tolist(),
      "note": "Recompute common robust statistics and online score acceptance during a separately authorized mixed-Prior experiment.",
    },
  )
  atomic_json(destination / "clips.json", clips)
  with (destination / "windows.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  atomic_json(
    destination / "physics_snapshot.json",
    {
      "identity_sha256": manifest["identity_sha256"],
      "clips": manifest["clips"],
      "publication_gate": gate,
      "review_sha256": sha256(review_path),
    },
  )
  report = {
    "schema": SCHEMA,
    "status": "accepted",
    "gate_passed": True,
    "identity_sha256": manifest["identity_sha256"],
    "source_manifest_sha256": manifest["identity"]["source_manifest_sha256"],
    "raw_unnormalized_features": True,
    "clips": len(clips),
    "independent_subjects": 3,
    "subject_splits": SUBJECT_SPLITS,
    "condition_names": list(CONDITIONS),
    "null_id": 3,
    "feature_contract": "smp_g1_last_heading_10x59_personal_pelvis_v1",
    "fps": FPS,
    "window_size": 10,
    "stride": 5,
    "phase_labels": False,
    "shards": shard_records,
    "clip_table": "clips.json",
    "clip_table_sha256": sha256(destination / "clips.json"),
    "windows_index": "windows.csv",
    "windows_index_sha256": sha256(destination / "windows.csv"),
    "diagnostic_norm": norm_path.name,
    "diagnostic_norm_sha256": sha256(norm_path),
    "physics_snapshot_sha256": sha256(destination / "physics_snapshot.json"),
    "prior_or_ppo_started": False,
    "sampling_note": "Balance subject, condition and source clip when mixing; overlapping windows are not independent motions.",
  }
  atomic_json(destination / "manifest.json", report)
  return report
