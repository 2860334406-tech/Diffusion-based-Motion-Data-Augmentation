"""Finalize accepted G1 shoulder-IK executions into audited SMP windows."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from smp.data.shoulder_ik import (
  EE_BODY_NAMES,
  FPS,
  JOINT_NAMES,
  WINDOW_SIZE,
  WINDOW_STRIDE,
  _build_model,
  _qpos_ids,
  atomic_json,
  classify_condition_windows,
  sha256,
)

FEATURE_DIMS = (3, 6, len(JOINT_NAMES), 3 * len(EE_BODY_NAMES), 3, 3)


def _angular_velocity(quaternions_xyzw: np.ndarray) -> np.ndarray:
  rotations = Rotation.from_quat(quaternions_xyzw)
  velocity = np.zeros((len(rotations), 3), dtype=np.float64)
  relative = rotations[2:] * rotations[:-2].inv()
  velocity[1:-1] = relative.as_rotvec() / (2.0 / FPS)
  velocity[0] = velocity[1]
  velocity[-1] = velocity[-2]
  return velocity


def compute_window_features(
  root_pos: np.ndarray,
  root_rot_xyzw: np.ndarray,
  dof_pos: np.ndarray,
  ee_pos_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Compute the repository's last-frame-heading `(N,10,59)` representation."""
  frames = len(root_pos)
  expected = ((frames, 4), (frames, len(JOINT_NAMES)), (frames, len(EE_BODY_NAMES), 3))
  actual = (root_rot_xyzw.shape, dof_pos.shape, ee_pos_world.shape)
  if actual != expected or frames < WINDOW_SIZE:
    raise ValueError(f"Invalid trajectory shapes: {actual}, expected {expected}")
  arrays = (root_pos, root_rot_xyzw, dof_pos, ee_pos_world)
  if not all(np.isfinite(array).all() for array in arrays):
    raise ValueError("Nonfinite executed trajectory")
  norms = np.linalg.norm(root_rot_xyzw, axis=1)
  if not np.allclose(norms, 1.0, atol=1e-3):
    raise ValueError("Root quaternions are not unit length")

  starts = np.arange(0, frames - WINDOW_SIZE + 1, WINDOW_STRIDE)
  indices = starts[:, None] + np.arange(WINDOW_SIZE)
  rotations = Rotation.from_quat(root_rot_xyzw)
  rotation_matrices = rotations.as_matrix()
  yaw = rotations.as_euler("xyz")[:, 2]
  heading_yaw = -yaw[indices[:, -1]]
  heading_inv = Rotation.from_rotvec(
    np.column_stack(
      (np.zeros_like(heading_yaw), np.zeros_like(heading_yaw), heading_yaw)
    )
  ).as_matrix()

  root_offsets = root_pos[indices] - root_pos[indices[:, -1], None]
  root_local = np.einsum("nij,nwj->nwi", heading_inv, root_offsets)
  root_local[..., 2] = root_pos[indices, 2]

  root_rotation_local = np.einsum(
    "nij,nwjk->nwik", heading_inv, rotation_matrices[indices]
  )
  root_rotation_6d = np.concatenate(
    (root_rotation_local[..., :, 0], root_rotation_local[..., :, 2]), axis=-1
  )

  ee_offsets = ee_pos_world[indices] - root_pos[indices, None, :]
  ee_local = np.einsum("nij,nwej->nwei", heading_inv, ee_offsets).reshape(
    len(starts), WINDOW_SIZE, -1
  )

  linear_velocity = np.gradient(root_pos, 1.0 / FPS, axis=0)
  angular_velocity = _angular_velocity(root_rot_xyzw)
  linear_local = np.einsum("nij,nwj->nwi", heading_inv, linear_velocity[indices])
  angular_local = np.einsum("nij,nwj->nwi", heading_inv, angular_velocity[indices])
  windows = np.concatenate(
    (
      root_local,
      root_rotation_6d,
      dof_pos[indices],
      ee_local,
      linear_local,
      angular_local,
    ),
    axis=-1,
  ).astype(np.float32)
  if windows.shape != (len(starts), WINDOW_SIZE, 59):
    raise AssertionError(windows.shape)
  return starts, windows


class ExecutedFk:
  def __init__(self):
    self.model = _build_model(soft_limits=False)
    self.data = mujoco.MjData(self.model)
    self.joint_qpos = _qpos_ids(self.model)
    self.body_ids = [self.model.body(name).id for name in EE_BODY_NAMES]

  def positions(
    self, root_pos: np.ndarray, root_rot_xyzw: np.ndarray, dof_pos: np.ndarray
  ) -> np.ndarray:
    result = np.empty((len(root_pos), len(EE_BODY_NAMES), 3), dtype=np.float64)
    for index in range(len(root_pos)):
      self.data.qpos[:3] = root_pos[index]
      self.data.qpos[3:7] = root_rot_xyzw[index, [3, 0, 1, 2]]
      self.data.qpos[self.joint_qpos] = dof_pos[index]
      self.data.qvel[:] = 0.0
      mujoco.mj_forward(self.model, self.data)
      result[index] = self.data.xpos[self.body_ids]
    return result


def _load_execution(output: Path, record: dict[str, Any]):
  metadata_path = output / record["accepted"]["metadata"]
  metadata = json.loads(metadata_path.read_text())
  execution_path = metadata_path.with_suffix(".npz")
  if sha256(execution_path) != metadata["execution_npz_sha256"]:
    raise ValueError(f"Execution identity mismatch: {execution_path}")
  with np.load(execution_path, allow_pickle=False) as data:
    root_pos = data["root_pos"].astype(np.float64)
    root_rot = data["root_rot"].astype(np.float64)
    dof_pos = data["dof_pos"].astype(np.float64)
  if (
    root_pos.shape != (metadata["frames"], 3)
    or root_rot.shape != (metadata["frames"], 4)
    or dof_pos.shape != (metadata["frames"], len(JOINT_NAMES))
  ):
    raise ValueError(f"Execution contract mismatch: {execution_path}")
  return metadata_path, metadata, root_pos, root_rot, dof_pos


def finalize_dataset(output_dir: str | Path) -> dict[str, Any]:
  output = Path(output_dir)
  source_manifest_path = output / "manifest.json"
  source_manifest = json.loads(source_manifest_path.read_text())
  if source_manifest.get("status") != "complete" or not source_manifest.get(
    "gate_passed"
  ):
    raise RuntimeError("Only a complete, fully gated generation may be finalized")
  stage = source_manifest["identity"]["stage"]
  expected_clips = 240 if stage == "pilot" else 2400
  records = [source_manifest["slots"][key] for key in sorted(source_manifest["slots"])]
  if len(records) != expected_clips or any(r["status"] != "accepted" for r in records):
    raise RuntimeError("Accepted clip inventory does not match the frozen stage quota")

  dataset_dir = output / "dataset"
  dataset_dir.mkdir(parents=True, exist_ok=True)
  identity = {
    "schema_version": 3,
    "source_manifest_sha256": sha256(source_manifest_path),
    "feature_contract": "smp_g1_window_10x59_continuous_yaw_v3",
    "fps": FPS,
    "window_size": WINDOW_SIZE,
    "stride": WINDOW_STRIDE,
    "feature_dims": list(FEATURE_DIMS),
    "normalization_source_split": "pilot" if stage == "pilot" else "train",
  }
  dataset_manifest_path = dataset_dir / "manifest.json"
  if dataset_manifest_path.exists():
    old = json.loads(dataset_manifest_path.read_text())
    if old.get("identity") != identity:
      raise RuntimeError("Existing finalized dataset has a different identity")

  fk = ExecutedFk()
  buckets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
    lambda: defaultdict(list)
  )
  rows: list[dict[str, Any]] = []
  clip_table: list[dict[str, Any]] = []
  clip_lookup: dict[str, int] = {}
  for record in records:
    metadata_path, metadata, root, quat, joints = _load_execution(output, record)
    spec = metadata["spec"]
    clip_id = spec["clip_id"]
    clip_index = len(clip_table)
    clip_lookup[clip_id] = clip_index
    clip_table.append(
      {
        "clip_index": clip_index,
        "clip_id": clip_id,
        "family_id": spec["family_id"],
        "split": spec["split"],
        "clip_condition_id": spec["condition_id"],
        "target_yaw_amplitude_deg": float(
          metadata["metrics"]["target_waist_amplitude_deg"]
        ),
        "actual_waist_amplitude_deg": float(
          metadata["metrics"]["actual_waist_amplitude_deg"]
        ),
        "target_mean_yaw_speed_deg_s": float(
          spec.get("target_mean_yaw_speed_deg_s", 0.0)
        ),
        "entry_duration_s": float(spec["entry_duration_s"]),
        "metadata": str(metadata_path.relative_to(output)),
        "metadata_sha256": sha256(metadata_path),
      }
    )
    motion36 = np.concatenate((root, quat, joints), axis=1)
    starts, labels = classify_condition_windows(motion36, spec["condition_id"])
    feature_starts, features = compute_window_features(
      root, quat, joints, fk.positions(root, quat, joints)
    )
    if not np.array_equal(starts, feature_starts):
      raise AssertionError("Feature and phase window starts diverged")
    keep = labels >= 0
    split = spec["split"]
    buckets[split]["windows"].append(features[keep])
    buckets[split]["condition_id"].append(labels[keep].astype(np.int64))
    buckets[split]["clip_index"].append(
      np.full(int(keep.sum()), clip_index, dtype=np.int32)
    )
    buckets[split]["frame50_start"].append(starts[keep].astype(np.int32))
    kept_count = int(keep.sum())
    target_amplitude = float(metadata["metrics"]["target_waist_amplitude_deg"] or 0.0)
    actual_amplitude = float(metadata["metrics"]["actual_waist_amplitude_deg"] or 0.0)
    target_speed = float(spec.get("target_mean_yaw_speed_deg_s", 0.0))
    buckets[split]["target_yaw_amplitude_deg"].append(
      np.full(kept_count, target_amplitude, dtype=np.float32)
    )
    buckets[split]["actual_waist_amplitude_deg"].append(
      np.full(kept_count, actual_amplitude, dtype=np.float32)
    )
    buckets[split]["target_mean_yaw_speed_deg_s"].append(
      np.full(kept_count, target_speed, dtype=np.float32)
    )
    buckets[split]["clip_balanced_weight"].append(
      np.full(kept_count, 1.0 / kept_count, dtype=np.float32)
    )
    for local_index, (start, condition) in enumerate(
      zip(starts[keep], labels[keep], strict=True)
    ):
      rows.append(
        {
          "split": split,
          "clip_index": clip_index,
          "clip_id": clip_id,
          "family_id": spec["family_id"],
          "clip_condition_id": spec["condition_id"],
          "frame50_start": int(start),
          "frame50_end": int(start + WINDOW_SIZE),
          "condition_id": int(condition),
          "target_yaw_amplitude_deg": target_amplitude,
          "actual_waist_amplitude_deg": actual_amplitude,
          "target_mean_yaw_speed_deg_s": target_speed,
          "clip_balanced_weight": 1.0 / kept_count,
          "index_in_clip": local_index,
        }
      )

  shard_records = []
  split_arrays: dict[str, dict[str, np.ndarray]] = {}
  condition_counts: dict[str, list[int]] = {}
  for split in sorted(buckets):
    arrays = {key: np.concatenate(values) for key, values in buckets[split].items()}
    if arrays["windows"].shape[1:] != (WINDOW_SIZE, 59):
      raise AssertionError(arrays["windows"].shape)
    if not np.isfinite(arrays["windows"]).all():
      raise ValueError(f"Nonfinite finalized windows in {split}")
    split_arrays[split] = arrays
    path = dataset_dir / f"{split}.npz"
    np.savez_compressed(
      path,
      **arrays,
      fps=np.asarray(FPS, dtype=np.float32),
      window_size=np.asarray(WINDOW_SIZE, dtype=np.int32),
      stride=np.asarray(WINDOW_STRIDE, dtype=np.int32),
      feature_dims=np.asarray(FEATURE_DIMS, dtype=np.int32),
      ee_body_names=np.asarray(EE_BODY_NAMES),
    )
    counts = np.bincount(arrays["condition_id"], minlength=3).tolist()
    condition_counts[split] = counts
    shard_records.append(
      {
        "path": path.name,
        "split": split,
        "windows": len(arrays["windows"]),
        "condition_counts": counts,
        "sha256": sha256(path),
      }
    )

  norm_split = identity["normalization_source_split"]
  flattened = split_arrays[norm_split]["windows"].reshape(-1, 59)
  q_low, q_high = np.percentile(flattened, [1, 99], axis=0).astype(np.float32)
  q_high = np.where(q_high - q_low < 1e-6, q_low + 1.0, q_high)
  norm_path = dataset_dir / "norm_stats.npz"
  np.savez(norm_path, q_low=q_low, q_high=q_high)

  index_path = dataset_dir / "windows.csv"
  with index_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  clip_table_path = dataset_dir / "clips.json"
  atomic_json(clip_table_path, clip_table)

  required_splits = ("pilot",) if stage == "pilot" else ("train", "validation", "test")
  coverage_passed = all(min(condition_counts[split]) > 0 for split in required_splits)
  yaw_min_deg, yaw_max_deg = source_manifest["identity"]["sampling"][
    "yaw_amplitude_deg"
  ]
  amplitude_edges = np.arange(yaw_min_deg, yaw_max_deg + 5.0, 5.0)
  amplitude_edges[-1] = yaw_max_deg
  window_amplitude_distribution = {}
  for split, arrays in sorted(split_arrays.items()):
    for condition, side in ((1, "left_shoulder"), (2, "right_shoulder")):
      mask = arrays["condition_id"] == condition
      values = arrays["actual_waist_amplitude_deg"][mask]
      counts, _ = np.histogram(values, bins=amplitude_edges)
      window_amplitude_distribution[f"{split}/{side}"] = {
        "windows": int(mask.sum()),
        "bin_edges_deg": amplitude_edges.tolist(),
        "bin_counts": counts.tolist(),
        "min_deg": float(values.min()) if len(values) else None,
        "max_deg": float(values.max()) if len(values) else None,
      }

  manifest = {
    "identity": identity,
    "status": "complete" if coverage_passed else "blocked",
    "gate_passed": coverage_passed,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "clips": expected_clips,
    "clip_table": clip_table_path.name,
    "clip_table_sha256": sha256(clip_table_path),
    "windows_index": index_path.name,
    "windows_index_sha256": sha256(index_path),
    "norm_stats": norm_path.name,
    "norm_stats_sha256": sha256(norm_path),
    "condition_counts": condition_counts,
    "window_weighted_amplitude_distribution": window_amplitude_distribution,
    "clip_balanced_sampling_note": (
      "clip_balanced_weight sums to one per source clip; use it or an equivalent "
      "clip-first sampler to avoid longer high-angle trajectories dominating training."
    ),
    "shards": shard_records,
    "normalization_note": (
      "Pilot-only diagnostic statistics; do not use for formal training."
      if stage == "pilot"
      else "Computed only from the formal training split."
    ),
  }
  atomic_json(dataset_manifest_path, manifest)
  if not coverage_passed:
    raise RuntimeError("Finalized windows do not cover every required condition")
  return manifest
