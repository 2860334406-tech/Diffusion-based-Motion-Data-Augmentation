"""Re-mine LAFAN for the independent shoulder80 prior; never launch training.

All thresholds are frozen before curation. A failed data gate is an intentional
stop: candidate shards and rejection counts remain available for inspection.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from scripts.csv_to_npz import _compute_windows
from scripts.curate_conditional_contact import (
  interpolate,
  motion_arrays,
  screening_model,
)
from scripts.curate_lafan_low_speed import mirror_g1_motion

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder80 import (
  CONDITION_NAMES,
  CONDITION_SCHEMA,
  PHASE_NAMES,
  SAMPLING,
  SPLIT_SUBJECTS,
  subject_split,
)


@dataclass(frozen=True)
class Cfg:
  input_dir: str = "datasets/csv/lafan"
  output_dir: str = "logs/reproduction/conditional_shoulder80_prior_v1/data"
  baseline_dir: str = "logs/reproduction/conditional_contact_v2/data"
  window_size: int = 10
  stride: int = 5
  fps: int = 50
  seed: int = 42
  early_min_delta: float = 0.01
  entry_min_yaw: float = 0.08
  entry_min_delta: float = 0.02
  hold_min_last: float = 0.12
  hold_min_window: float = 0.10
  hold_max_delta: float = 0.01
  max_yaw: float = 0.35
  max_shoulder_speed: float = 0.20
  max_pelvis_yaw_change: float = 0.05
  context_seconds: float = 1.0
  context_max_displacement: float = 0.10
  context_max_path: float = 0.15
  early_link_seconds: float = 1.0
  stand_max_yaw_delta: float = 0.005


def now():
  return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  os.replace(temporary, path)


def context_motion(positions, starts, cfg: Cfg):
  """51 endpoint positions span exactly one second at 50 Hz.

  The half-frame window center is rounded down. Context is not truncated at
  source boundaries. Invalid contexts get finite placeholder metrics and fail.
  """
  span = int(round(cfg.context_seconds * cfg.fps))
  centers = starts + (cfg.window_size - 1) // 2
  left, right = centers - span // 2, centers - span // 2 + span
  available = (left >= 0) & (right < len(positions))
  a, b = left.clip(0, len(positions) - 1), right.clip(0, len(positions) - 1)
  path_prefix = np.r_[
    0.0, np.cumsum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
  ]
  displacement = np.linalg.norm(positions[b, :2] - positions[a, :2], axis=1)
  path = path_prefix[b] - path_prefix[a]
  keep = (
    available
    & (displacement <= cfg.context_max_displacement)
    & (path <= cfg.context_max_path)
  )
  return keep, available, displacement, path


def phase_counts(labels, phases, keep=None):
  if keep is None:
    keep = labels >= 0
  return [
    [int(np.sum(keep & (labels == c) & (phases == p))) for p in range(4)]
    for c in range(3)
  ]


def classify_windows(motion, cfg: Cfg):
  arrays = motion_arrays(motion, cfg.fps)
  pos, quat, vel, ang, joints = arrays
  starts = np.arange(0, len(motion) - cfg.window_size + 1, cfg.stride)
  indices = starts[:, None] + np.arange(cfg.window_size)
  w, x, y, z = quat.T
  yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
  speed = np.linalg.norm(vel[indices, :2], axis=-1).max(1)
  pelvis_delta = yaw[indices[:, -1]] - yaw[indices[:, 0]]
  waist = joints[indices, 12]
  waist_delta = waist[:, -1] - waist[:, 0]
  joint_velocity = np.gradient(joints, 1 / cfg.fps, axis=0)
  joint_rms = np.sqrt(np.mean(joint_velocity[indices] ** 2, axis=(1, 2)))
  stable = (
    (pos[indices, 2].min(1) >= 0.60)
    & (np.abs(roll[indices]).max(1) <= np.deg2rad(20))
    & (np.abs(pitch[indices]).max(1) <= np.deg2rad(20))
    & (np.abs(pelvis_delta) <= cfg.max_pelvis_yaw_change)
  )
  masks = [
    stable
    & (speed <= 0.08)
    & (np.abs(waist).max(1) <= 0.06)
    & (np.abs(waist_delta) <= cfg.stand_max_yaw_delta)
    & (joint_rms <= 0.25)
    & (np.abs(ang[indices, 2]).max(1) <= 0.15)
  ]
  condition_phases = [np.zeros(len(starts), dtype=np.int64)]
  for sign in (1, -1):
    q, delta = sign * waist, sign * waist_delta
    early = (
      (q[:, 0] >= 0) & (q[:, -1] < cfg.entry_min_yaw) & (delta >= cfg.early_min_delta)
    )
    entry = (
      (q[:, 0] >= 0) & (q[:, -1] >= cfg.entry_min_yaw) & (delta >= cfg.entry_min_delta)
    )
    hold = (
      (q[:, -1] >= cfg.hold_min_last)
      & (q.min(1) >= cfg.hold_min_window)
      & (np.abs(delta) <= cfg.hold_max_delta)
    )
    if np.any((early.astype(int) + entry + hold) > 1):
      raise ValueError("Overlapping shoulder phase definitions")
    masks.append(
      stable
      & (speed <= cfg.max_shoulder_speed)
      & (q.max(1) <= cfg.max_yaw)
      & (early | entry | hold)
    )
    condition_phases.append(
      np.where(early, 1, np.where(entry, 2, np.where(hold, 3, 0)))
    )
  masks = np.stack(masks, axis=1)
  unique = masks.sum(1) == 1
  labels = np.where(unique, masks.argmax(1), -1).astype(np.int64)
  phases = np.stack(condition_phases, axis=1)[np.arange(len(starts)), labels.clip(0)]
  phases[~unique] = -1
  context_ok, available, displacement, path = context_motion(pos, starts, cfg)
  pre_context = phase_counts(labels, phases)
  labels[~context_ok], phases[~context_ok] = -1, -1
  metrics = np.stack(
    (speed, waist_delta, waist[:, -1], pelvis_delta, joint_rms, displacement, path),
    axis=1,
  )
  return {
    "starts": starts,
    "indices": indices,
    "labels": labels,
    "phases": phases,
    "metrics": metrics,
    "arrays": arrays,
    "audit": {
      "all_windows": len(starts),
      "pre_context": pre_context,
      "after_context": phase_counts(labels, phases),
      "ambiguous_windows": int((masks.sum(1) > 1).sum()),
      "context_unavailable": int((~available).sum()),
    },
  }


def link_early_windows(starts, labels, phases, keep, waist, cfg: Cfg):
  """Link to a later, physically accepted mature window of the same direction."""
  keep = keep.copy()
  links = np.full(len(starts), -1, dtype=np.int64)
  for condition, sign in ((1, 1), (2, -1)):
    mature = np.flatnonzero(keep & (labels == condition) & np.isin(phases, (2, 3)))
    for index in np.flatnonzero(keep & (labels == condition) & (phases == 1)):
      future = mature[
        (starts[mature] > starts[index])
        & (starts[mature] <= starts[index] + round(cfg.early_link_seconds * cfg.fps))
      ]
      for candidate in future:
        connection = sign * waist[starts[index] : starts[candidate] + cfg.window_size]
        if connection.min() >= 0 and connection.max() <= cfg.max_yaw:
          links[index] = starts[candidate]
          break
      if links[index] < 0:
        keep[index] = False
  return keep, links


def screen_frames(result, screening):
  model, data, joint_ids, body_ids, foot_ids = screening
  pos, quat, _, _, joints = result["arrays"]
  free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
  free_adr, joint_adr = int(model.jnt_qposadr[free]), model.jnt_qposadr[joint_ids]
  limits, floor_id = model.jnt_range[joint_ids], model.geom("conditional_floor").id
  valid = np.zeros(len(pos), dtype=bool)
  ee = np.zeros((len(pos), 5, 3), dtype=np.float32)
  rejected = {
    "joint_limit": 0,
    "support": 0,
    "self_penetration": 0,
    "ground_penetration": 0,
  }
  selected = np.unique(result["indices"][result["labels"] >= 0])
  for frame in selected:
    if not (
      (joints[frame] >= limits[:, 0] + 1e-4) & (joints[frame] <= limits[:, 1] - 1e-4)
    ).all():
      rejected["joint_limit"] += 1
      continue
    data.qpos[:] = model.qpos0
    data.qpos[free_adr : free_adr + 3] = pos[frame]
    data.qpos[free_adr + 3 : free_adr + 7] = quat[frame]
    data.qpos[joint_adr] = joints[frame]
    mujoco.mj_forward(model, data)
    bottoms = data.geom_xpos[foot_ids, 2] - model.geom_size[foot_ids, 0]
    if bottoms.min() < -0.02 or bottoms.min() > 0.03:
      rejected["support"] += 1
      continue
    reason = None
    for contact in data.contact:
      a, b = map(int, contact.geom)
      if floor_id not in (a, b) and contact.dist < -1e-4:
        reason = "self_penetration"
        break
      if (
        floor_id in (a, b)
        and (b if a == floor_id else a) not in foot_ids
        and contact.dist < -0.002
      ):
        reason = "ground_penetration"
        break
    if reason:
      rejected[reason] += 1
      continue
    valid[frame], ee[frame] = True, data.xpos[body_ids]
  return (
    valid,
    ee,
    {
      "selected_frames": len(selected),
      "accepted_frames": int(valid.sum()),
      "rejected": rejected,
    },
  )


def nonoverlap_count(intervals):
  """Count disjoint 10-frame intervals; mirrors and repeated rows count once."""
  by_source = defaultdict(set)
  for source, start, end in intervals:
    by_source[source].add((start, end))
  count = 0
  for spans in by_source.values():
    previous_end = -1
    for start, end in sorted(spans, key=lambda span: (span[1], span[0])):
      if start >= previous_end:
        count += 1
        previous_end = end
  return count


def baseline_intervals(root: Path):
  manifest = json.loads((root / "manifest.json").read_text())
  if sha256(root / "windows.csv") != manifest["windows_manifest_sha256"]:
    raise ValueError("Baseline lineage checksum mismatch")
  with (root / "windows.csv").open(newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if int(row["condition_id"]) in (1, 2)]
  intervals = {name: [] for name in SPLIT_SUBJECTS}
  for row in rows:
    intervals[subject_split(row["source"])].append(
      (row["source"], int(row["frame50_start"]), int(row["frame50_end_exclusive"]))
    )
  return intervals, sha256(root / "manifest.json")


def data_gate(records, baseline):
  for record in records:
    if subject_split(record["source"]) != record["split"]:
      raise ValueError("Subject or mirror leakage in dataset lineage")
    condition, phase = record["condition_id"], record["phase_id"]
    if condition not in (0, 1, 2) or phase not in (
      (0,) if condition == 0 else (1, 2, 3)
    ):
      raise ValueError("Invalid condition/phase in dataset lineage")
  checks, coverage, comparisons = {}, {}, {}
  for split in SPLIT_SUBJECTS:
    rows = [r for r in records if r["split"] == split]
    coverage[split] = {}
    for condition in range(3):
      for phase in (0,) if condition == 0 else (1, 2, 3):
        selected = [
          r for r in rows if r["condition_id"] == condition and r["phase_id"] == phase
        ]
        sources = sorted({r["source"] for r in selected})
        key = f"{CONDITION_NAMES[condition]}/{PHASE_NAMES[phase]}"
        coverage[split][key] = {
          "windows": len(selected),
          "sources": sources,
          "nonoverlap_windows_without_mirror_duplication": nonoverlap_count(
            [
              (r["source"], r["frame50_start"], r["frame50_end_exclusive"])
              for r in selected
            ]
          ),
        }
        checks[f"{split}/{key}/nonempty"] = bool(selected)
        if split == "train" and condition:
          checks[f"{split}/{key}/at_least_two_original_files"] = len(sources) >= 2
    intervals = {
      (r["source"], r["frame50_start"], r["frame50_end_exclusive"])
      for r in rows
      if r["condition_id"]
    }
    previous = defaultdict(list)
    for source, start, end in baseline[split]:
      previous[source].append((start, end))
    novel = [
      item
      for item in intervals
      if not any(item[1] < end and start < item[2] for start, end in previous[item[0]])
    ]
    comparisons[split] = {
      "baseline_nonoverlap_windows": nonoverlap_count(baseline[split]),
      "new_nonoverlap_windows": nonoverlap_count(intervals),
      "novel_nonoverlap_windows_outside_baseline_intervals": nonoverlap_count(novel),
      "mirrors_count_as_independent": False,
    }
  comp = comparisons["train"]
  checks["train/shoulder_nonoverlap_count_increased"] = (
    comp["new_nonoverlap_windows"] > comp["baseline_nonoverlap_windows"]
  )
  checks["train/novel_shoulder_intervals_present"] = (
    comp["novel_nonoverlap_windows_outside_baseline_intervals"] > 0
  )
  return {
    "passed": all(checks.values()),
    "checks": checks,
    "failed_checks": [name for name, passed in checks.items() if not passed],
    "coverage": coverage,
    "baseline_comparison": comparisons,
  }


def curate(cfg: Cfg):
  if (cfg.window_size, cfg.stride, cfg.fps, cfg.seed) != (10, 5, 50, 42):
    raise ValueError("This experiment freezes W=10, stride=5, fps=50 and seed=42")
  torch.set_num_threads(2)
  output, baseline = Path(cfg.output_dir), Path(cfg.baseline_dir)
  if not output.resolve().is_relative_to(
    Path("/personal_fast_storage/smp_experiments")
  ):
    raise RuntimeError("Experiment data must be on the dedicated persistent storage")
  sources = sorted(Path(cfg.input_dir).glob("*.csv"))
  if not sources:
    raise FileNotFoundError(cfg.input_dir)
  source_hashes = {path.name: sha256(path) for path in sources}
  old_intervals, baseline_sha = baseline_intervals(baseline)
  script_sha = sha256(__file__)
  manifest_path = output / "manifest.json"
  if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    if (
      existing["config"] != asdict(cfg)
      or existing["sources"] != source_hashes
      or existing["baseline_manifest_sha256"] != baseline_sha
      or existing["curator_sha256"] != script_sha
    ):
      raise RuntimeError(
        "Immutable dataset identity changed; do not overwrite this experiment"
      )
    for item in existing["files"]:
      if sha256(output / item["path"]) != item["sha256"]:
        raise RuntimeError("Dataset shard checksum mismatch")
    for key, name in (
      ("windows_manifest_sha256", "windows.csv"),
      ("norm_stats_sha256", "norm_stats.npz"),
      ("audit_sha256", "audit.json"),
    ):
      if existing.get(key) and sha256(output / name) != existing[key]:
        raise RuntimeError(f"Dataset checksum mismatch: {name}")
    print(
      json.dumps({"status": "verified_existing", "gate": existing["gate"]}), flush=True
    )
    return existing
  if output.exists() and any(output.iterdir()):
    raise RuntimeError(
      "Partial curation exists; inspect it before creating a separate attempt directory"
    )
  output.mkdir(parents=True, exist_ok=True)
  manifest = {
    "schema_version": 1,
    "created_at_utc": now(),
    "experiment": "EXP-027",
    "status": "curating",
    "config": asdict(cfg),
    "curator_sha256": script_sha,
    "condition_schema": CONDITION_SCHEMA,
    "sampling": SAMPLING,
    "split_subjects": SPLIT_SUBJECTS,
    "sources": source_hashes,
    "baseline_manifest_sha256": baseline_sha,
    "files": [],
    "counts": {s: [0] * 3 for s in SPLIT_SUBJECTS},
    "phase_counts": {s: [[0] * 4 for _ in range(3)] for s in SPLIT_SUBJECTS},
    "metric_columns": [
      "max_speed",
      "waist_delta",
      "waist_last",
      "pelvis_yaw_delta",
      "joint_rms_velocity",
      "context_displacement",
      "context_path",
    ],
    "safety": {
      "joint_margin": 0.0001,
      "self_penetration_m": 0.0001,
      "support_gap_m": 0.03,
      "foot_penetration_m": 0.02,
      "nonfoot_ground_penetration_m": 0.002,
      "root_height_min_m": 0.60,
      "max_abs_roll_pitch_degrees": 20,
      "method": "MuJoCo FK and collision screening; not a dynamic stability guarantee",
    },
    "early_link": "same-direction accepted mature window starts within 1 second; intervening signed yaw remains in [0,0.35]",
    "context": "51 position endpoints, 1.00 second, center rounded down; applied to all conditions",
    "normalization": "training subjects 1-3 only, percentiles 1/99 over curated windows",
    "training_authorized_by_this_script": False,
  }
  screening, records, audits, training = screening_model(), [], [], []
  for source in sources:
    raw = np.loadtxt(source, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != 36 or not np.isfinite(raw).all():
      raise ValueError(f"Invalid motion: {source}")
    interpolated, split = interpolate(raw, cfg.fps), subject_split(source.name)
    for mirrored in (False, True):
      motion = mirror_g1_motion(interpolated) if mirrored else interpolated
      result = classify_windows(motion, cfg)
      valid, ee, frame_audit = screen_frames(result, screening)
      starts, labels, phases = (result[k] for k in ("starts", "labels", "phases"))
      keep = (labels >= 0) & valid[result["indices"]].all(1)
      after_fk = phase_counts(labels, phases, keep)
      keep, links = link_early_windows(starts, labels, phases, keep, motion[:, 19], cfg)
      audits.append(
        {
          "source": source.name,
          "split": split,
          "mirrored": mirrored,
          **result["audit"],
          "frames": frame_audit,
          "after_fk": after_fk,
          "after_early_link": phase_counts(labels, phases, keep),
        }
      )
      if not keep.any():
        print(f"{source.name} mirror={mirrored}: no accepted windows", flush=True)
        continue
      pos, quat, vel, ang, joints = result["arrays"]
      tensors = [
        torch.tensor(a, dtype=torch.float32) for a in (pos, quat, vel, ang, ee, joints)
      ]
      windows = _compute_windows(*tensors, cfg.window_size, cfg.stride).numpy()[keep]
      selected_labels, selected_phases = labels[keep], phases[keep]
      if windows.shape[1:] != (10, 59) or not np.isfinite(windows).all():
        raise ValueError("Nonfinite or malformed curated features")
      relative = f"{split}/{source.stem}{'_mirror' if mirrored else ''}.npz"
      destination = output / relative
      destination.parent.mkdir(exist_ok=True)
      np.savez_compressed(
        destination,
        windows=windows,
        condition_id=selected_labels,
        phase_id=selected_phases,
        frame50_start=starts[keep],
        metrics=result["metrics"][keep],
        early_link_frame50=links[keep],
      )
      counts = np.bincount(selected_labels, minlength=3)
      phase_total = np.asarray(phase_counts(labels, phases, keep))
      manifest["counts"][split] = (
        np.array(manifest["counts"][split]) + counts
      ).tolist()
      manifest["phase_counts"][split] = (
        np.array(manifest["phase_counts"][split]) + phase_total
      ).tolist()
      manifest["files"].append(
        {
          "path": relative,
          "source": source.name,
          "source_sha256": source_hashes[source.name],
          "split": split,
          "mirrored": mirrored,
          "sha256": sha256(destination),
          "counts": counts.tolist(),
          "phase_counts": phase_total.tolist(),
          "accepted_windows": len(windows),
        }
      )
      for index, old_index in enumerate(np.flatnonzero(keep)):
        condition, phase, start = (
          int(labels[old_index]),
          int(phases[old_index]),
          int(starts[old_index]),
        )
        records.append(
          {
            "source": source.name,
            "source_sha256": source_hashes[source.name],
            "split": split,
            "mirrored": mirrored,
            "frame50_start": start,
            "frame50_end_exclusive": start + 10,
            "condition_id": condition,
            "condition": CONDITION_NAMES[condition],
            "phase_id": phase,
            "phase": PHASE_NAMES[phase],
            "early_link_frame50": int(links[old_index]),
            "source_group": source.name,
            "shard": relative,
            "index": index,
          }
        )
      if split == "train":
        training.append(windows)
      print(
        f"{relative}: classes={counts.tolist()}, phases={phase_total.tolist()}",
        flush=True,
      )
  fields = [
    "source",
    "source_sha256",
    "split",
    "mirrored",
    "frame50_start",
    "frame50_end_exclusive",
    "condition_id",
    "condition",
    "phase_id",
    "phase",
    "early_link_frame50",
    "source_group",
    "shard",
    "index",
  ]
  with (output / "windows.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
  if training:
    flat = np.concatenate(training).reshape(-1, 59)
    low, high = np.percentile(flat, [1, 99], axis=0).astype(np.float32)
    high = np.where(high - low < 1e-6, low + 1.0, high)
    np.savez(output / "norm_stats.npz", q_low=low, q_high=high)
    manifest["norm_stats_sha256"] = sha256(output / "norm_stats.npz")
  gate = data_gate(records, old_intervals)
  atomic_json(
    output / "audit.json",
    {
      "created_at_utc": now(),
      "sources": audits,
      "gate": gate,
      "counts": manifest["counts"],
      "phase_counts": manifest["phase_counts"],
      "sampling": SAMPLING,
      "no_independent_sample_claim_for_mirrors_or_overlap": True,
    },
  )
  manifest.update(
    status="ready" if gate["passed"] else "blocked_data_gate",
    completed_at_utc=now(),
    gate=gate,
    audit_sha256=sha256(output / "audit.json"),
    windows_manifest_sha256=sha256(output / "windows.csv"),
  )
  atomic_json(manifest_path, manifest)
  print(
    json.dumps(
      {
        "status": manifest["status"],
        "counts": manifest["counts"],
        "phase_counts": manifest["phase_counts"],
        "failed_checks": gate["failed_checks"],
        "baseline_comparison": gate["baseline_comparison"],
      }
    ),
    flush=True,
  )
  return manifest


if __name__ == "__main__":
  result = curate(tyro.cli(Cfg))
  raise SystemExit(0 if result["gate"]["passed"] else 2)
