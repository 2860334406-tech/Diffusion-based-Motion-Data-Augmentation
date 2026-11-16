"""Strengthen shoulders by selecting immutable, FK-screened v1 windows.

No joint-angle amplification or synthetic labels; the v1 evaluation gates stay fixed.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from smp.pretrain.conditional_dataset import ConditionalMotionDataset
from smp.pretrain.conditioning import CONDITION_NAMES, CONDITION_SCHEMA, sha256


@dataclass(frozen=True)
class Cfg:
  parent_dir: str = "logs/reproduction/conditional_contact_v1/data"
  output_dir: str = "logs/reproduction/conditional_contact_v2/data"
  entry_min_yaw: float = .08
  entry_min_delta: float = .02
  entry_min_start_yaw: float = 0.
  hold_min_yaw: float = .12
  hold_min_window_yaw: float = .10
  hold_max_abs_delta: float = .01
  max_yaw: float = .35
  seed: int = 42


def shoulder_phases(windows, labels, cfg: Cfg):
  """Keep mask and disjoint phases: 0 other, 1 entering, 2 holding."""
  shoulder = np.isin(labels, (1, 2))
  sign = np.where(labels == 2, -1., 1.)
  waist = sign[:, None] * windows[..., 21]
  delta = waist[:, -1] - waist[:, 0]
  entering = (
    shoulder & (waist[:, -1] >= cfg.entry_min_yaw)
    & (waist[:, 0] >= cfg.entry_min_start_yaw)
    & (delta >= cfg.entry_min_delta)
  )
  holding = (
    shoulder & (waist[:, -1] >= cfg.hold_min_yaw)
    & (waist.min(1) >= cfg.hold_min_window_yaw)
    & (np.abs(delta) <= cfg.hold_max_abs_delta)
  )
  if np.any(entering & holding):
    raise ValueError("Shoulder entry/hold selection must be disjoint")
  keep = ~shoulder | ((entering | holding) & (waist.max(1) <= cfg.max_yaw))
  phase = np.where(entering, 1, np.where(holding, 2, 0)).astype(np.int64)
  return keep, phase


def curate(cfg: Cfg):
  if cfg.entry_min_delta <= cfg.hold_max_abs_delta:
    raise ValueError("Entry delta must exceed the holding band")
  torch.set_num_threads(2)
  parent, output = Path(cfg.parent_dir), Path(cfg.output_dir)
  parent_manifest = json.loads((parent / "manifest.json").read_text())
  parent_sha = sha256(parent / "manifest.json")
  for split in ("train", "validation"):
    ConditionalMotionDataset(parent, split)
  if sha256(parent / "windows.csv") != parent_manifest["windows_manifest_sha256"]:
    raise ValueError("Parent window lineage checksum mismatch")
  manifest_path = output / "manifest.json"
  if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    if existing["config"] != asdict(cfg) or existing["parent_sha256"] != parent_sha:
      raise RuntimeError("Existing v2 dataset differs; use a new version directory")
    for split in ("train", "validation"):
      ConditionalMotionDataset(output, split)
    print(json.dumps({"status": "verified_existing", "counts": existing["counts"]}), flush=True)
    return existing
  output.mkdir(parents=True, exist_ok=True)
  manifest = {
    "schema_version": 2, "condition_schema": CONDITION_SCHEMA,
    "config": asdict(cfg), "parent_sha256": parent_sha,
    "parent_manifest": str((parent / "manifest.json").resolve()),
    "sources": parent_manifest["sources"], "files": [],
    "safety": {**parent_manifest["safety"], "method": "subset of verified FK-screened v1 windows"},
    "metric_columns": parent_manifest["metric_columns"],
    "counts": {split: [0]*5 for split in ("train", "validation")},
    "phase_counts": {split: [[0]*3 for _ in range(5)] for split in ("train", "validation")},
    "sampling": {
      "algorithm": "shoulder_phase_balanced_v1", "samples_per_epoch": 3360,
      "per_condition": [672]*5, "shoulder_entering_per_epoch": 336,
      "shoulder_holding_per_epoch": 336, "phase_names": ["not_shoulder", "entering", "holding"],
      "within_phase": "uniform_with_replacement", "seed": cfg.seed,
    },
    "acceptance": "unchanged v1 thresholds; stronger curation is not a relaxed gate",
  }
  training = []
  with (output / "windows.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["source", "source_sha256", "split", "mirrored", "frame50_start",
                     "frame50_end_exclusive", "condition_id", "condition", "phase_id",
                     "shard", "index", "parent_shard", "parent_index"])
    for item in parent_manifest["files"]:
      with np.load(parent / item["path"], allow_pickle=False) as old:
        windows, labels = old["windows"], old["condition_id"]
        starts, metrics = old["frame50_start"], old["metrics"]
      keep, phases = shoulder_phases(windows, labels, cfg)
      if not keep.any():
        continue
      selected = np.flatnonzero(keep)
      windows, labels, phases = windows[keep], labels[keep], phases[keep]
      destination = output / item["path"]
      destination.parent.mkdir(parents=True, exist_ok=True)
      np.savez_compressed(destination, windows=windows, condition_id=labels, phase_id=phases,
                          frame50_start=starts[keep], metrics=metrics[keep], parent_index=selected)
      split = item["split"]
      counts = np.bincount(labels, minlength=5)
      manifest["counts"][split] = (np.array(manifest["counts"][split])+counts).tolist()
      for condition in range(5):
        phase_counts = np.bincount(phases[labels == condition], minlength=3)
        manifest["phase_counts"][split][condition] = (
          np.array(manifest["phase_counts"][split][condition])+phase_counts
        ).tolist()
      manifest["files"].append({
        "path": item["path"], "source": item["source"], "source_sha256": item["source_sha256"],
        "split": split, "mirrored": item["mirrored"], "sha256": sha256(destination),
        "counts": counts.tolist(), "accepted_windows": len(windows),
        "parent_shard_sha256": item["sha256"], "parent_windows": item["accepted_windows"],
      })
      for index, old_index in enumerate(selected):
        start, label = int(starts[old_index]), int(labels[index])
        writer.writerow([item["source"], item["source_sha256"], split, item["mirrored"],
                         start, start+10, label, CONDITION_NAMES[label], int(phases[index]),
                         item["path"], index, item["path"], int(old_index)])
      if split == "train":
        training.append(windows)
  for split, counts in manifest["counts"].items():
    if min(counts) == 0 or any(min(manifest["phase_counts"][split][c][1:]) == 0 for c in (1, 2)):
      raise ValueError(f"Missing condition or shoulder phase in {split}")
  flat = np.concatenate(training).reshape(-1, 59)
  low, high = np.percentile(flat, [1, 99], axis=0).astype(np.float32)
  high = np.where(high-low < 1e-6, low+1., high)
  np.savez(output / "norm_stats.npz", q_low=low, q_high=high)
  manifest["norm_stats_sha256"] = sha256(output / "norm_stats.npz")
  manifest["windows_manifest_sha256"] = sha256(output / "windows.csv")
  manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
  print(json.dumps({"counts": manifest["counts"], "phase_counts": manifest["phase_counts"]}), flush=True)
  return manifest


if __name__ == "__main__":
  curate(tyro.cli(Cfg))
