"""Run the gated G1 shoulder-yield IK data-production pipeline.

This driver produces data only.  It never launches Prior or PPO training.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tyro

from smp.data.shoulder_ik import (
  MEAN_YAW_SPEED_MAX_DEG_S,
  MEAN_YAW_SPEED_MIN_DEG_S,
  SAMPLING_SCHEMA,
  WAIST_YAW_TRACKING_KP,
  YAW_MAX_DEG,
  YAW_MIN_DEG,
  Gates,
  ShoulderIkGenerator,
  atomic_json,
  make_specs,
  retry_spec,
  sha256,
  split_counts,
  write_result,
)
from smp.data.shoulder_ik_dataset import finalize_dataset


@dataclass(frozen=True)
class Cfg:
  stage: str = "pilot"
  output_dir: str = "logs/reproduction/g1_shoulder_ik_v2/pilot"
  seed: int = 42
  workers: int = 8
  max_clips: int = 0
  pilot_manifest: str = "logs/reproduction/g1_shoulder_ik_v2/pilot/manifest.json"
  require_visual_review: bool = True
  yaw_min_deg: float = YAW_MIN_DEG
  yaw_max_deg: float = YAW_MAX_DEG
  mean_yaw_speed_min_deg_s: float = MEAN_YAW_SPEED_MIN_DEG_S
  mean_yaw_speed_max_deg_s: float = MEAN_YAW_SPEED_MAX_DEG_S


_GENERATOR: ShoulderIkGenerator | None = None


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def generator() -> ShoulderIkGenerator:
  global _GENERATOR
  if _GENERATOR is None:
    _GENERATOR = ShoulderIkGenerator()
  return _GENERATOR


def run_slot(spec, output_dir: str):
  output = Path(output_dir)
  attempts = []
  for candidate in (spec, retry_spec(spec)):
    try:
      passed, motion, execution, metrics = generator().produce(candidate)
      record = write_result(
        output,
        candidate,
        passed,
        motion,
        execution,
        metrics,
        generator().sim_model,
      )
      attempts.append(record)
      if passed:
        return {
          "slot": f"{spec.family_id}:{spec.condition_id}",
          "status": "accepted",
          "accepted": record,
          "attempts": attempts,
        }
    except Exception as error:
      attempts.append(
        {
          "clip_id": candidate.clip_id,
          "attempt": candidate.attempt,
          "passed": False,
          "exception": f"{type(error).__name__}: {error}",
        }
      )
  return {
    "slot": f"{spec.family_id}:{spec.condition_id}",
    "status": "failed_after_two_attempts",
    "attempts": attempts,
  }


def source_identity() -> dict[str, str]:
  paths = [
    Path("src/smp/data/shoulder_ik.py"),
    Path("src/smp/data/shoulder_ik_dataset.py"),
    Path("scripts/run_g1_shoulder_ik_data.py"),
    Path("docs/G1_SHOULDER_IK_DATA_PRODUCTION.md"),
    Path("pyproject.toml"),
    Path("uv.lock"),
  ]
  return {str(path): sha256(path) for path in paths}


def sampling_identity(cfg: Cfg) -> dict:
  return {
    "schema": SAMPLING_SCHEMA,
    "yaw_amplitude_deg": [cfg.yaw_min_deg, cfg.yaw_max_deg],
    "target_mean_yaw_speed_deg_s": [
      cfg.mean_yaw_speed_min_deg_s,
      cfg.mean_yaw_speed_max_deg_s,
    ],
    "entry_duration": "ceil(amplitude_deg / speed_deg_s * 50) / 50",
    "amplitude_strategy": "randomized_latin_hypercube_per_stage_split_posture",
    "speed_strategy": "independent_randomized_latin_hypercube",
  }


def specs_sha256(specs) -> str:
  payload = json.dumps(
    [asdict(spec) for spec in specs],
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode()
  return hashlib.sha256(payload).hexdigest()


def manifest_identity(cfg: Cfg, specs) -> dict:
  return {
    "schema_version": 2,
    "experiment": "EXP-032",
    "stage": cfg.stage,
    "seed": cfg.seed,
    "sampling": sampling_identity(cfg),
    "specs_sha256": specs_sha256(specs),
    "gates": asdict(Gates()),
    "controller": {"waist_yaw_tracking_kp": WAIST_YAW_TRACKING_KP},
    "requested_counts": split_counts(specs),
    "run_limit": cfg.max_clips,
    "max_attempts_per_slot": 2,
    "source_identity": source_identity(),
    "dependencies": {
      "python": platform.python_version(),
      "numpy": version("numpy"),
      "mink": version("mink"),
      "mujoco": version("mujoco"),
      "mjlab": version("mjlab"),
    },
  }


def amplitude_distribution(records: list[dict], cfg: Cfg) -> dict:
  edges = np.arange(cfg.yaw_min_deg, cfg.yaw_max_deg + 5.0, 5.0)
  edges[-1] = cfg.yaw_max_deg
  required_splits = (
    ("pilot",)
    if cfg.stage == "pilot"
    else (
      "train",
      "validation",
      "test",
    )
  )
  groups = {}
  all_passed = True
  for split in required_splits:
    for condition, side in ((1, "left_shoulder"), (2, "right_shoulder")):
      selected = [
        record["accepted"]
        for record in records
        if record
        and record.get("status") == "accepted"
        and record["accepted"]["split"] == split
        and record["accepted"]["condition_id"] == condition
      ]
      actual = np.asarray(
        [item["actual_waist_amplitude_deg"] for item in selected], dtype=np.float64
      )
      target = np.asarray(
        [item["target_yaw_amplitude_deg"] for item in selected], dtype=np.float64
      )
      speed = np.asarray(
        [item["target_mean_yaw_speed_deg_s"] for item in selected],
        dtype=np.float64,
      )
      if len(actual):
        counts, _ = np.histogram(actual, bins=edges)
        finite = bool(
          np.isfinite(actual).all()
          and np.isfinite(target).all()
          and np.isfinite(speed).all()
        )
        passed = bool(
          finite
          and (actual >= cfg.yaw_min_deg).all()
          and (actual <= cfg.yaw_max_deg).all()
          and (counts > 0).all()
          and actual.min() <= cfg.yaw_min_deg + 1.0
          and actual.max() >= cfg.yaw_max_deg - 1.0
        )
        summary = {
          "count": int(len(actual)),
          "target_deg": {
            "min": float(target.min()),
            "p50": float(np.quantile(target, 0.5)),
            "max": float(target.max()),
          },
          "actual_deg": {
            "min": float(actual.min()),
            "p05": float(np.quantile(actual, 0.05)),
            "p50": float(np.quantile(actual, 0.5)),
            "p95": float(np.quantile(actual, 0.95)),
            "max": float(actual.max()),
          },
          "speed_deg_s": {
            "min": float(speed.min()),
            "p50": float(np.quantile(speed, 0.5)),
            "max": float(speed.max()),
          },
          "angle_speed_correlation": (
            float(np.corrcoef(target, speed)[0, 1]) if len(target) > 1 else 0.0
          ),
          "bin_edges_deg": edges.tolist(),
          "bin_counts": counts.tolist(),
          "passed": passed,
        }
      else:
        summary = {
          "count": 0,
          "bin_edges_deg": edges.tolist(),
          "bin_counts": [0] * (len(edges) - 1),
          "passed": False,
        }
        passed = False
      groups[f"{split}/{side}"] = summary
      all_passed &= passed
  return {
    "schema_version": 1,
    "five_degree_bins": True,
    "endpoint_requirements_deg": {
      "minimum_at_most": cfg.yaw_min_deg + 1.0,
      "maximum_at_least": cfg.yaw_max_deg - 1.0,
    },
    "groups": groups,
    "passed": bool(all_passed),
  }


def check_formal_parent(cfg: Cfg):
  path = Path(cfg.pilot_manifest)
  if not path.is_file():
    raise FileNotFoundError(f"Formal generation requires pilot manifest: {path}")
  pilot = json.loads(path.read_text())
  if pilot.get("status") != "complete" or not pilot.get("gate_passed"):
    raise RuntimeError("Pilot manifest has not passed all automated gates")
  if pilot.get("identity", {}).get("experiment") != "EXP-032":
    raise RuntimeError("Formal generation requires the independent EXP-032 pilot")
  if pilot["identity"].get("sampling") != sampling_identity(cfg):
    raise RuntimeError("Pilot and formal sampling identities do not match")
  if not pilot.get("amplitude_distribution", {}).get("passed"):
    raise RuntimeError("Pilot amplitude-distribution gate did not pass")
  if cfg.require_visual_review:
    review_path = path.parent / "visual_review.json"
    if not review_path.is_file():
      raise FileNotFoundError(
        f"Formal generation requires visual review: {review_path}"
      )
    review = json.loads(review_path.read_text())
    if review.get("passed") is not True or review.get("reviewed_clips") != 60:
      raise RuntimeError("Pilot visual review is absent or did not pass 60 clips")


def main(cfg: Cfg):
  if sys.version_info < (3, 11):
    raise RuntimeError("G1 shoulder IK data generation requires Python >= 3.11")
  if cfg.stage not in ("pilot", "formal"):
    raise ValueError("stage must be pilot or formal")
  if cfg.workers < 1:
    raise ValueError("workers must be positive")
  if cfg.stage == "formal":
    check_formal_parent(cfg)
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  os.environ.setdefault("MKL_NUM_THREADS", "1")

  output = Path(cfg.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  probe = output / ".write_probe"
  probe.write_bytes(b"g1-shoulder-ik\n")
  with probe.open("rb") as stream:
    os.fsync(stream.fileno())
  probe.unlink()

  all_specs = make_specs(
    cfg.stage,
    cfg.seed,
    yaw_min_deg=cfg.yaw_min_deg,
    yaw_max_deg=cfg.yaw_max_deg,
    mean_yaw_speed_min_deg_s=cfg.mean_yaw_speed_min_deg_s,
    mean_yaw_speed_max_deg_s=cfg.mean_yaw_speed_max_deg_s,
  )
  specs = all_specs
  if cfg.max_clips:
    specs = specs[: cfg.max_clips]
  identity = manifest_identity(cfg, all_specs)
  manifest_path = output / "manifest.json"
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != identity:
      raise RuntimeError("Existing output identity mismatch; choose a new directory")
  else:
    manifest = {
      "identity": identity,
      "status": "running",
      "gate_passed": False,
      "created_at_utc": now(),
      "updated_at_utc": now(),
      "output_dir": str(output.resolve()),
      "slots": {},
    }
    atomic_json(manifest_path, manifest)

  pending = [
    spec
    for spec in specs
    if f"{spec.family_id}:{spec.condition_id}" not in manifest["slots"]
  ]
  print(
    json.dumps(
      {
        "stage": cfg.stage,
        "requested": len(specs),
        "completed": len(specs) - len(pending),
        "pending": len(pending),
        "workers": cfg.workers,
        "output": str(output),
      }
    ),
    flush=True,
  )
  if pending:
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
      futures = {pool.submit(run_slot, spec, str(output)): spec for spec in pending}
      for index, future in enumerate(as_completed(futures), 1):
        spec = futures[future]
        try:
          record = future.result()
        except BaseException as error:
          record = {
            "slot": f"{spec.family_id}:{spec.condition_id}",
            "status": "worker_failed",
            "attempts": [],
            "exception": f"{type(error).__name__}: {error}",
          }
        manifest["slots"][record["slot"]] = record
        manifest["updated_at_utc"] = now()
        atomic_json(manifest_path, manifest)
        print(
          f"[{index}/{len(pending)}] {record['slot']} {record['status']}",
          flush=True,
        )

  selected_slots = {f"{spec.family_id}:{spec.condition_id}" for spec in specs}
  records = [manifest["slots"].get(slot) for slot in sorted(selected_slots)]
  accepted = [record for record in records if record and record["status"] == "accepted"]
  failed = [
    record for record in records if not record or record["status"] != "accepted"
  ]
  manifest["completed_counts"] = {
    "requested": len(specs),
    "accepted": len(accepted),
    "failed": len(failed),
  }
  distribution = amplitude_distribution(records, cfg)
  manifest["amplitude_distribution"] = distribution
  is_full_run = cfg.max_clips == 0 and len(specs) == len(all_specs)
  inventory_passed = not failed and len(accepted) == len(specs)
  manifest["gate_passed"] = is_full_run and inventory_passed and distribution["passed"]
  manifest["status"] = (
    "complete"
    if manifest["gate_passed"]
    else "smoke_complete"
    if not is_full_run and inventory_passed
    else "blocked"
  )
  manifest["finished_at_utc"] = now()
  atomic_json(manifest_path, manifest)
  print(json.dumps(manifest["completed_counts"]), flush=True)
  if failed:
    raise RuntimeError(f"{len(failed)} slots failed the frozen gates")
  if manifest["gate_passed"]:
    dataset = finalize_dataset(output)
    print(
      json.dumps(
        {
          "dataset_status": dataset["status"],
          "condition_counts": dataset["condition_counts"],
        }
      ),
      flush=True,
    )


if __name__ == "__main__":
  main(tyro.cli(Cfg))
