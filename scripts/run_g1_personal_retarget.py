"""Run the EXP-034 personal recording retargeting pipeline on CPU."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mink
import numpy as np
import tyro

from smp.data.personal_dataset import export_dataset
from smp.data.personal_retarget import (
  SCHEMA,
  STANCES,
  SUBJECT_SPLITS,
  Config,
  PersonalRetargeter,
  SimulationFailure,
  config_dict,
  coverage,
  inventory,
  load_source,
  verify_attempt,
  write_attempt,
)
from smp.data.shoulder_ik import atomic_json, sha256


@dataclass(frozen=True)
class Cfg:
  stage: str = "smoke"
  input_dir: str = "datasets/G1-GVHMR-GMR"
  output_dir: str = "logs/reproduction/g1_personal_retarget_v1"
  workers: int = 3
  seed: int = 42
  resume: bool = True


def now():
  return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(value):
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def storage(path: Path):
  if path.is_symlink():
    physical = path.resolve(strict=True)
    if not physical.is_relative_to(Path("/personal_fast_storage/smp_experiments")):
      raise ValueError(f"Unexpected experiment storage: {physical}")
  elif path.exists():
    raise ValueError(f"Output exists and is not an experiment symlink: {path}")
  else:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    physical = Path("/personal_fast_storage/smp_experiments") / f"{SCHEMA}_{stamp}"
    physical.mkdir(parents=True, exist_ok=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(physical, target_is_directory=True)
  if shutil.disk_usage(physical).free < 1024**3:
    raise OSError("Less than 1 GiB persistent space available")
  probe = physical / "write_probe.tmp"
  with probe.open("wb") as stream:
    stream.write(b"EXP-034")
    stream.flush()
    os.fsync(stream.fileno())
  probe.unlink()
  return physical


_GENERATOR = None


def process_clip(root: str, output_string: str, item: dict, cfg: Config):
  global _GENERATOR
  if _GENERATOR is None:
    _GENERATOR = PersonalRetargeter(cfg)
  generator = _GENERATOR
  output = Path(output_string)
  source = load_source(Path(root), item, generator.sim_model, cfg)
  record = {
    "source_clip_id": source.clip_id,
    "subject_id": source.metadata["subject_id"],
    "condition_id": source.condition_id,
    "split": source.split,
    "status": "rejected",
    "attempts": [],
  }
  for attempt in range(len(STANCES)):
    folder = output / "attempts" / source.clip_id / f"attempt_{attempt}"
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
      metadata = json.loads(metadata_path.read_text())
      if metadata["source_identity"] != source.identity:
        raise ValueError("Attempt source identity differs")
      result = {
        "attempt": attempt,
        "passed": bool(metadata["metrics"].get("passed")),
        "metadata": str(metadata_path.relative_to(output)),
        "metadata_sha256": sha256(metadata_path),
        "failed_checks": metadata["metrics"].get("failed_checks", []),
      }
      verify_attempt(output, result)
    else:
      reference = None
      try:
        reference, ik = generator.reference(source, attempt)
        execution, audit = generator.execute_checked(reference)
        metrics = generator.validate(source, reference, execution, ik, audit)
        result = write_attempt(
          output, source, attempt, reference, execution, metrics, generator.sim_model
        )
      except (
        RuntimeError,
        ValueError,
        FloatingPointError,
        mink.NoSolutionFound,
      ) as error:
        folder.mkdir(parents=True, exist_ok=True)
        metrics = {
          "passed": False,
          "exception": f"{type(error).__name__}: {error}",
          "failed_checks": ["solver_or_simulation_failure"],
        }
        if reference is not None:
          result = write_attempt(
            output, source, attempt, reference, None, metrics, generator.sim_model
          )
          if isinstance(error, SimulationFailure):
            fault = folder / "fault_state.npz"
            np.savez_compressed(
              fault, qpos=error.qpos, qvel=error.qvel, not_training_input=np.bool_(True)
            )
            metadata = json.loads(metadata_path.read_text())
            metadata["files"]["fault_state.npz"] = {
              "path": str(fault.relative_to(output)),
              "sha256": sha256(fault),
            }
            atomic_json(metadata_path, metadata)
            result["metadata_sha256"] = sha256(metadata_path)
        else:
          atomic_json(
            metadata_path,
            {"source_identity": source.identity, "metrics": metrics, "files": {}},
          )
          result = {
            "attempt": attempt,
            "passed": False,
            "metadata": str(metadata_path.relative_to(output)),
            "metadata_sha256": sha256(metadata_path),
            "failed_checks": metrics["failed_checks"],
          }
    record["attempts"].append(result)
    print(
      json.dumps(
        {
          "clip": source.clip_id,
          "attempt": attempt,
          "passed": result["passed"],
          "failed_checks": result["failed_checks"],
        }
      ),
      flush=True,
    )
    if result["passed"]:
      record.update(status="accepted", accepted=result)
      break
  return record


def experiment_identity(source: Path, items: list, cfg: Config):
  paths = (
    "src/smp/data/personal_retarget.py",
    "src/smp/data/personal_dataset.py",
    "src/smp/data/shoulder_ik.py",
    "src/smp/data/shoulder_ik_dataset.py",
    "scripts/run_g1_personal_retarget.py",
  )
  return {
    "schema": SCHEMA,
    "config": config_dict(cfg),
    "source_dir": str(source.resolve()),
    "source_manifest_sha256": sha256(source / "manifest.csv"),
    "sources": items,
    "code": {p: sha256(Path(p)) for p in paths},
    "dependencies": {
      p: version(p) for p in ("mink", "mujoco", "numpy", "scipy", "mjlab")
    },
  }


def save_summary(output, manifest):
  covered = coverage(manifest["clips"])
  reasons = {}
  metrics = []
  for record in manifest["clips"].values():
    for attempt in record["attempts"]:
      metadata = verify_attempt(output, attempt)
      for check in attempt["failed_checks"]:
        reasons[check] = reasons.get(check, 0) + 1
      if attempt["passed"]:
        metrics.append(metadata["metrics"])
  bounds = {}
  if metrics:
    for key in metrics[0]:
      values = [
        x[key]
        for x in metrics
        if key in x
        and isinstance(x[key], (int, float))
        and not isinstance(x[key], bool)
      ]
      if values:
        bounds[key] = {
          "min": float(np.min(values)),
          "p50": float(np.median(values)),
          "max": float(np.max(values)),
        }
  report = {
    "experiment": "EXP-034",
    "created_at_beijing": now(),
    "status": manifest["status"],
    "processed": len(manifest["clips"]),
    "accepted": sum(x["status"] == "accepted" for x in manifest["clips"].values()),
    "coverage": covered,
    "failed_checks_by_attempt": reasons,
    "accepted_metric_ranges": bounds,
    "prior_or_ppo_started": False,
  }
  atomic_json(output / "report.json", report)


def main(args: Cfg):
  if args.stage not in ("smoke", "full", "export", "report") or args.workers < 1:
    raise ValueError("stage must be smoke/full/export/report; workers must be positive")
  source = Path(args.input_dir).resolve()
  output = Path(args.output_dir).absolute()
  physical = storage(output)
  lock = (output / "driver.lock").open("a+")
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  cfg = Config(seed=args.seed)
  items = inventory(source)
  identity = experiment_identity(source, items, cfg)
  identity_sha = digest(identity)
  path = output / "manifest.json"
  if path.exists():
    if not args.resume:
      raise ValueError("Output exists; resume is required")
    manifest = json.loads(path.read_text())
    if manifest["identity_sha256"] != identity_sha:
      raise ValueError(
        "Frozen code/config/source identity differs; use an independent output directory"
      )
    for record in manifest["clips"].values():
      for attempt in record["attempts"]:
        verify_attempt(output, attempt)
    manifest.setdefault("resume_history", []).append(
      {"time_beijing": now(), "stage": args.stage, "processed": len(manifest["clips"])}
    )
  else:
    manifest = {
      "schema_version": 1,
      "experiment": "EXP-034",
      "status": "created",
      "identity": identity,
      "identity_sha256": identity_sha,
      "created_at_beijing": now(),
      "storage": {"logical": str(output), "physical": str(physical)},
      "clips": {},
      "resume_history": [],
      "smoke": {"passed": False},
      "prior_or_ppo_started": False,
    }
    snapshot = output / "sources"
    snapshot.mkdir()
    for item in items:
      clip_id = item["metadata"]["clip_id"]
      for extension in ("json", "csv", "npz"):
        shutil.copy2(
          source / f"{clip_id}.{extension}", snapshot / f"{clip_id}.{extension}"
        )
    shutil.copy2(source / "manifest.csv", snapshot / "manifest.csv")
  manifest["updated_at_beijing"] = now()
  atomic_json(path, manifest)
  if args.stage == "export":
    report = export_dataset(output)
    manifest["dataset"] = {
      "manifest": "dataset/manifest.json",
      "sha256": sha256(output / "dataset/manifest.json"),
      "clips": report["clips"],
    }
    manifest["status"] = "published"
    atomic_json(path, manifest)
    return
  if args.stage == "report":
    save_summary(output, manifest)
    return
  if args.stage == "full" and not manifest["smoke"]["passed"]:
    raise RuntimeError("A passed nine-clip smoke is required before full processing")
  selected = items
  if args.stage == "smoke":
    selected = [
      next(
        x
        for x in items
        if x["metadata"]["subject_id"] == subject
        and x["metadata"]["condition_id"] == condition
      )
      for subject in SUBJECT_SPLITS
      for condition in range(3)
    ]
  manifest["status"] = f"running_{args.stage}"
  atomic_json(path, manifest)
  pending = [
    item for item in selected if item["metadata"]["clip_id"] not in manifest["clips"]
  ]
  try:
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
      futures = {
        pool.submit(process_clip, str(source), str(output), item, cfg): item
        for item in pending
      }
      for future in as_completed(futures):
        record = future.result()
        manifest["clips"][record["source_clip_id"]] = record
        manifest["updated_at_beijing"] = now()
        atomic_json(path, manifest)
        save_summary(output, manifest)
    if args.stage == "smoke":
      names = [x["metadata"]["clip_id"] for x in selected]
      passed = all(manifest["clips"][name]["status"] == "accepted" for name in names)
      manifest["smoke"] = {
        "passed": passed,
        "source_clips": names,
        "completed_at_beijing": now(),
      }
      manifest["status"] = "smoke_passed" if passed else "blocked_smoke"
    else:
      manifest["status"] = (
        "physics_complete_pending_review"
        if coverage(manifest["clips"])["passed"]
        else "blocked_coverage"
      )
    manifest["updated_at_beijing"] = now()
    # Recheck source identity at the end as well as before loading workers.
    if inventory(source) != items:
      raise ValueError("Source inputs changed while processing")
    atomic_json(path, manifest)
    save_summary(output, manifest)
    print(
      json.dumps(
        {
          "status": manifest["status"],
          "output": str(output),
          "coverage": coverage(manifest["clips"]),
        }
      ),
      flush=True,
    )
  except BaseException as error:
    manifest["status"] = "interrupted"
    manifest["last_exception"] = f"{type(error).__name__}: {error}"
    atomic_json(path, manifest)
    raise


if __name__ == "__main__":
  main(tyro.cli(Cfg))
