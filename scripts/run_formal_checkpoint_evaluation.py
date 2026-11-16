"""Run restartable formal evaluations for explicitly selected checkpoints.

The suite evaluates one fixed chest target, both shoulder targets, and all five
conditional-contact modes.  Each condition receives an independent deterministic
1024-episode report and a 600-control-frame replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC80_PRIOR = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_static80.pt"
STATIC80_PRIOR_SHA256 = (
  "2f9682f9d1e3e4a5298576f637736345e69fe9fea510992d4b9f7afd5f430dec"
)
CONDITIONAL_PRIOR = "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
CONDITIONAL_PRIOR_SHA256 = (
  "ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734"
)


@dataclass(frozen=True)
class SuiteCfg:
  output_dir: str = "logs/reproduction/formal_evaluation_20261112"
  num_episodes: int = 1024
  num_envs: int = 256
  video_length: int = 600
  seed: int = 42
  gpu_id: int = 0


@dataclass(frozen=True)
class CheckpointSpec:
  name: str
  path: str
  sha256: str
  training_stage: str
  selection_note: str


@dataclass(frozen=True)
class EvaluationSpec:
  name: str
  checkpoint_name: str
  task_id: str
  stage: str
  prior: str
  stationary_reward_weight: float
  target_probabilities: tuple[float, ...]
  reference_speed: float
  max_force_weight_ratio: float
  ramp_time: float
  wait_range_s: tuple[float, float]


CHECKPOINTS = (
  CheckpointSpec(
    name="speed_chest_model31995",
    path=(
      "logs/reproduction/rsl_rl/smp_force_unloading_g1/"
      "2026-11-10_17-32-59_gsi015_decay10_static80_r010_"
      "v005_010_020_chest_c_seed42/model_31995.pt"
    ),
    sha256="82ddfe95ee383cd4cf62c65d6f080a46138ad43fbdd92f19e28d9272ea4e0d33",
    training_stage="c",
    selection_note="completed speed-gradient chest curriculum checkpoint",
  ),
  CheckpointSpec(
    name="speed_shoulders_model12000",
    path=(
      "logs/reproduction/rsl_rl/smp_force_unloading_g1/"
      "2026-11-11_04-42-14_gsi015_decay10_static80_r010_"
      "v005_010_020_shoulders_b2_seed42/model_12000.pt"
    ),
    sha256="ea925cc60875137dd9ffc4f70e799b20c1c7ebd8151743c2f3a70aab446ff648",
    training_stage="b2",
    selection_note="user-selected intermediate B2 checkpoint; not full-budget final",
  ),
  CheckpointSpec(
    name="conditional_model17000",
    path=(
      "logs/reproduction/conditional_contact_v2/rsl_rl/"
      "smp_force_unloading_conditional_g1/"
      "2026-11-11_08-41-56_conditional_contact_v2_c_seed42/model_17000.pt"
    ),
    sha256="9bfb8c308f143d10771d5828b97b043c46e866a1dcb0a6e3d1227b93217ec29e",
    training_stage="c",
    selection_note="user-selected intermediate C checkpoint; not full-budget final",
  ),
)


EVALUATIONS = (
  EvaluationSpec(
    "speed_chest_model31995_chest",
    "speed_chest_model31995",
    "Smp-Force-Unloading-G1",
    "c",
    STATIC80_PRIOR,
    0.1,
    (1.0, 0.0, 0.0),
    0.20,
    0.15,
    0.25,
    (1.0, 3.0),
  ),
  EvaluationSpec(
    "speed_shoulders_model12000_left",
    "speed_shoulders_model12000",
    "Smp-Force-Unloading-G1",
    "b2",
    STATIC80_PRIOR,
    0.1,
    (0.0, 1.0, 0.0),
    0.10,
    0.10,
    0.50,
    (1.0, 3.0),
  ),
  EvaluationSpec(
    "speed_shoulders_model12000_right",
    "speed_shoulders_model12000",
    "Smp-Force-Unloading-G1",
    "b2",
    STATIC80_PRIOR,
    0.1,
    (0.0, 0.0, 1.0),
    0.10,
    0.10,
    0.50,
    (1.0, 3.0),
  ),
  EvaluationSpec(
    "conditional_model17000_stand",
    "conditional_model17000",
    "Smp-Force-Unloading-Conditional-G1",
    "a",
    CONDITIONAL_PRIOR,
    0.9,
    (1.0, 0.0, 0.0, 0.0),
    0.05,
    0.15,
    0.25,
    (1e9, 1e9),
  ),
  EvaluationSpec(
    "conditional_model17000_left_shoulder",
    "conditional_model17000",
    "Smp-Force-Unloading-Conditional-G1",
    "c",
    CONDITIONAL_PRIOR,
    0.1,
    (0.0, 1.0, 0.0, 0.0),
    0.05,
    0.15,
    0.25,
    (1.0, 3.0),
  ),
  EvaluationSpec(
    "conditional_model17000_right_shoulder",
    "conditional_model17000",
    "Smp-Force-Unloading-Conditional-G1",
    "c",
    CONDITIONAL_PRIOR,
    0.1,
    (0.0, 0.0, 1.0, 0.0),
    0.05,
    0.15,
    0.25,
    (1.0, 3.0),
  ),
  EvaluationSpec(
    "conditional_model17000_chest",
    "conditional_model17000",
    "Smp-Force-Unloading-Conditional-G1",
    "c",
    CONDITIONAL_PRIOR,
    0.1,
    (1.0, 0.0, 0.0, 0.0),
    0.05,
    0.15,
    0.25,
    (1.0, 3.0),
  ),
  EvaluationSpec(
    "conditional_model17000_back",
    "conditional_model17000",
    "Smp-Force-Unloading-Conditional-G1",
    "c",
    CONDITIONAL_PRIOR,
    0.1,
    (0.0, 0.0, 0.0, 1.0),
    0.05,
    0.15,
    0.25,
    (1.0, 3.0),
  ),
)


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  os.replace(temporary, path)


def _gpu_identity(gpu_id: int) -> dict[str, Any]:
  result = subprocess.run(
    [
      "nvidia-smi",
      "-i",
      str(gpu_id),
      "--query-gpu=index,uuid,name,memory.total",
      "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  index, uuid, name, memory_mib = (
    item.strip() for item in result.stdout.strip().split(",", maxsplit=3)
  )
  return {
    "physical_index": int(index),
    "uuid": uuid,
    "name": name,
    "memory_mib": int(memory_mib),
    "process_device": "cuda:0",
  }


def _validate_inputs() -> dict[str, CheckpointSpec]:
  lookup = {spec.name: spec for spec in CHECKPOINTS}
  expected = {
    STATIC80_PRIOR: STATIC80_PRIOR_SHA256,
    CONDITIONAL_PRIOR: CONDITIONAL_PRIOR_SHA256,
    **{spec.path: spec.sha256 for spec in CHECKPOINTS},
  }
  for relative_path, expected_sha in expected.items():
    path = REPO_ROOT / relative_path
    if not path.is_file():
      raise FileNotFoundError(path)
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
      raise RuntimeError(
        f"SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
      )
  return lookup


def _command(
  cfg: SuiteCfg,
  spec: EvaluationSpec,
  checkpoint: CheckpointSpec,
  output: Path,
  video_dir: Path,
) -> list[str]:
  return [
    sys.executable,
    "scripts/evaluate_force_unloading.py",
    "--task-id",
    spec.task_id,
    "--stationary-reward-weight",
    str(spec.stationary_reward_weight),
    "--checkpoint",
    str((REPO_ROOT / checkpoint.path).resolve()),
    "--output",
    str(output),
    "--stage",
    spec.stage,
    "--prior-checkpoint",
    str((REPO_ROOT / spec.prior).resolve()),
    "--gsi-max-root-planar-speed",
    "0.15",
    "--stationary-speed-scale",
    "10.0",
    "--sphere-radius",
    "0.10",
    "--target-probabilities",
    *(str(value) for value in spec.target_probabilities),
    "--push-direction-mode",
    "front_to_back",
    "--num-episodes",
    str(cfg.num_episodes),
    "--num-envs",
    str(cfg.num_envs),
    "--seed",
    str(cfg.seed),
    "--device",
    "cuda:0",
    "--reference-speed",
    str(spec.reference_speed),
    "--max-force-weight-ratio",
    str(spec.max_force_weight_ratio),
    "--ramp-time",
    str(spec.ramp_time),
    "--wait-range-s",
    *(str(value) for value in spec.wait_range_s),
    "--video",
    "--video-length",
    str(cfg.video_length),
    "--video-folder",
    str(video_dir),
  ]


def _valid_completed_job(
  output: Path, video_dir: Path, cfg: SuiteCfg
) -> tuple[bool, dict[str, Any] | None]:
  if not output.is_file():
    return False, None
  report = json.loads(output.read_text(encoding="utf-8"))
  videos = sorted(video_dir.glob("*.mp4"))
  valid = (
    report.get("finite") is True
    and report.get("completed_episodes") == cfg.num_episodes
    and bool(videos)
    and all(video.stat().st_size > 0 for video in videos)
  )
  return valid, report if valid else None


def _build_summary(
  cfg: SuiteCfg,
  output_dir: Path,
  manifest: dict[str, Any],
) -> dict[str, Any]:
  reports: dict[str, Any] = {}
  checks: dict[str, bool] = {}
  for spec in EVALUATIONS:
    path = output_dir / "reports" / f"{spec.name}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    reports[spec.name] = {
      "report": str(path.resolve()),
      "video_dir": str((output_dir / "replays" / spec.name).resolve()),
      "gate": report["gate"],
      "metrics": report["metrics"],
    }
    checks.update(
      {
        f"{spec.name}/{name}": bool(value)
        for name, value in report["gate"]["checks"].items()
      }
    )
  return {
    "schema_version": 1,
    "created_at_utc": _utc_now(),
    "config": asdict(cfg),
    "hardware": manifest["hardware"],
    "checkpoint_caveat": (
      "model_12000.pt and model_17000.pt are explicitly selected intermediate "
      "checkpoints and are not described as full-budget final policies."
    ),
    "checkpoints": manifest["checkpoints"],
    "finite": True,
    "all_gates_passed": all(checks.values()),
    "gate_checks": checks,
    "evaluations": reports,
  }


def run(cfg: SuiteCfg) -> None:
  checkpoint_lookup = _validate_inputs()
  output_dir = (REPO_ROOT / cfg.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = output_dir / "manifest.json"
  hardware = _gpu_identity(cfg.gpu_id)
  manifest: dict[str, Any]
  if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  else:
    manifest = {
      "schema_version": 1,
      "status": "running",
      "created_at_utc": _utc_now(),
      "hardware": hardware,
      "config": asdict(cfg),
      "checkpoints": [asdict(spec) for spec in CHECKPOINTS],
      "priors": {
        STATIC80_PRIOR: STATIC80_PRIOR_SHA256,
        CONDITIONAL_PRIOR: CONDITIONAL_PRIOR_SHA256,
      },
      "jobs": {},
    }
  if manifest["hardware"]["uuid"] != hardware["uuid"]:
    raise RuntimeError("Evaluation manifest belongs to a different GPU UUID")
  manifest["config"] = asdict(cfg)
  manifest["updated_at_utc"] = _utc_now()
  _atomic_json(manifest_path, manifest)

  environment = os.environ.copy()
  environment["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
  environment["MUJOCO_GL"] = "egl"
  environment["MUJOCO_EGL_DEVICE_ID"] = "0"
  environment["PYTHONUNBUFFERED"] = "1"

  for spec in EVALUATIONS:
    output = output_dir / "reports" / f"{spec.name}.json"
    video_dir = output_dir / "replays" / spec.name
    valid, _ = _valid_completed_job(output, video_dir, cfg)
    if valid:
      print(f"[FORMAL EVAL] already complete: {spec.name}", flush=True)
      continue
    checkpoint = checkpoint_lookup[spec.checkpoint_name]
    command = _command(cfg, spec, checkpoint, output, video_dir)
    log_path = output_dir / "logs" / f"{spec.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
      "status": "running",
      "started_at_utc": _utc_now(),
      "command": command,
      "checkpoint": asdict(checkpoint),
      "evaluation": asdict(spec),
      "report": str(output),
      "video_dir": str(video_dir),
      "log": str(log_path),
    }
    manifest["jobs"][spec.name] = record
    manifest["updated_at_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    print(f"[FORMAL EVAL] {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
      log.write(f"\n[{_utc_now()}] {shlex.join(command)}\n")
      log.flush()
      process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
      record["pid"] = process.pid
      _atomic_json(manifest_path, manifest)
      return_code = process.wait()
    valid, report = _valid_completed_job(output, video_dir, cfg)
    record.update(
      {
        "status": "complete" if return_code == 0 and valid else "failed",
        "finished_at_utc": _utc_now(),
        "exit_code": return_code,
        "gate": report["gate"] if report else None,
      }
    )
    manifest["updated_at_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    if record["status"] != "complete":
      manifest["status"] = "failed"
      _atomic_json(manifest_path, manifest)
      raise RuntimeError(f"Formal evaluation failed: {spec.name}")

  summary = _build_summary(cfg, output_dir, manifest)
  _atomic_json(output_dir / "summary.json", summary)
  manifest["status"] = "complete"
  manifest["finished_at_utc"] = _utc_now()
  manifest["summary"] = str((output_dir / "summary.json").resolve())
  _atomic_json(manifest_path, manifest)
  print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
  run(tyro.cli(SuiteCfg))
