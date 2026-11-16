"""Run reduced-radius chest and shoulder speed-gradient curricula on GPU0."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_static80.pt"
PRIOR_SHA256 = "2f9682f9d1e3e4a5298576f637736345e69fe9fea510992d4b9f7afd5f430dec"
PARENT_STAGE_A_MANIFEST = (
  "logs/reproduction/curriculum/gsi015_decay10_static80_chest_seed42/"
  "curriculum_state.json"
)
STATE_PATH = Path(
  "logs/reproduction/curriculum/"
  "gsi015_decay10_static80_r010_speed_gradient_gpu0_suite_state.json"
)
LOG_DIR = Path(
  "logs/reproduction/curriculum/gsi015_decay10_static80_r010_speed_gradient_gpu0_logs"
)
SPHERE_RADIUS = 0.10
SPEEDS = (0.05, 0.10, 0.20)
GPU_ID = 0


@dataclass(frozen=True)
class SuiteCfg:
  seed: int = 42
  dry_run: bool = False


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_state(state: dict[str, Any]) -> None:
  state["updated_at_utc"] = _utc_now()
  path = REPO_ROOT / STATE_PATH
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  os.replace(temporary, path)


def _gpu0_identity() -> dict[str, Any]:
  result = subprocess.run(
    [
      "nvidia-smi",
      "-i",
      "0",
      "--query-gpu=index,uuid,name",
      "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  index, uuid, name = (
    part.strip() for part in result.stdout.strip().split(",", maxsplit=2)
  )
  if int(index) != GPU_ID:
    raise RuntimeError(f"Expected physical GPU0, got GPU {index}")
  return {"physical_index": GPU_ID, "uuid": uuid, "name": name}


def curriculum_command(target: str, seed: int = 42) -> list[str]:
  if target not in {"chest", "shoulders"}:
    raise ValueError(target)
  suffix = "chest" if target == "chest" else "shoulders"
  prefix = f"gsi015_decay10_static80_r010_v005_010_020_{suffix}"
  command = [
    sys.executable,
    "scripts/train_force_unloading_curriculum.py",
    "--prior-checkpoint",
    PRIOR,
    "--prior-sha256",
    PRIOR_SHA256,
    "--seed",
    str(seed),
    "--gpu-id",
    "0",
    "--gsi-max-root-planar-speed",
    "0.15",
    "--stationary-speed-scale",
    "10.0",
    "--sphere-radius",
    str(SPHERE_RADIUS),
    "--contact-reference-speeds",
    *(str(speed) for speed in SPEEDS),
    "--target-probabilities",
    *(("1.0", "0.0", "0.0") if target == "chest" else ("0.0", "0.5", "0.5")),
    "--push-direction-mode",
    "front_to_back",
    "--parent-stage-a-manifest",
    PARENT_STAGE_A_MANIFEST,
    "--run-prefix",
    prefix,
    "--state-dir",
    f"logs/reproduction/curriculum/{prefix}_seed{seed}",
    "--evaluation-dir",
    "logs/reproduction/evaluation/gsi015_static80_r010_speed_gradient_gpu0",
    "--no-run-smoke",
  ]
  if target == "shoulders":
    command.append("--shoulder-split-evaluation")
  return command


def smoke_command(target: str, seed: int = 42) -> list[str]:
  return [
    sys.executable,
    "scripts/smoke_force_unloading_speed_gradient.py",
    "--target",
    target,
    "--sphere-radius",
    str(SPHERE_RADIUS),
    "--reference-speed",
    str(SPEEDS[-1]),
    "--device",
    "cuda:0",
    "--seed",
    str(seed),
  ]


def _run_step(
  state: dict[str, Any], name: str, command: list[str], *, dry_run: bool
) -> None:
  existing = state["steps"].get(name, {})
  if existing.get("status") == "complete":
    return
  attempt = int(existing.get("attempt", 0)) + 1
  log_path = (REPO_ROOT / LOG_DIR / f"{name}.log").resolve()
  log_path.parent.mkdir(parents=True, exist_ok=True)
  record = {
    "status": "running",
    "attempt": attempt,
    "started_at_utc": _utc_now(),
    "command": command,
    "physical_gpu": GPU_ID,
    "log": str(log_path),
  }
  state["steps"][name] = record
  _atomic_state(state)
  if dry_run:
    record["status"] = "dry_run"
    _atomic_state(state)
    return

  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = "0"
  env["MUJOCO_GL"] = "egl"
  env["MUJOCO_EGL_DEVICE_ID"] = "0"
  env["PYTHONUNBUFFERED"] = "1"
  with log_path.open("a", encoding="utf-8") as log:
    log.write(f"\n[{_utc_now()}] {' '.join(command)}\n")
    log.flush()
    process = subprocess.Popen(
      command,
      cwd=REPO_ROOT,
      env=env,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
    record["pid"] = process.pid
    _atomic_state(state)
    return_code = process.wait()
  record.update(
    {
      "status": "complete" if return_code == 0 else "failed",
      "finished_at_utc": _utc_now(),
      "exit_code": return_code,
    }
  )
  if return_code != 0:
    record["error"] = f"Child exited with code {return_code}"
  _atomic_state(state)
  if return_code != 0:
    raise RuntimeError(f"{name} failed with exit code {return_code}")


def run(cfg: SuiteCfg) -> None:
  prior = REPO_ROOT / PRIOR
  parent = REPO_ROOT / PARENT_STAGE_A_MANIFEST
  if _sha256(prior) != PRIOR_SHA256:
    raise RuntimeError("80% static prior SHA-256 mismatch")
  if not parent.is_file():
    raise FileNotFoundError(parent)
  parent_payload = json.loads(parent.read_text(encoding="utf-8"))
  parent_stages = parent_payload.get("stages", {})
  if parent_stages.get("a_extension", {}).get("status") != "complete":
    raise RuntimeError("Parent 80% static-prior A extension is not complete")

  state_path = REPO_ROOT / STATE_PATH
  if state_path.is_file():
    state = json.loads(state_path.read_text(encoding="utf-8"))
  else:
    state = {
      "schema_version": 1,
      "created_at_utc": _utc_now(),
      "steps": {},
    }
  state.update(
    {
      "status": "running",
      "config": asdict(cfg),
      "physical_gpu": _gpu0_identity(),
      "prior": {"path": str(prior.resolve()), "sha256": PRIOR_SHA256},
      "parent_stage_a_manifest": str(parent.resolve()),
      "sphere": {
        "radius_m": SPHERE_RADIUS,
        "baseline_radius_m": 0.175,
        "volume_ratio": (SPHERE_RADIUS / 0.175) ** 3,
      },
      "contact_reference_speeds_m_s": list(SPEEDS),
      "execution": "strictly serial on physical GPU0",
    }
  )
  _atomic_state(state)

  steps = (
    ("chest_contact_smoke", smoke_command("chest", cfg.seed)),
    ("shoulder_contact_smoke", smoke_command("shoulders", cfg.seed)),
    ("chest_speed_gradient", curriculum_command("chest", cfg.seed)),
    ("shoulder_speed_gradient", curriculum_command("shoulders", cfg.seed)),
  )
  for name, command in steps:
    _run_step(state, name, command, dry_run=cfg.dry_run)

  state["status"] = "dry_run" if cfg.dry_run else "complete"
  state["finished_at_utc"] = _utc_now()
  _atomic_state(state)


if __name__ == "__main__":
  run(tyro.cli(SuiteCfg))
