"""Run the three approved force-unloading experiments on two isolated GPUs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_PRIOR = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_balance.pt"
OLD_PRIOR_SHA256 = "5574dc5720a356937f6a9bc01b77f3d5e8456970d59e59c59b05e2ca6a92d0b7"
NEW_PRIOR = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_static80.pt"
NEW_PRIOR_SHA256 = (
  "2f9682f9d1e3e4a5298576f637736345e69fe9fea510992d4b9f7afd5f430dec"
)
SAMPLING_MANIFEST = "docs/LAFAN_LOW_SPEED_STATIC80_SAMPLING.json"
PRIOR_METADATA = (
  "datasets/pretrain_ckpt/pretrained_lafan_low_speed_static80.metadata.json"
)
SUITE_STATE = Path(
  "logs/reproduction/curriculum/gsi015_static80_shoulder_suite_state.json"
)


@dataclass(frozen=True)
class SuiteCfg:
  seed: int = 42
  run_prior: bool = True
  run_ppo: bool = True
  parallel_ppo: bool = True
  gpu_ids: tuple[int, int] = (0, 1)
  poll_interval_s: float = 5.0
  shutdown_timeout_s: float = 90.0
  dry_run: bool = False


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _safe_print(*args: Any, **kwargs: Any) -> None:
  try:
    print(*args, **kwargs)
  except BrokenPipeError:
    pass


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_state(state: dict[str, Any]) -> None:
  state["updated_at_utc"] = _utc_now()
  path = REPO_ROOT / SUITE_STATE
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  os.replace(temporary, path)


def gpu_inventory() -> dict[int, dict[str, Any]]:
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-gpu=index,uuid,name",
      "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  inventory: dict[int, dict[str, Any]] = {}
  for line in result.stdout.splitlines():
    index_text, uuid, name = (part.strip() for part in line.split(",", maxsplit=2))
    index = int(index_text)
    inventory[index] = {"physical_index": index, "uuid": uuid, "name": name}
  return inventory


def validate_gpu_ids(
  gpu_ids: tuple[int, int], inventory: dict[int, dict[str, Any]]
) -> None:
  if len(set(gpu_ids)) != 2:
    raise ValueError(f"Parallel PPO requires two distinct GPU ids, got {gpu_ids}")
  missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in inventory]
  if missing:
    raise RuntimeError(f"Requested physical GPU ids are unavailable: {missing}")


def terminal_stage_a(manifest_path: Path) -> str | None:
  """Return the completed terminal A stage, respecting conditional extension."""
  if not manifest_path.is_file():
    return None
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  stages = manifest.get("stages", {})
  extension = stages.get("a_extension", {})
  if extension.get("status") == "complete":
    return "a_extension"
  stage_a = stages.get("a", {})
  if stage_a.get("status") != "complete":
    return None
  if stage_a.get("gate_passed") is False:
    return None
  return "a"


def choose_experiment3_gpu(
  free_gpu_ids: list[int], *, parent_ready: bool
) -> int | None:
  if not parent_ready or not free_gpu_ids:
    return None
  return free_gpu_ids[0]


@dataclass
class RunningStep:
  name: str
  gpu_id: int
  process: subprocess.Popen[Any]
  log_stream: TextIO


def _run_step(
  state: dict[str, Any],
  name: str,
  command: list[str],
  *,
  dry_run: bool,
) -> None:
  record = state["steps"].setdefault(name, {})
  attempt = int(record.get("attempt", 0)) + 1
  record.clear()
  record.update(
    {
      "status": "running",
      "attempt": attempt,
      "started_at_utc": _utc_now(),
      "command": command,
    }
  )
  _write_state(state)
  _safe_print(f"[SUITE] {name}: {' '.join(command)}", flush=True)
  if dry_run:
    record["status"] = "dry_run"
    _write_state(state)
    return
  env = os.environ.copy()
  env["MUJOCO_GL"] = "egl"
  env["PYTHONUNBUFFERED"] = "1"
  try:
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
  except Exception as exc:
    record.update(
      {"status": "failed", "finished_at_utc": _utc_now(), "error": str(exc)}
    )
    _write_state(state)
    raise
  record.update({"status": "complete", "finished_at_utc": _utc_now()})
  _write_state(state)


def _pretrain_command(name: str, epochs: int, use_wandb: bool) -> list[str]:
  command = [
    sys.executable,
    "scripts/pretrain.py",
    "--data-dir",
    "datasets/npz/lafan_low_speed_balance",
    "--sampling-manifest-file",
    SAMPLING_MANIFEST,
    "--train-split",
    "1.0",
    "--d-model",
    "128",
    "--num-layers",
    "2",
    "--batch-size",
    "1024",
    "--num-epochs",
    str(epochs),
    "--num-noise-samples",
    "10",
    "--num-timesteps",
    "50",
    "--lr",
    "0.0003",
    "--save-interval",
    "5000" if epochs > 1 else "1",
    "--seed",
    "42",
    "--name",
    name,
    "--log-dir",
    "logs/reproduction/pretrain",
    "--no-use-ema",
    "--no-upload-model",
  ]
  if not use_wandb:
    command.append("--no-use-wandb")
  return command


def _ensure_static80_prior(state: dict[str, Any], cfg: SuiteCfg) -> str:
  stable = REPO_ROOT / NEW_PRIOR
  if not stable.exists():
    _run_step(
      state,
      "prior_smoke",
      _pretrain_command("lafan_low_speed_static80_smoke_seed42", 1, False),
      dry_run=cfg.dry_run,
    )
    _run_step(
      state,
      "prior_train",
      _pretrain_command("lafan_low_speed_static80_seed42", 10_000, True),
      dry_run=cfg.dry_run,
    )
    if cfg.dry_run:
      return ""
    candidates = list(
      (REPO_ROOT / "logs/reproduction/pretrain/lafan_low_speed_static80_seed42").glob(
        "*/pretrained.pt"
      )
    )
    if not candidates:
      raise RuntimeError("Static80 prior training produced no pretrained.pt")
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, stable)
    state["new_prior_source"] = str(source.resolve())
    _write_state(state)

  sha = _sha256(stable)
  sampling_path = REPO_ROOT / SAMPLING_MANIFEST
  metadata_path = REPO_ROOT / PRIOR_METADATA
  metadata = {
    "schema_version": 1,
    "checkpoint": str(stable.resolve()),
    "checkpoint_sha256": sha,
    "source": state.get("new_prior_source"),
    "sampling_manifest": {
      "path": str(sampling_path.resolve()),
      "sha256": _sha256(sampling_path),
      "config": json.loads(sampling_path.read_text(encoding="utf-8")),
    },
    "training": {
      "seed": 42,
      "epochs": 10_000,
      "batch_size": 1024,
      "d_model": 128,
      "num_layers": 2,
      "diffusion_steps": 50,
      "noise_samples": 10,
      "learning_rate": 3.0e-4,
      "ema": False,
      "samples_per_epoch": 3360,
    },
  }
  metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  state["new_prior"] = {
    "path": str(stable.resolve()),
    "sha256": sha,
    "metadata": str(metadata_path.resolve()),
  }
  _write_state(state)
  _run_step(
    state,
    "prior_validate",
    [
      sys.executable,
      "scripts/validate_low_speed_prior.py",
      "--ckpt-path",
      NEW_PRIOR,
      "--reference-ckpt-path",
      OLD_PRIOR,
      "--output",
      "logs/reproduction/data/low_speed_static80_prior_validation.json",
      "--max-root-planar-speed",
      "0.15",
      "--min-gsi-acceptance",
      "0.5",
      "--min-p95-reduction=-1.0",
      "--seed",
      "42",
    ],
    dry_run=cfg.dry_run,
  )
  return sha


def _curriculum_command(
  *,
  prior: str,
  prior_sha: str,
  prefix: str,
  state_dir: str,
  gpu_id: int = 0,
  extra: list[str] | None = None,
) -> list[str]:
  command = [
    sys.executable,
    "scripts/train_force_unloading_curriculum.py",
    "--prior-checkpoint",
    prior,
    "--prior-sha256",
    prior_sha,
    "--seed",
    "42",
    "--gpu-id",
    str(gpu_id),
    "--gsi-max-root-planar-speed",
    "0.15",
    "--stationary-speed-scale",
    "10.0",
    "--run-prefix",
    prefix,
    "--state-dir",
    state_dir,
    "--evaluation-dir",
    "logs/reproduction/evaluation/gsi015_static80_shoulder_suite",
  ]
  if extra:
    command.extend(extra)
  return command


def _start_async_step(
  state: dict[str, Any],
  name: str,
  command: list[str],
  gpu_id: int,
  inventory: dict[int, dict[str, Any]],
) -> RunningStep:
  record = state["steps"].setdefault(name, {})
  attempt = int(record.get("attempt", 0)) + 1
  log_dir = REPO_ROOT / SUITE_STATE.parent / "gsi015_static80_shoulder_suite_logs"
  log_dir.mkdir(parents=True, exist_ok=True)
  log_path = log_dir / f"{name}.log"
  record.clear()
  record.update(
    {
      "status": "running",
      "attempt": attempt,
      "started_at_utc": _utc_now(),
      "command": command,
      "gpu": inventory[gpu_id],
      "log": str(log_path.resolve()),
    }
  )
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
  env["MUJOCO_GL"] = "egl"
  env["MUJOCO_EGL_DEVICE_ID"] = "0"
  env["PYTHONUNBUFFERED"] = "1"
  log_stream = log_path.open("a", encoding="utf-8")
  log_stream.write(f"\n[{_utc_now()}] {' '.join(command)}\n")
  log_stream.flush()
  process = subprocess.Popen(
    command,
    cwd=REPO_ROOT,
    env=env,
    stdout=log_stream,
    stderr=subprocess.STDOUT,
    start_new_session=True,
  )
  record["pid"] = process.pid
  _write_state(state)
  _safe_print(
    f"[SUITE] started {name} on physical GPU {gpu_id}, pid={process.pid}",
    flush=True,
  )
  return RunningStep(name, gpu_id, process, log_stream)


def _finish_async_step(
  state: dict[str, Any], step: RunningStep, return_code: int
) -> bool:
  step.log_stream.flush()
  step.log_stream.close()
  record = state["steps"][step.name]
  success = return_code == 0
  record.update(
    {
      "status": "complete" if success else "failed",
      "finished_at_utc": _utc_now(),
      "exit_code": return_code,
    }
  )
  if not success:
    record["error"] = f"Child exited with code {return_code}"
  _write_state(state)
  _safe_print(
    f"[SUITE] {step.name} finished on GPU {step.gpu_id} "
    f"with exit code {return_code}",
    flush=True,
  )
  return success


def _interrupt_running_steps(
  state: dict[str, Any],
  running: dict[str, RunningStep],
  timeout_s: float,
) -> None:
  for step in running.values():
    if step.process.poll() is None:
      try:
        os.killpg(step.process.pid, signal.SIGINT)
      except ProcessLookupError:
        pass
  deadline = time.monotonic() + timeout_s
  for step in running.values():
    if step.process.poll() is None:
      try:
        step.process.wait(timeout=max(0.0, deadline - time.monotonic()))
      except subprocess.TimeoutExpired:
        try:
          os.killpg(step.process.pid, signal.SIGTERM)
        except ProcessLookupError:
          pass
        try:
          step.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
          try:
            os.killpg(step.process.pid, signal.SIGKILL)
          except ProcessLookupError:
            pass
          step.process.wait()
    if not step.log_stream.closed:
      step.log_stream.flush()
      step.log_stream.close()
    state["steps"][step.name].update(
      {
        "status": "interrupted",
        "recoverable": True,
        "finished_at_utc": _utc_now(),
        "exit_code": step.process.returncode,
      }
    )
  _write_state(state)


def _record_parallel_dry_run(
  state: dict[str, Any], jobs: list[tuple[str, int, list[str]]]
) -> None:
  for name, gpu_id, command in jobs:
    state["steps"][name] = {
      "status": "dry_run",
      "gpu_id": gpu_id,
      "command": command,
    }
  state["status"] = "dry_run"
  _write_state(state)


def _run_parallel_ppo(
  state: dict[str, Any], cfg: SuiteCfg, new_prior_sha: str
) -> None:
  inventory = gpu_inventory()
  validate_gpu_ids(cfg.gpu_ids, inventory)
  gpu0, gpu1 = cfg.gpu_ids
  exp1_state = "logs/reproduction/curriculum/gsi015_decay10_static50_chest_seed42"
  exp2_state = "logs/reproduction/curriculum/gsi015_decay10_static80_chest_seed42"
  exp3_state = "logs/reproduction/curriculum/gsi015_decay10_static80_shoulder_seed42"
  parent_manifest = REPO_ROOT / exp2_state / "curriculum_state.json"

  exp1_command = _curriculum_command(
    prior=OLD_PRIOR,
    prior_sha=OLD_PRIOR_SHA256,
    prefix="gsi015_decay10_static50_chest",
    state_dir=exp1_state,
    gpu_id=gpu0,
  )
  exp2_command = _curriculum_command(
    prior=NEW_PRIOR,
    prior_sha=new_prior_sha,
    prefix="gsi015_decay10_static80_chest",
    state_dir=exp2_state,
    gpu_id=gpu1,
  )

  def exp3_command(gpu_id: int) -> list[str]:
    return _curriculum_command(
      prior=NEW_PRIOR,
      prior_sha=new_prior_sha,
      prefix="gsi015_decay10_static80_shoulders",
      state_dir=exp3_state,
      gpu_id=gpu_id,
      extra=[
        "--target-probabilities",
        "0.0",
        "0.5",
        "0.5",
        "--push-direction-mode",
        "front_to_back",
        "--parent-stage-a-manifest",
        f"{exp2_state}/curriculum_state.json",
        "--shoulder-split-evaluation",
        "--no-run-smoke",
      ],
    )

  smoke_command = [sys.executable, "scripts/smoke_force_unloading_shoulder.py"]
  state["scheduler"] = {
    "mode": "dual_gpu_event",
    "physical_gpus": [inventory[gpu0], inventory[gpu1]],
    "experiment_3_dependency": str(parent_manifest.resolve()),
  }
  state["status"] = "running"
  _write_state(state)

  if cfg.dry_run:
    _record_parallel_dry_run(
      state,
      [
        ("experiment_1_static50_chest", gpu0, exp1_command),
        ("experiment_2_static80_chest", gpu1, exp2_command),
        ("experiment_3_shoulder_contact_smoke", gpu0, smoke_command),
        ("experiment_3_static80_shoulders", gpu0, exp3_command(gpu0)),
      ],
    )
    return

  running: dict[str, RunningStep] = {}
  free_gpu_ids: list[int] = []
  exp3_smoke_started = False
  exp3_started = False
  exp3_blocked = False
  running["experiment_1_static50_chest"] = _start_async_step(
    state,
    "experiment_1_static50_chest",
    exp1_command,
    gpu0,
    inventory,
  )
  running["experiment_2_static80_chest"] = _start_async_step(
    state,
    "experiment_2_static80_chest",
    exp2_command,
    gpu1,
    inventory,
  )

  try:
    while running:
      for name, step in list(running.items()):
        return_code = step.process.poll()
        if return_code is None:
          continue
        success = _finish_async_step(state, step, return_code)
        del running[name]
        free_gpu_ids.append(step.gpu_id)
        if name == "experiment_3_shoulder_contact_smoke":
          if success:
            free_gpu_ids.remove(step.gpu_id)
            exp3_started = True
            running["experiment_3_static80_shoulders"] = _start_async_step(
              state,
              "experiment_3_static80_shoulders",
              exp3_command(step.gpu_id),
              step.gpu_id,
              inventory,
            )
          else:
            exp3_blocked = True
            state["steps"]["experiment_3_static80_shoulders"] = {
              "status": "blocked",
              "reason": "shoulder contact smoke failed",
            }
            _write_state(state)

      parent_stage = terminal_stage_a(parent_manifest)
      parent_ready = parent_stage is not None
      if parent_ready:
        state["scheduler"]["experiment_3_parent_stage"] = parent_stage
        _write_state(state)
      if not exp3_smoke_started and not exp3_blocked:
        selected_gpu = choose_experiment3_gpu(
          free_gpu_ids, parent_ready=parent_ready
        )
        if selected_gpu is not None:
          free_gpu_ids.remove(selected_gpu)
          exp3_smoke_started = True
          running["experiment_3_shoulder_contact_smoke"] = _start_async_step(
            state,
            "experiment_3_shoulder_contact_smoke",
            smoke_command,
            selected_gpu,
            inventory,
          )

      exp2_record = state["steps"].get("experiment_2_static80_chest", {})
      exp2_finished = exp2_record.get("status") in {"complete", "failed"}
      if exp2_finished and not parent_ready and not exp3_smoke_started:
        exp3_blocked = True
        state["steps"]["experiment_3_static80_shoulders"] = {
          "status": "blocked",
          "reason": "experiment 2 ended before terminal stage A completed",
        }
        _write_state(state)

      if not running:
        break
      time.sleep(cfg.poll_interval_s)
  except KeyboardInterrupt:
    state["status"] = "interrupting"
    _write_state(state)
    _interrupt_running_steps(state, running, cfg.shutdown_timeout_s)
    state["status"] = "interrupted"
    state["finished_at_utc"] = _utc_now()
    _write_state(state)
    raise

  statuses = [
    record.get("status") for record in state.get("steps", {}).values()
  ]
  state["status"] = (
    "complete"
    if exp3_started and "failed" not in statuses and "blocked" not in statuses
    else "failed"
  )
  state["finished_at_utc"] = _utc_now()
  _write_state(state)


def run(cfg: SuiteCfg) -> None:
  state: dict[str, Any]
  state_path = REPO_ROOT / SUITE_STATE
  if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
  else:
    state = {
      "schema_version": 1,
      "created_at_utc": _utc_now(),
      "config": asdict(cfg),
      "steps": {},
    }
  state["config"] = asdict(cfg)
  state["updated_at_utc"] = _utc_now()
  _write_state(state)

  new_prior_sha = ""
  if cfg.run_prior:
    new_prior_sha = _ensure_static80_prior(state, cfg)
  elif (REPO_ROOT / NEW_PRIOR).is_file():
    new_prior_sha = _sha256(REPO_ROOT / NEW_PRIOR)
  if new_prior_sha and new_prior_sha != NEW_PRIOR_SHA256 and not cfg.dry_run:
    raise RuntimeError(
      f"Static80 prior SHA-256 mismatch: {new_prior_sha} != {NEW_PRIOR_SHA256}"
    )

  if not cfg.run_ppo:
    state["status"] = "prior_complete"
    _write_state(state)
    return
  if not new_prior_sha and not cfg.dry_run:
    raise RuntimeError("Static80 prior is required before PPO experiments")
  if cfg.parallel_ppo:
    _run_parallel_ppo(state, cfg, new_prior_sha)
    return

  exp1_state = "logs/reproduction/curriculum/gsi015_decay10_static50_chest_seed42"
  exp2_state = "logs/reproduction/curriculum/gsi015_decay10_static80_chest_seed42"
  exp3_state = "logs/reproduction/curriculum/gsi015_decay10_static80_shoulder_seed42"

  _run_step(
    state,
    "experiment_1_static50_chest",
    _curriculum_command(
      prior=OLD_PRIOR,
      prior_sha=OLD_PRIOR_SHA256,
      prefix="gsi015_decay10_static50_chest",
      state_dir=exp1_state,
    ),
    dry_run=cfg.dry_run,
  )
  _run_step(
    state,
    "experiment_2_static80_chest",
    _curriculum_command(
      prior=NEW_PRIOR,
      prior_sha=new_prior_sha,
      prefix="gsi015_decay10_static80_chest",
      state_dir=exp2_state,
    ),
    dry_run=cfg.dry_run,
  )
  _run_step(
    state,
    "experiment_3_shoulder_contact_smoke",
    [sys.executable, "scripts/smoke_force_unloading_shoulder.py"],
    dry_run=cfg.dry_run,
  )
  _run_step(
    state,
    "experiment_3_static80_shoulders",
    _curriculum_command(
      prior=NEW_PRIOR,
      prior_sha=new_prior_sha,
      prefix="gsi015_decay10_static80_shoulders",
      state_dir=exp3_state,
      extra=[
        "--target-probabilities",
        "0.0",
        "0.5",
        "0.5",
        "--push-direction-mode",
        "front_to_back",
        "--parent-stage-a-manifest",
        f"{exp2_state}/curriculum_state.json",
        "--shoulder-split-evaluation",
        "--no-run-smoke",
      ],
    ),
    dry_run=cfg.dry_run,
  )

  state["status"] = "complete"
  state["finished_at_utc"] = _utc_now()
  _write_state(state)


if __name__ == "__main__":
  import tyro

  run(tyro.cli(SuiteCfg))
