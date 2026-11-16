"""Strict GPU0 driver for the conditional shoulder-score V3 B1/B2 experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.train_force_unloading_curriculum import (
  CONTACT_STAGES,
  CurriculumCfg,
  ResumeSource,
  StageSpec,
  _find_new_run_dir,
  _run_process,
  build_evaluation_command,
  build_train_command,
  find_latest_checkpoint,
  remaining_iterations,
  validate_checkpoint,
)

from smp.pretrain.conditioning import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/conditional_contact_shoulder_score_v3"
PERSISTENT = Path(
  "/personal_fast_storage/smp_experiments/conditional_contact_shoulder_score_v3_20261112"
)
TASK = "Smp-Force-Unloading-Conditional-ShoulderScoreV3-G1"
LEGACY_TASK = "Smp-Force-Unloading-Conditional-G1"
EXPERIMENT = "smp_force_unloading_conditional_shoulder_score_v3"
GPU_UUID = "GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771"
PRIOR = ROOT / "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
PRIOR_SHA256 = "ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734"
PARENT = (
  ROOT / "logs/reproduction/conditional_contact_v2/rsl_rl/"
  "smp_force_unloading_conditional_g1/"
  "2026-11-11_02-34-11_conditional_contact_v2_a_seed42/model_2999.pt"
)
PARENT_SHA256 = "b2b6dc8f4fae3a93ce0028f3a594bceb09494590bb4b1d4f29ee24355006c3fb"
CALIBRATION = OUTPUT / "calibration/shoulder_score_calibration.pt"
CALIBRATION_REPORT = OUTPUT / "calibration/report.json"
MIGRATED = OUTPUT / f"rsl_rl/{EXPERIMENT}/parent_a_migrated/model_2999.pt"
BASELINE = (
  ROOT / "logs/reproduction/conditional_contact_v2/rsl_rl/"
  "smp_force_unloading_conditional_g1/"
  "2026-11-11_08-41-56_conditional_contact_v2_c_seed42/model_17000.pt"
)
MANIFEST = OUTPUT / "manifest.json"
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9]+")

B1 = CONTACT_STAGES[0]
B2 = CONTACT_STAGES[1]


class CalibrationBlocked(RuntimeError):
  pass


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  os.replace(temporary, path)


def require_storage() -> dict[str, Any]:
  if not OUTPUT.is_symlink() or OUTPUT.resolve() != PERSISTENT.resolve():
    raise RuntimeError("V3 output must resolve to its dedicated persistent directory")
  free = shutil.disk_usage(OUTPUT).free
  if free < 2 * 1024**3:
    raise RuntimeError("Less than 2 GiB free on V3 experiment storage")
  return {
    "logical_path": str(OUTPUT),
    "physical_path": str(OUTPUT.resolve()),
    "free_bytes": free,
  }


def require_gpu0_idle() -> dict[str, Any]:
  row = subprocess.check_output(
    [
      "nvidia-smi",
      "-i",
      "0",
      "--query-gpu=index,uuid,name",
      "--format=csv,noheader,nounits",
    ],
    text=True,
  ).strip()
  index, uuid, name = (part.strip() for part in row.split(",", maxsplit=2))
  if index != "0" or uuid != GPU_UUID:
    raise RuntimeError(f"GPU0 identity changed: {row}")
  active = subprocess.check_output(
    [
      "nvidia-smi",
      "-i",
      "0",
      "--query-compute-apps=pid,process_name",
      "--format=csv,noheader",
    ],
    text=True,
  ).strip()
  if active:
    raise RuntimeError(f"GPU0 already has compute processes: {active}")
  return {"physical_index": 0, "uuid": uuid, "name": name, "process_device": "cuda:0"}


def configure_environment() -> None:
  values = {
    "CUDA_VISIBLE_DEVICES": "0",
    "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "2",
    "SMP_PHYSICAL_GPU_UUID": GPU_UUID,
    "WANDB_DIR": str(OUTPUT),
    "WANDB_CACHE_DIR": str(OUTPUT / "cache/wandb"),
    "MPLCONFIGDIR": str(OUTPUT / "cache/matplotlib"),
    "WARP_CACHE_PATH": str(OUTPUT / "cache/warp"),
    "TORCHINDUCTOR_CACHE_DIR": str(OUTPUT / "cache/inductor"),
    "TRITON_CACHE_DIR": str(OUTPUT / "cache/triton"),
    "CUDA_CACHE_PATH": str(OUTPUT / "cache/cuda"),
    "XDG_CACHE_HOME": str(OUTPUT / "cache/xdg"),
    "TMPDIR": str(OUTPUT / "tmp"),
  }
  os.environ.update(values)
  for key in (
    "WANDB_CACHE_DIR",
    "MPLCONFIGDIR",
    "WARP_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "XDG_CACHE_HOME",
    "TMPDIR",
  ):
    Path(values[key]).mkdir(parents=True, exist_ok=True)


def curriculum_cfg() -> CurriculumCfg:
  return CurriculumCfg(
    task_id=TASK,
    prior_checkpoint=str(PRIOR),
    prior_sha256=PRIOR_SHA256,
    conditional_score_calibration=str(CALIBRATION),
    seed=42,
    gpu_id=0,
    num_envs=4096,
    smoke_num_envs=64,
    evaluation_num_envs=256,
    evaluation_num_episodes=1024,
    action_std=0.10,
    learning_rate=1.0e-4,
    save_interval=1000,
    gsi_max_root_planar_speed=0.15,
    stationary_speed_scale=10.0,
    sphere_radius=0.10,
    target_probabilities=(0.0, 0.5, 0.5, 0.0),
    push_direction_mode="front_to_back",
    experiment_name=EXPERIMENT,
    wandb_project="smp",
    run_prefix="conditional_shoulder_score_v3",
    log_root=str(OUTPUT / "rsl_rl"),
    state_dir=str(OUTPUT / "state"),
    evaluation_dir=str(OUTPUT / "evaluation"),
    run_smoke=False,
    record_final_video=True,
    final_video_length=600,
  )


def initial_manifest(storage: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "experiment": "EXP-026",
    "status": "running",
    "created_at_utc": now(),
    "updated_at_utc": now(),
    "storage": storage,
    "gpu": gpu,
    "identity": {
      "task": TASK,
      "prior": {"path": str(PRIOR), "sha256": PRIOR_SHA256},
      "parent_checkpoint": {"path": str(PARENT), "sha256": PARENT_SHA256},
      "score": {
        "timesteps": [8, 15, 22],
        "waist_yaw_feature_index": 21,
        "waist_yaw_feature_weight": 10.0,
        "contrast_temperature": 4.0,
        "transition": "three_frame_debounce_then_masked_current",
      },
    },
    "fixed_config": {
      **asdict(curriculum_cfg()),
      "stages": {"b1": asdict(B1), "b2": asdict(B2)},
      "stop_after_b2": True,
      "explicit_yaw_reward": False,
      "condition_one_hot": False,
    },
    "phases": {},
  }


def update_manifest(state: dict[str, Any]) -> None:
  state["updated_at_utc"] = now()
  atomic_json(MANIFEST, state)


def phase_record(state: dict[str, Any], name: str, **updates: Any) -> None:
  record = state["phases"].setdefault(name, {})
  record.update(updates)
  state["current_phase"] = name
  update_manifest(state)


def command_phase(
  state: dict[str, Any],
  name: str,
  command: list[str],
  *,
  require_gpu: bool = True,
  allow_retry: bool = False,
) -> None:
  record = state["phases"].get(name, {})
  if record.get("status") == "complete":
    return
  if record.get("status") in {"failed", "interrupted"} and not allow_retry:
    raise RuntimeError(
      f"Phase {name} previously failed; diagnosis approval is required"
    )
  if require_gpu:
    require_gpu0_idle()
  phase_record(
    state,
    name,
    status="running",
    started_at_utc=now(),
    command=command,
  )

  def on_start(pid: int) -> None:
    phase_record(state, name, pid=pid)

  try:
    wandb_url = _run_process(
      command,
      OUTPUT / "logs" / f"{name}.log",
      dry_run=False,
      on_start=on_start,
    )
  except Exception as error:
    phase_record(state, name, status="failed", finished_at_utc=now(), error=str(error))
    raise
  phase_record(
    state,
    name,
    status="complete",
    finished_at_utc=now(),
    wandb_url=wandb_url,
  )


def validate_fixed_inputs() -> None:
  if sha256(PRIOR) != PRIOR_SHA256:
    raise RuntimeError("Conditional v2 Prior SHA-256 mismatch")
  if sha256(PARENT) != PARENT_SHA256:
    raise RuntimeError("Parent A checkpoint SHA-256 mismatch")
  if not BASELINE.is_file():
    raise FileNotFoundError(BASELINE)


def ensure_calibration(state: dict[str, Any], allow_retry: bool) -> None:
  if not CALIBRATION_REPORT.is_file():
    command_phase(
      state,
      "calibration",
      [sys.executable, "-u", "-m", "scripts.calibrate_conditional_shoulder_score"],
      allow_retry=allow_retry,
    )
  report = json.loads(CALIBRATION_REPORT.read_text())
  phase_record(
    state,
    "calibration",
    status="complete" if report["gate"]["passed"] else "blocked",
    report=str(CALIBRATION_REPORT),
    calibration=str(CALIBRATION),
    calibration_sha256=report["calibration_sha256"],
    gate=report["gate"],
  )
  if not report["gate"]["passed"]:
    raise CalibrationBlocked(
      "Frozen score calibration failed; formal PPO is prohibited"
    )


def ensure_migration(state: dict[str, Any], allow_retry: bool) -> ResumeSource:
  command_phase(
    state,
    "checkpoint_migration",
    [sys.executable, "-u", "-m", "scripts.migrate_conditional_score_checkpoint"],
    require_gpu=False,
    allow_retry=allow_retry,
  )
  iteration = validate_checkpoint(MIGRATED)
  if iteration != 2999:
    raise RuntimeError("Migrated parent checkpoint iteration mismatch")
  phase_record(
    state,
    "checkpoint_migration",
    checkpoint=str(MIGRATED),
    checkpoint_sha256=sha256(MIGRATED),
    iteration=iteration,
    parent_sha256=PARENT_SHA256,
    actor_input_dim=128,
    critic_input_dim=992,
  )
  return ResumeSource(MIGRATED.parent, MIGRATED)


def training_phase(
  state: dict[str, Any],
  name: str,
  stage: StageSpec,
  parent: ResumeSource,
  *,
  allow_retry: bool,
  smoke: bool = False,
) -> ResumeSource:
  cfg = curriculum_cfg()
  experiment_root = Path(cfg.log_root) / cfg.experiment_name
  experiment_root.mkdir(parents=True, exist_ok=True)
  parent_iteration = validate_checkpoint(parent.checkpoint)
  requested = 2 if smoke else stage.iterations
  target_iteration = parent_iteration + requested - 1
  record = state["phases"].get(name, {})
  if record.get("status") == "complete":
    checkpoint = Path(record["checkpoint"])
    if validate_checkpoint(checkpoint) != target_iteration:
      raise RuntimeError(f"Completed phase {name} checkpoint no longer matches target")
    return ResumeSource(Path(record["run_dir"]), checkpoint)

  resume = parent
  active_stage = replace(stage, iterations=requested)
  recovery = None
  if record.get("status") in {"failed", "interrupted"}:
    if not allow_retry:
      raise RuntimeError(f"Phase {name} failed; use --resume-after-diagnosis")
    previous_run = record.get("run_dir")
    if previous_run:
      run_dir = Path(previous_run)
    else:
      run_name = record.get("run_name", "")
      candidates = list(experiment_root.glob(f"*_{run_name}"))
      if not candidates:
        raise RuntimeError(f"No partial run directory found for {name}")
      run_dir = max(candidates, key=lambda value: value.stat().st_mtime_ns)
    partial = find_latest_checkpoint(run_dir)
    partial_iteration = validate_checkpoint(partial)
    if not parent_iteration < partial_iteration < target_iteration:
      raise RuntimeError(f"Invalid recovery checkpoint iteration {partial_iteration}")
    resume = ResumeSource(run_dir, partial)
    active_stage = replace(
      stage,
      iterations=remaining_iterations(target_iteration, partial_iteration),
    )
    recovery = {
      "checkpoint": str(partial),
      "iteration": partial_iteration,
      "remaining_iterations": active_stage.iterations,
    }

  base_run_name = f"conditional_shoulder_score_v3_{name}_seed42"
  run_name = (
    base_run_name
    if recovery is None
    else f"{base_run_name}_resume_from_{recovery['iteration']}"
  )
  command = build_train_command(cfg, active_stage, run_name, resume, smoke=smoke)
  before = set(experiment_root.glob(f"*_{run_name}"))
  phase_record(
    state,
    name,
    status="running",
    run_name=run_name,
    requested_iterations=requested,
    target_iteration=target_iteration,
    parent_checkpoint=str(parent.checkpoint),
    actual_resume_checkpoint=str(resume.checkpoint),
    recovery=recovery,
    command=command,
  )
  require_gpu0_idle()

  def on_start(pid: int) -> None:
    phase_record(state, name, pid=pid)

  try:
    wandb_url = _run_process(
      command,
      OUTPUT / "logs" / f"{name}.log",
      dry_run=False,
      on_start=on_start,
    )
    run_dir = _find_new_run_dir(experiment_root, run_name, before)
    checkpoint = find_latest_checkpoint(run_dir)
    iteration = validate_checkpoint(checkpoint)
    if iteration != target_iteration:
      raise RuntimeError(
        f"{name} stopped at iteration {iteration}, expected {target_iteration}"
      )
  except Exception as error:
    run_dirs = list(experiment_root.glob(f"*_{run_name}"))
    phase_record(
      state,
      name,
      status="failed",
      finished_at_utc=now(),
      error=str(error),
      run_dir=str(max(run_dirs, key=lambda value: value.stat().st_mtime_ns))
      if run_dirs
      else None,
    )
    raise
  phase_record(
    state,
    name,
    status="complete",
    finished_at_utc=now(),
    run_dir=str(run_dir),
    checkpoint=str(checkpoint),
    checkpoint_sha256=sha256(checkpoint),
    iteration=iteration,
    wandb_url=wandb_url,
  )
  return ResumeSource(run_dir, checkpoint)


def evaluation_phase(
  state: dict[str, Any],
  name: str,
  stage: StageSpec,
  checkpoint: Path,
  target_probabilities: tuple[float, float, float, float],
  *,
  shoulder_gate: bool,
  video: bool = False,
  smoke: bool = False,
  legacy: bool = False,
  allow_retry: bool,
) -> dict[str, Any]:
  cfg = curriculum_cfg()
  if legacy:
    cfg = replace(
      cfg,
      task_id=LEGACY_TASK,
      conditional_score_calibration="",
      run_prefix="conditional_v2_model17000_baseline",
    )
  if smoke:
    cfg = replace(cfg, evaluation_num_envs=64, evaluation_num_episodes=64)
  output = OUTPUT / "evaluation" / f"{name}.json"
  video_folder = OUTPUT / "videos" / name if video else None
  command = build_evaluation_command(
    cfg,
    stage,
    checkpoint,
    output,
    video_folder=video_folder,
    target_probabilities=target_probabilities,
  )
  if shoulder_gate:
    command.append("--shoulder-gate")
  command_phase(state, name, command, allow_retry=allow_retry)
  report = json.loads(output.read_text())
  if not report.get("finite"):
    raise RuntimeError(f"Evaluation {name} contains non-finite values")
  if report["metrics"].get("failure_nan", {}).get("mean", 0.0) > 0.0:
    raise RuntimeError(f"Evaluation {name} observed NaN termination")
  if smoke:
    coverage = report["metrics"]["contact_coverage"]["mean"]
    classification = report["metrics"]["contact_condition_classification_correct"][
      "mean"
    ]
    if coverage < 0.80 or classification < 0.80:
      raise RuntimeError(
        f"Contact smoke failed: coverage={coverage}, classification={classification}"
      )
  phase_record(
    state,
    name,
    report=str(output),
    gate=report["gate"],
    warning=None if report["gate"]["passed"] else "quality_gate_failed",
    video_folder=str(video_folder) if video_folder else None,
  )
  return report


def aggregate_stage_gate(
  state: dict[str, Any], stage: str, reports: dict[str, dict]
) -> None:
  checks = {
    f"{condition}/{key}": passed
    for condition, report in reports.items()
    for key, passed in report["gate"]["checks"].items()
  }
  result = {"passed": all(checks.values()), "checks": checks}
  phase_record(
    state,
    f"{stage}_quality",
    status="complete",
    gate=result,
    warning=None if result["passed"] else "quality_gate_failed_but_continue",
  )


def compare_results(
  state: dict[str, Any],
  v3: dict[str, dict],
  baseline: dict[str, dict],
) -> None:
  metric_names = (
    "contact_coverage",
    "contact_condition_classification_correct",
    "response_direction_correct_given_contact",
    "signed_waist_response_peak_rad",
    "pelvis_yaw_abs_peak_rad",
    "event_success",
    "failure_emergency_force",
    "peak_force_weight_ratio",
    "max_contact_foot_slip_m_s",
  )
  comparison = {
    "created_at_utc": now(),
    "v3_checkpoint": state["phases"]["b2_train"]["checkpoint"],
    "baseline_checkpoint": str(BASELINE),
    "conditions": {},
  }
  for side in ("left", "right"):
    comparison["conditions"][side] = {
      "v3": {name: v3[side]["metrics"].get(name) for name in metric_names},
      "conditional_v2_model_17000": {
        name: baseline[side]["metrics"].get(name) for name in metric_names
      },
    }
  output = OUTPUT / "evaluation/final_comparison.json"
  atomic_json(output, comparison)
  phase_record(state, "final_comparison", status="complete", report=str(output))


def run(args: argparse.Namespace) -> None:
  os.chdir(ROOT)
  storage = require_storage()
  gpu = require_gpu0_idle()
  configure_environment()
  validate_fixed_inputs()
  state = (
    json.loads(MANIFEST.read_text())
    if MANIFEST.is_file()
    else initial_manifest(storage, gpu)
  )
  state.update(status="running", pid=os.getpid(), storage=storage, gpu=gpu)
  update_manifest(state)
  try:
    command_phase(
      state,
      "unit_tests",
      [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_conditional_shoulder_score_v3",
        "tests.test_conditional_contact",
      ],
      require_gpu=False,
      allow_retry=args.resume_after_diagnosis,
    )
    ensure_calibration(state, args.resume_after_diagnosis)
    migrated = ensure_migration(state, args.resume_after_diagnosis)
    training_phase(
      state,
      "ppo_smoke",
      B1,
      migrated,
      allow_retry=args.resume_after_diagnosis,
      smoke=True,
    )
    for side, target in {
      "left": (0.0, 1.0, 0.0, 0.0),
      "right": (0.0, 0.0, 1.0, 0.0),
    }.items():
      evaluation_phase(
        state,
        f"contact_smoke_{side}",
        B1,
        migrated.checkpoint,
        target,
        shoulder_gate=False,
        smoke=True,
        allow_retry=args.resume_after_diagnosis,
      )

    b1 = training_phase(
      state, "b1_train", B1, migrated, allow_retry=args.resume_after_diagnosis
    )
    stand_b1 = replace(
      B1,
      evaluation_stage="a",
      wait_range_s=(1.0e9, 1.0e9),
      stationary_reward_weight=0.9,
    )
    b1_reports = {
      "stand": evaluation_phase(
        state,
        "b1_eval_stand",
        stand_b1,
        b1.checkpoint,
        (1.0, 0.0, 0.0, 0.0),
        shoulder_gate=False,
        allow_retry=args.resume_after_diagnosis,
      ),
      "left": evaluation_phase(
        state,
        "b1_eval_left",
        B1,
        b1.checkpoint,
        (0.0, 1.0, 0.0, 0.0),
        shoulder_gate=True,
        allow_retry=args.resume_after_diagnosis,
      ),
      "right": evaluation_phase(
        state,
        "b1_eval_right",
        B1,
        b1.checkpoint,
        (0.0, 0.0, 1.0, 0.0),
        shoulder_gate=True,
        allow_retry=args.resume_after_diagnosis,
      ),
    }
    aggregate_stage_gate(state, "b1", b1_reports)

    b2 = training_phase(
      state, "b2_train", B2, b1, allow_retry=args.resume_after_diagnosis
    )
    stand_b2 = replace(
      B2,
      evaluation_stage="a",
      wait_range_s=(1.0e9, 1.0e9),
      stationary_reward_weight=0.9,
    )
    b2_reports = {
      "stand": evaluation_phase(
        state,
        "b2_eval_stand",
        stand_b2,
        b2.checkpoint,
        (1.0, 0.0, 0.0, 0.0),
        shoulder_gate=False,
        allow_retry=args.resume_after_diagnosis,
      ),
      "left": evaluation_phase(
        state,
        "b2_eval_left",
        B2,
        b2.checkpoint,
        (0.0, 1.0, 0.0, 0.0),
        shoulder_gate=True,
        video=True,
        allow_retry=args.resume_after_diagnosis,
      ),
      "right": evaluation_phase(
        state,
        "b2_eval_right",
        B2,
        b2.checkpoint,
        (0.0, 0.0, 1.0, 0.0),
        shoulder_gate=True,
        video=True,
        allow_retry=args.resume_after_diagnosis,
      ),
    }
    aggregate_stage_gate(state, "b2", b2_reports)

    baseline_reports = {
      side: evaluation_phase(
        state,
        f"baseline_model17000_{side}",
        B2,
        BASELINE,
        target,
        shoulder_gate=True,
        legacy=True,
        allow_retry=args.resume_after_diagnosis,
      )
      for side, target in {
        "left": (0.0, 1.0, 0.0, 0.0),
        "right": (0.0, 0.0, 1.0, 0.0),
      }.items()
    }
    compare_results(state, b2_reports, baseline_reports)
    state.update(status="complete", finished_at_utc=now(), current_phase=None)
  except CalibrationBlocked as error:
    state.update(status="blocked_calibration", blocked_at_utc=now(), error=str(error))
    raise
  except Exception as error:
    state.update(status="failed", failed_at_utc=now(), error=str(error))
    raise
  finally:
    update_manifest(state)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--resume-after-diagnosis", action="store_true")
  args = parser.parse_args()
  OUTPUT.mkdir(parents=True, exist_ok=True)
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run(args)


if __name__ == "__main__":
  main()
