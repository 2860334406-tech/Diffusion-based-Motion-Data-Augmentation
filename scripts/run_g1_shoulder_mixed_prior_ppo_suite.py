"""EXP-035 driver: mixed-source Prior gate followed by scratch A/B1/B2 shoulder PPO."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scripts.train_force_unloading_curriculum as curriculum
import torch
from scripts.curate_conditional_shoulder80 import atomic_json, now
from scripts.train_force_unloading_curriculum import (
  CONTACT_STAGES,
  STAGE_A,
  STAGE_A_EXTENSION,
  CurriculumCfg,
  ResumeSource,
  StageSpec,
  _assert_finite_evaluation,
  _run_process,
  _run_smoke,
  build_evaluation_command,
)

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_mixed import CONDITION_SCHEMA, SAMPLING
from smp.pretrain.shoulder_mixed_dataset import DATA_DIR, data_sha, prepare

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/g1_shoulder_mixed_prior_ppo_v1"
PERSISTENT_ROOT = Path("/personal_fast_storage/smp_experiments")
GPU_UUID = "GPU-eb0fc39f-6914-9e1d-65af-570524e8e061"
TASK = "Smp-Force-Unloading-Conditional-ShoulderMixedV1-G1"
EXPERIMENT = "smp_force_unloading_conditional_shoulder_mixed"
PUBLISHED_PRIOR = (
  ROOT / "datasets/pretrain_ckpt/pretrained_g1_shoulder_mixed_conditional_v1.pt"
)
CALIBRATION = OUTPUT / "calibration/final.pt"
SUITE_MANIFEST = OUTPUT / "manifest.json"
PPO_MANIFEST = OUTPUT / "ppo/manifest.json"
B1, B2 = CONTACT_STAGES[:2]


def _utc_stamp() -> str:
  return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def prepare_storage() -> dict[str, Any]:
  if not Path("/personal_fast_storage").is_mount():
    raise RuntimeError("Persistent storage is absent; root-disk fallback is prohibited")
  if OUTPUT.is_symlink():
    physical = OUTPUT.resolve()
  elif OUTPUT.exists():
    raise RuntimeError("Logical output exists but is not a persistent-storage symlink")
  else:
    physical = PERSISTENT_ROOT / f"g1_shoulder_mixed_prior_ppo_v1_{_utc_stamp()}"
    if physical.exists():
      raise RuntimeError(f"Refusing to reuse unexpected output: {physical}")
    physical.mkdir(parents=True)
    OUTPUT.symlink_to(physical, target_is_directory=True)
  if shutil.disk_usage(physical).free < 10 * 1024**3:
    raise RuntimeError("Less than 10 GiB free on persistent storage")
  probe = physical / ".write_probe"
  with probe.open("xb") as stream:
    stream.write(bytes(1024**2))
    stream.flush()
    os.fsync(stream.fileno())
  probe.unlink()
  return {
    "logical": str(OUTPUT),
    "physical": str(physical),
    "free_bytes": shutil.disk_usage(physical).free,
  }


def require_gpu0_idle() -> dict[str, Any]:
  row = subprocess.check_output(
    [
      "nvidia-smi",
      "-i",
      "0",
      "--query-gpu=index,uuid,name,memory.total",
      "--format=csv,noheader,nounits",
    ],
    text=True,
  ).strip()
  index, uuid, name, memory = (part.strip() for part in row.split(",", maxsplit=3))
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
  return {
    "physical_index": 0,
    "uuid": uuid,
    "name": name,
    "memory_mib": int(memory),
    "process_device": "cuda:0",
  }


def configure_environment() -> dict[str, str]:
  values = {
    "CUDA_VISIBLE_DEVICES": "0",
    "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "SMP_PHYSICAL_GPU_UUID": GPU_UUID,
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "WANDB_DIR": str(OUTPUT),
    "WANDB_CACHE_DIR": str(OUTPUT / "cache/wandb"),
    "XDG_CACHE_HOME": str(OUTPUT / "cache/xdg"),
    "TMPDIR": str(OUTPUT / "tmp"),
    "WARP_CACHE_PATH": str(OUTPUT / "cache/warp"),
    "MPLCONFIGDIR": str(OUTPUT / "cache/matplotlib"),
    "CUDA_CACHE_PATH": str(OUTPUT / "cache/cuda"),
    "TRITON_CACHE_DIR": str(OUTPUT / "cache/triton"),
    "TORCHINDUCTOR_CACHE_DIR": str(OUTPUT / "cache/inductor"),
  }
  os.environ.update(values)
  for key in (
    "WANDB_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TMPDIR",
    "WARP_CACHE_PATH",
    "MPLCONFIGDIR",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
  ):
    Path(values[key]).mkdir(parents=True, exist_ok=True)
  return values


def initial_manifest(storage, gpu):
  return {
    "schema_version": 1,
    "experiment": "EXP-035",
    "status": "starting",
    "created_at_utc": now(),
    "updated_at_utc": now(),
    "storage": storage,
    "gpu": gpu,
    "data": {
      "manifest": str(DATA_DIR / "manifest.json"),
      "manifest_sha256": data_sha(),
      "pure_ik_v2": False,
      "mixed_external_motion": True,
    },
    "prior": {
      "schema": CONDITION_SCHEMA,
      "sampling": SAMPLING,
      "budget_epochs": 10000,
      "initialization": "scratch",
      "published": False,
    },
    "ppo": {
      "task": TASK,
      "policy_initialization": "scratch",
      "stages": ["a", "a_extension_if_needed", "b1", "b2"],
      "stage_c": False,
    },
    "history": [],
  }


def update(state, **values):
  state.update(values, updated_at_utc=now())
  atomic_json(SUITE_MANIFEST, state)


def journal_outcome(state):
  """Append actual stage outcomes; never rewrite historical experiment entries."""
  prior_path = OUTPUT / "prior/run.json"
  prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
  ppo = json.loads(PPO_MANIFEST.read_text()) if PPO_MANIFEST.exists() else {}
  text = (
    f"\n- EXP-035 驱动状态更新（UTC {now()}）：status={state.get('status')}；"
    f"Prior实际{prior.get('completed_epochs', 0)}/10000 epochs，"
    f"W&B={prior.get('wandb_url')}。"
  )
  for name, stage in ppo.get("stages", {}).items():
    text += (
      f" {name}:status={stage.get('status')},"
      f"iteration={stage.get('checkpoint_iteration')},gate={stage.get('gate_passed')},"
      f"W&B={stage.get('wandb_url')}。"
    )
  if state.get("error"):
    text += f" 原因：{state['error']}。"
  with (ROOT / "docs/FORCE_UNLOADING_EXPERIMENT_LOG.md").open("a") as stream:
    stream.write(text + "\n")
  atomic_json(
    OUTPUT / "outcome.json",
    {
      "status": state.get("status"),
      "prior": prior,
      "ppo": ppo,
      "recorded_at_utc": now(),
    },
  )


def run_child(state, name: str, command: list[str], log: Path) -> None:
  require_gpu0_idle()
  attempt = {
    "stage": name,
    "command": command,
    "started_at_utc": now(),
    "status": "running",
  }
  state["history"].append(attempt)
  update(state, status=f"{name}_running")
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("a", buffering=1) as stream:
    child = subprocess.Popen(
      command,
      cwd=ROOT,
      env=os.environ.copy(),
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
    attempt["pid"] = child.pid
    update(state)

    def forward_signal(signum, _frame):
      if child.poll() is None:
        child.send_signal(signal.SIGTERM)
      attempt["interrupt_signal"] = signum
      update(state, status="interrupt_requested")

    old_term = signal.signal(signal.SIGTERM, forward_signal)
    old_int = signal.signal(signal.SIGINT, forward_signal)
    try:
      code = child.wait()
    finally:
      signal.signal(signal.SIGTERM, old_term)
      signal.signal(signal.SIGINT, old_int)
  attempt.update(
    status="complete" if code == 0 else "failed",
    returncode=code,
    finished_at_utc=now(),
  )
  update(state)
  if code:
    raise RuntimeError(f"{name} exited {code}; inspect {log}")


def run_prior(state, resume: bool) -> None:
  smoke_run = OUTPUT / "prior_smoke/run.json"
  smoke_status = (
    json.loads(smoke_run.read_text()).get("status") if smoke_run.exists() else None
  )
  if smoke_status != "smoke_passed":
    command = [
      sys.executable,
      "-u",
      "-m",
      "scripts.pretrain_shoulder_mixed",
      "--output",
      str(OUTPUT / "prior_smoke"),
      "--smoke",
    ]
    if resume and (OUTPUT / "prior_smoke/latest.pt").exists():
      command.append("--resume")
    run_child(state, "prior_smoke", command, OUTPUT / "logs/prior_smoke.log")
  smoke = json.loads((OUTPUT / "prior_smoke/smoke_report.json").read_text())
  if not all(smoke[key] for key in ("finite_loss", "finite_score", "finite_ddpm")):
    raise RuntimeError("Prior smoke produced a nonfinite result")
  state["prior"]["smoke"] = smoke
  update(state)

  prior_run_path = OUTPUT / "prior/run.json"
  prior_status = (
    json.loads(prior_run_path.read_text()).get("status")
    if prior_run_path.exists()
    else None
  )
  terminal = {"candidate_locked_pending_test", "failed_no_score_passing_candidate"}
  if prior_status not in terminal:
    command = [
      sys.executable,
      "-u",
      "-m",
      "scripts.pretrain_shoulder_mixed",
      "--output",
      str(OUTPUT / "prior"),
    ]
    if resume and (OUTPUT / "prior/latest.pt").exists():
      command.append("--resume")
    run_child(state, "prior_train", command, OUTPUT / "logs/prior_train.log")
  training = json.loads(prior_run_path.read_text())
  state["prior"].update(
    training_status=training["status"],
    completed_epochs=training["completed_epochs"],
    selected=training.get("selected"),
    wandb_url=training.get("wandb_url"),
  )
  update(state)
  selected = training.get("selected")
  if not selected:
    update(state, status="blocked_prior_validation", ppo_authorized=False)
    raise RuntimeError("No 1000-epoch candidate passed the frozen validation gate")

  test_report_path = OUTPUT / "prior/test/report.json"
  resume_online = False
  if test_report_path.exists():
    previous_test = json.loads(test_report_path.read_text())
    if not previous_test.get("gate", {}).get("passed") or not previous_test.get(
      "ppo_authorized"
    ):
      offline_path = OUTPUT / "prior/test/offline_report.json"
      online_path = OUTPUT / "prior/test/online/report.json"
      resume_online = (
        resume
        and offline_path.exists()
        and not online_path.exists()
        and json.loads(offline_path.read_text())["gate"]["passed"]
      )
      if not resume_online:
        update(state, status="blocked_prior_test_or_online", ppo_authorized=False)
        raise RuntimeError(
          "Frozen test/online gate previously failed; automatic repetition is prohibited"
        )
  if not test_report_path.exists() or resume_online:
    command = [
      sys.executable,
      "-u",
      "-m",
      "scripts.validate_shoulder_mixed_prior",
      "--checkpoint",
      selected["checkpoint"],
      "--calibration",
      selected["calibration"],
      "--output",
      str(OUTPUT / "prior/test"),
      "--publish",
      str(PUBLISHED_PRIOR),
    ]
    if resume_online:
      command.append("--resume-online")
    try:
      run_child(state, "prior_test", command, OUTPUT / "logs/prior_test.log")
    except RuntimeError:
      if test_report_path.exists():
        failed = json.loads(test_report_path.read_text())
        if not failed.get("gate", {}).get("passed", False):
          state["prior"]["test_report"] = str(test_report_path)
          update(state, status="blocked_prior_test", ppo_authorized=False)
      raise
  report = json.loads(test_report_path.read_text())
  if not report["gate"]["passed"] or not report.get("ppo_authorized"):
    update(state, status="blocked_prior_test", ppo_authorized=False)
    raise RuntimeError("Blind Prior test failed; PPO is prohibited")
  state["prior"].update(
    published=True,
    published_path=str(PUBLISHED_PRIOR),
    published_sha256=sha256(PUBLISHED_PRIOR),
    calibration=str(CALIBRATION),
    calibration_sha256=sha256(CALIBRATION),
    test_report=str(test_report_path),
  )
  update(state, status="prior_accepted", ppo_authorized=True)


def curriculum_cfg() -> CurriculumCfg:
  return CurriculumCfg(
    task_id=TASK,
    prior_checkpoint=str(PUBLISHED_PRIOR),
    prior_sha256=sha256(PUBLISHED_PRIOR),
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
    contact_reference_speeds=(0.01, 0.03, 0.05),
    target_probabilities=(0.0, 0.5, 0.5, 0.0),
    push_direction_mode="front_to_back",
    experiment_name=EXPERIMENT,
    wandb_project="smp",
    run_prefix="g1_shoulder_mixed",
    log_root=str(OUTPUT / "ppo/rsl_rl"),
    state_dir=str(OUTPUT / "ppo/state"),
    evaluation_dir=str(OUTPUT / "ppo/evaluation"),
    run_smoke=True,
    shoulder_split_evaluation=True,
    record_final_video=False,
  )


def install_shoulder_gate_command_wrapper() -> None:
  original = curriculum.build_evaluation_command

  def wrapped(cfg, stage, checkpoint, output, **kwargs):
    probabilities = kwargs.get("target_probabilities")
    if probabilities is not None and len(probabilities) == 3:
      kwargs["target_probabilities"] = (*probabilities, 0.0)
    command = original(cfg, stage, checkpoint, output, **kwargs)
    command[1] = "scripts/evaluate_shoulder_mixed.py"
    probabilities = kwargs.get("target_probabilities")
    shoulder = (
      probabilities is not None
      and len(probabilities) >= 3
      and (float(probabilities[1]) + float(probabilities[2]) > 0.999)
    )
    if shoulder and stage.key in {"b1", "b2", "c"}:
      command.append("--shoulder-gate")
    return command

  curriculum.build_evaluation_command = wrapped
  globals()["build_evaluation_command"] = wrapped


def contact_smoke(state, cfg, parent: ResumeSource, side: str) -> None:
  name = f"contact_smoke_{side}"
  ppo = state.setdefault("ppo_runtime", {"stages": {}})
  if ppo["stages"].get(name, {}).get("status") == "complete":
    return
  target = (0.0, 1.0, 0.0, 0.0) if side == "left" else (0.0, 0.0, 1.0, 0.0)
  small = replace(cfg, evaluation_num_envs=64, evaluation_num_episodes=128)
  output = OUTPUT / f"ppo/evaluation/{name}.json"
  command = curriculum.build_evaluation_command(
    small, B1, parent.checkpoint, output, target_probabilities=target
  )
  ppo["stages"][name] = {
    "status": "running",
    "command": command,
    "started_at_utc": now(),
  }
  update(state, status=name)
  _run_process(command, OUTPUT / f"ppo/state/{name}.log", dry_run=False)
  report = json.loads(output.read_text())
  coverage = float(report["metrics"]["contact_coverage"]["mean"])
  classified = float(
    report["metrics"]["contact_condition_classification_correct"]["mean"]
  )
  if not report.get("finite") or coverage < 0.80 or classified < 0.80:
    raise RuntimeError(
      f"{name} failed geometry gate: finite={report.get('finite')}, "
      f"coverage={coverage}, classification={classified}"
    )
  ppo["stages"][name].update(
    status="complete",
    finished_at_utc=now(),
    report=str(output),
    contact_coverage=coverage,
    classification_correct=classified,
  )
  update(state)


def evaluate_no_contact(state, cfg, stage, parent: ResumeSource) -> None:
  name = f"{stage.key}_stand"
  ppo = state.setdefault("ppo_runtime", {"stages": {}})
  previous = ppo["stages"].get(name, {})
  if (
    previous.get("status") == "complete"
    and previous.get("wandb_sync", {}).get("status") == "synced"
  ):
    return
  stand_stage = replace(
    stage,
    evaluation_stage="a",
    wait_range_s=(1.0e9, 1.0e9),
    stationary_reward_weight=0.9,
  )
  output = OUTPUT / f"ppo/evaluation/{name}.json"
  command = build_evaluation_command(
    cfg,
    stand_stage,
    parent.checkpoint,
    output,
    target_probabilities=(1.0, 0.0, 0.0, 0.0),
  )
  if previous.get("status") != "complete":
    _run_process(command, OUTPUT / f"ppo/state/{name}.log", dry_run=False)
  report = json.loads(output.read_text())
  _assert_finite_evaluation(report, name)
  ppo["stages"][name] = {
    "status": "complete",
    "report": str(output),
    "gate_passed": bool(report["gate"]["passed"]),
  }
  if not report["gate"]["passed"]:
    warning = (
      f"EXP-035 {name}: no-contact quality gate failed; continuing approved course"
    )
    ppo["stages"][name]["warning"] = warning
    print(warning, flush=True)
  training = json.loads(PPO_MANIFEST.read_text())["stages"][stage.key]
  ppo["stages"][name]["wandb_sync"] = sync_no_contact(training.get("wandb_url"), report)
  update(state)


def sync_no_contact(url, report):
  if not url:
    return {"status": "pending", "reason": "training URL unavailable"}
  try:
    import wandb

    run_path = url.split("wandb.ai/", 1)[1].replace("/runs/", "/")
    run = wandb.Api(timeout=30).run(run_path)
    values = {
      "evaluation/no_contact/gate_passed": report["gate"]["passed"],
      "evaluation/no_contact/gate_checks": report["gate"]["checks"],
    }
    for metric, stats in report["metrics"].items():
      for statistic, value in stats.items():
        values[f"evaluation/no_contact/{metric}/{statistic}"] = value
    run.summary.update(values)
    return {"status": "synced", "url": url}
  except Exception as error:
    return {"status": "pending", "url": url, "error_type": type(error).__name__}


def record_final_replays(state, cfg, checkpoint: Path) -> None:
  ppo = state.setdefault("ppo_runtime", {"stages": {}})
  for name, probabilities, stage in (
    (
      "stand",
      (1.0, 0.0, 0.0, 0.0),
      replace(
        B2,
        evaluation_stage="a",
        wait_range_s=(1.0e9, 1.0e9),
        stationary_reward_weight=0.9,
      ),
    ),
    ("left", (0.0, 1.0, 0.0, 0.0), B2),
    ("right", (0.0, 0.0, 1.0, 0.0), B2),
  ):
    key = f"replay_{name}"
    if ppo["stages"].get(key, {}).get("status") == "complete":
      continue
    small = replace(
      cfg,
      evaluation_num_envs=4,
      evaluation_num_episodes=4,
      final_video_length=600,
    )
    report = OUTPUT / f"ppo/evaluation/{key}.json"
    video = OUTPUT / f"ppo/replay/{name}"
    command = curriculum.build_evaluation_command(
      small,
      stage,
      checkpoint,
      report,
      target_probabilities=probabilities,
      video_folder=video,
    )
    _run_process(command, OUTPUT / f"ppo/state/{key}.log", dry_run=False)
    result = json.loads(report.read_text())
    _assert_finite_evaluation(result, key)
    ppo["stages"][key] = {
      "status": "complete",
      "report": str(report),
      "video_folder": str(video),
    }
    update(state)


def expected_terminal_iteration(stage, resume_iteration):
  return (
    (0 if resume_iteration is None else resume_iteration + 1) + stage.iterations - 1
  )


def remaining_iterations(target_iteration, checkpoint_iteration):
  if checkpoint_iteration >= target_iteration:
    raise ValueError(
      "Training already reached its target; evaluation-only recovery required"
    )
  return target_iteration - checkpoint_iteration


def sync_stage_evaluation(cfg, url, report, warning):
  translated = {**report, "conditions": report.get("sides", {"stand": report})}
  return curriculum._sync_conditional_evaluation(
    replace(cfg, all_contact_evaluation=True), url, translated, warning
  )


def _run_training_stage(
  cfg: CurriculumCfg,
  manifest: dict[str, Any],
  manifest_path: Path,
  state_dir: Path,
  evaluation_dir: Path,
  experiment_root: Path,
  stage: StageSpec,
  resume: ResumeSource | None,
) -> tuple[ResumeSource, dict[str, Any]]:
  existing = curriculum._completed_source(manifest, stage.key)
  if existing is not None:
    record = manifest["stages"][stage.key]
    if (record.get("wandb_evaluation_sync") or {}).get("status") != "synced":
      sync = sync_stage_evaluation(
        cfg, record.get("wandb_url"), existing[1], record.get("warning")
      )
      curriculum._record_stage(
        manifest, manifest_path, stage.key, {"wandb_evaluation_sync": sync}
      )
    return existing

  previous_record = dict(manifest["stages"].get(stage.key, {}))
  if previous_record.get("status") in {"running", "evaluating"} and manifest.get(
    "resume_requested"
  ):
    pid = previous_record.get("pid")
    alive = False
    if pid:
      try:
        os.kill(int(pid), 0)
        alive = True
      except ProcessLookupError:
        pass
    if alive:
      raise RuntimeError(
        f"Stage {stage.key} PID {pid} is still alive; refusing a duplicate"
      )
    previous_record.update(
      status="failed",
      verified_recoverable_interruption=True,
      error="Verified dead stage process after driver interruption",
      finished_at_utc=now(),
    )
    curriculum._record_stage(manifest, manifest_path, stage.key, previous_record)
  if previous_record:
    status = previous_record.get("status")
    if status in {"running", "evaluating"}:
      raise RuntimeError(
        "Conditional stage is already running or was interrupted without a clean exit; "
        "inspect its PID/log and checkpoint before marking a verified interruption. "
        "Refusing to silently repeat the training budget."
      )
    if status == "failed" and not curriculum.is_recoverable_interruption(
      previous_record
    ):
      raise RuntimeError(
        "Previous conditional stage hard-failed. Diagnose before resuming its "
        "validated checkpoint; automatic retraining is prohibited."
      )
  original_resume = resume
  resume_iteration = (
    curriculum.validate_checkpoint(resume.checkpoint) if resume else None
  )
  target_iteration = expected_terminal_iteration(stage, resume_iteration)
  training_stage = stage
  recovery: dict[str, Any] | None = None
  if curriculum.is_recoverable_interruption(previous_record):
    partial = curriculum._find_latest_stage_checkpoint(cfg, stage, experiment_root)
    if partial is None:
      raise RuntimeError(
        f"No valid partial checkpoint found for interrupted stage {stage.key}"
      )
    partial_source, partial_iteration = partial
    if partial_iteration == target_iteration:
      evaluation_output, report, video = curriculum._evaluate_stage(
        cfg, stage, partial_source.checkpoint, state_dir, evaluation_dir
      )
      warning = (
        None
        if report["gate"]["passed"]
        else f"Quality gate failed for {stage.key}; continuing"
      )
      sync = sync_stage_evaluation(
        cfg, previous_record.get("wandb_url"), report, warning
      )
      curriculum._record_stage(
        manifest,
        manifest_path,
        stage.key,
        {
          "status": "complete",
          "evaluation_only_recovery": True,
          "run_dir": str(partial_source.run_dir),
          "checkpoint": str(partial_source.checkpoint),
          "checkpoint_iteration": partial_iteration,
          "evaluation": str(evaluation_output),
          "gate_passed": bool(report["gate"]["passed"]),
          "gate_checks": report["gate"]["checks"],
          "warning": warning,
          "wandb_evaluation_sync": sync,
          "video_folder": video,
          "finished_at_utc": now(),
        },
      )
      return partial_source, report
    if partial_iteration > target_iteration:
      raise RuntimeError(
        f"Interrupted stage {stage.key} already reached target iteration "
        f"{target_iteration}; evaluation-only recovery is required"
      )
    if resume_iteration is not None and partial_iteration <= resume_iteration:
      raise RuntimeError(
        f"Partial checkpoint {partial_source.checkpoint} made no progress beyond "
        f"parent iteration {resume_iteration}"
      )
    training_stage = replace(
      stage,
      iterations=remaining_iterations(target_iteration, partial_iteration),
    )
    resume = partial_source
    recovery = {
      "reason": previous_record.get("error"),
      "interrupted_at_utc": previous_record.get("finished_at_utc"),
      "partial_run_dir": str(partial_source.run_dir.resolve()),
      "partial_checkpoint": str(partial_source.checkpoint.resolve()),
      "partial_checkpoint_iteration": partial_iteration,
      "remaining_iterations": training_stage.iterations,
      "target_checkpoint_iteration": target_iteration,
    }

  if recovery is None:
    run_name = f"{cfg.run_prefix}_{stage.key}_seed{cfg.seed}"
  else:
    run_name = (
      f"{cfg.run_prefix}_{stage.key}_resume_from_"
      f"{recovery['partial_checkpoint_iteration']}_seed{cfg.seed}"
    )
  command = curriculum.build_train_command(cfg, training_stage, run_name, resume)
  before = set(experiment_root.glob(f"*_{run_name}"))
  curriculum._record_stage(
    manifest,
    manifest_path,
    stage.key,
    {
      "status": "running",
      "label": stage.label,
      "started_at_utc": now(),
      "iterations_requested": stage.iterations,
      "iterations_this_run": training_stage.iterations,
      "target_checkpoint_iteration": target_iteration,
      "command": command,
      "resumed_from": str(resume.checkpoint.resolve()) if resume else None,
      "original_parent_checkpoint": (
        str(original_resume.checkpoint.resolve()) if original_resume else None
      ),
      "scratch": original_resume is None,
      "recovery": recovery,
    },
  )

  def on_start(pid: int) -> None:
    curriculum._record_stage(manifest, manifest_path, stage.key, {"pid": pid})

  try:
    wandb_url = _run_process(
      command,
      state_dir / f"{stage.key}_train.log",
      dry_run=cfg.dry_run,
      on_start=on_start,
    )
    if cfg.dry_run:
      curriculum._record_stage(
        manifest, manifest_path, stage.key, {"status": "dry_run"}
      )
      raise RuntimeError("Dry-run mode does not produce curriculum checkpoints")

    run_dir = curriculum._find_new_run_dir(experiment_root, run_name, before)
    checkpoint = curriculum.find_latest_checkpoint(run_dir)
    iteration = curriculum.validate_checkpoint(checkpoint)
    if iteration != target_iteration:
      raise RuntimeError(
        f"Stage {stage.key} ended at checkpoint iteration {iteration}, "
        f"expected {target_iteration}"
      )
    curriculum._assert_finite_training_log(run_dir)

    evaluation_output, report, video_record = curriculum._evaluate_stage(
      cfg,
      stage,
      checkpoint,
      state_dir,
      evaluation_dir,
    )

    gate_passed = bool(report["gate"]["passed"])
    warning = (
      None
      if gate_passed
      else (f"Quality gate failed for {stage.key}; continuing by configured policy")
    )
    evaluation_sync = sync_stage_evaluation(cfg, wandb_url, report, warning)
    curriculum._record_stage(
      manifest,
      manifest_path,
      stage.key,
      {
        "status": "complete",
        "wandb_evaluation_sync": evaluation_sync,
        "finished_at_utc": now(),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_iteration": iteration,
        "evaluation": str(evaluation_output.resolve()),
        "gate_passed": gate_passed,
        "gate_checks": report["gate"]["checks"],
        "warning": warning,
        "wandb_url": wandb_url,
        "video_folder": video_record,
      },
    )
    if warning:
      curriculum._safe_print(f"[CURRICULUM] WARNING: {warning}", flush=True)
    return ResumeSource(run_dir, checkpoint), report
  except Exception as exc:
    curriculum._record_stage(
      manifest,
      manifest_path,
      stage.key,
      {"status": "failed", "error": str(exc), "finished_at_utc": now()},
    )
    raise


def run_ppo(state) -> None:
  install_shoulder_gate_command_wrapper()
  process = curriculum._run_process

  def checked_process(*args, **kwargs):
    require_gpu0_idle()
    return process(*args, **kwargs)

  curriculum._run_process = checked_process
  globals()["_run_process"] = checked_process
  cfg = curriculum_cfg()
  state_dir = Path(cfg.state_dir)
  evaluation_dir = Path(cfg.evaluation_dir)
  experiment_root = Path(cfg.log_root) / cfg.experiment_name
  for path in (state_dir, evaluation_dir, experiment_root):
    path.mkdir(parents=True, exist_ok=True)
  if PPO_MANIFEST.exists():
    manifest = json.loads(PPO_MANIFEST.read_text())
    if manifest["prior"]["sha256"] != cfg.prior_sha256:
      raise RuntimeError("PPO manifest Prior identity mismatch")
  else:
    manifest = {
      "schema_version": 1,
      "status": "running",
      "created_at_utc": now(),
      "updated_at_utc": now(),
      "task_id": TASK,
      "prior": {"path": str(PUBLISHED_PRIOR), "sha256": cfg.prior_sha256},
      "calibration": {"path": str(CALIBRATION), "sha256": sha256(CALIBRATION)},
      "config": asdict(cfg),
      "scratch_policy": {
        "actor": True,
        "critic": True,
        "observation_normalizers": True,
        "optimizer": True,
        "old_a_checkpoint_loaded": False,
      },
      "stages": {},
    }
    atomic_json(PPO_MANIFEST, manifest)
  update(state, status="ppo_running", ppo_manifest=str(PPO_MANIFEST))
  manifest["resume_requested"] = bool(state.get("resume_requested"))
  atomic_json(PPO_MANIFEST, manifest)

  _run_smoke(cfg, manifest, PPO_MANIFEST, state_dir, experiment_root)
  smoke_checkpoint = Path(manifest["stages"]["smoke"]["checkpoint"])
  smoke_state = torch.load(smoke_checkpoint, map_location="cpu", weights_only=True)
  actor_input = int(smoke_state["actor_state_dict"]["mlp.0.weight"].shape[1])
  critic_input = int(smoke_state["critic_state_dict"]["mlp.0.weight"].shape[1])
  if (actor_input, critic_input) != (128, 992):
    raise RuntimeError(
      f"PPO observation contract changed: actor={actor_input}, critic={critic_input}"
    )
  manifest["stages"]["smoke"]["observation_input_dims"] = {
    "actor": actor_input,
    "critic": critic_input,
  }
  atomic_json(PPO_MANIFEST, manifest)
  source, report_a = _run_training_stage(
    cfg,
    manifest,
    PPO_MANIFEST,
    state_dir,
    evaluation_dir,
    experiment_root,
    STAGE_A,
    None,
  )
  if not report_a["gate"]["passed"]:
    source, report_a = _run_training_stage(
      cfg,
      manifest,
      PPO_MANIFEST,
      state_dir,
      evaluation_dir,
      experiment_root,
      STAGE_A_EXTENSION,
      source,
    )
    if not report_a["gate"]["passed"]:
      manifest.update(status="blocked_stage_a", finished_at_utc=now())
      atomic_json(PPO_MANIFEST, manifest)
      update(state, status="blocked_stage_a")
      raise RuntimeError(
        "Stage A still failed after the approved 2000-iteration extension"
      )

  contact_smoke(state, cfg, source, "left")
  contact_smoke(state, cfg, source, "right")
  source, report_b1 = _run_training_stage(
    cfg,
    manifest,
    PPO_MANIFEST,
    state_dir,
    evaluation_dir,
    experiment_root,
    B1,
    source,
  )
  evaluate_no_contact(state, cfg, B1, source)
  source, report_b2 = _run_training_stage(
    cfg,
    manifest,
    PPO_MANIFEST,
    state_dir,
    evaluation_dir,
    experiment_root,
    B2,
    source,
  )
  evaluate_no_contact(state, cfg, B2, source)
  record_final_replays(state, cfg, source.checkpoint)
  pending = any(
    (r.get("wandb_evaluation_sync") or {}).get("status") == "pending"
    for r in manifest["stages"].values()
  )
  pending |= any(
    (r.get("wandb_sync") or {}).get("status") == "pending"
    for r in state.get("ppo_runtime", {}).get("stages", {}).values()
  )
  final_status = "complete_local_wandb_pending" if pending else "complete"
  manifest.update(
    status=final_status,
    finished_at_utc=now(),
    final_checkpoint=str(source.checkpoint.resolve()),
    final_checkpoint_sha256=sha256(source.checkpoint),
    b1_gate_passed=bool(report_b1["gate"]["passed"]),
    b2_gate_passed=bool(report_b2["gate"]["passed"]),
  )
  atomic_json(PPO_MANIFEST, manifest)
  update(
    state,
    status=final_status,
    finished_at_utc=now(),
    final_checkpoint=str(source.checkpoint.resolve()),
    final_checkpoint_sha256=sha256(source.checkpoint),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--prior-only", action="store_true")
  parser.add_argument("--prepare-only", action="store_true")
  args = parser.parse_args()
  os.chdir(ROOT)
  storage = prepare_storage()
  gpu = require_gpu0_idle()
  configure_environment()
  prepare()
  state = (
    json.loads(SUITE_MANIFEST.read_text())
    if SUITE_MANIFEST.exists()
    else initial_manifest(storage, gpu)
  )
  if state["data"]["manifest_sha256"] != data_sha():
    raise RuntimeError("Suite resume data identity mismatch")
  state.update(
    storage=storage, gpu=gpu, driver_pid=os.getpid(), resume_requested=args.resume
  )
  sources = [
    Path(__file__).resolve().relative_to(ROOT),
    Path("scripts/pretrain_shoulder_mixed.py"),
    Path("scripts/validate_shoulder_mixed_prior.py"),
    Path("scripts/probe_shoulder_mixed_online.py"),
    Path("scripts/evaluate_shoulder_mixed.py"),
    Path("scripts/train_force_unloading_curriculum.py"),
    Path("src/smp/pretrain/shoulder_mixed.py"),
    Path("src/smp/pretrain/shoulder_mixed_dataset.py"),
    Path("src/smp/pretrain/shoulder_mixed_score.py"),
    Path("src/smp/pretrain/model.py"),
    Path("src/smp/pretrain/schema.py"),
    Path("src/smp/pretrain/scheduler.py"),
    Path("src/smp/rl/rewards.py"),
    Path("src/smp/rl/events.py"),
    Path("src/smp/rl/conditional_score.py"),
    Path("src/smp/rl/tasks/force_unloading/conditional_shoulder_mixed.py"),
  ]
  code_identity = {str(p): sha256(p) for p in sources}
  if state.get("code_identity", code_identity) != code_identity:
    raise RuntimeError("Frozen experiment code changed; diagnose before resume")
  state["code_identity"] = code_identity
  for path in sources:
    destination = OUTPUT / "audit/code" / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
      shutil.copyfile(path, destination)
  update(state)
  if args.prepare_only:
    update(state, status="prepared")
    print(json.dumps({"storage": storage, "gpu": gpu}, indent=2))
    return
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
      run_prior(state, args.resume)
      if args.prior_only:
        update(state, status="prior_complete")
        return
      run_ppo(state)
    except Exception as error:
      if not str(state.get("status", "")).startswith("blocked"):
        update(
          state,
          status="failed",
          failed_at_utc=now(),
          error_type=type(error).__name__,
          error=str(error),
        )
      raise
    finally:
      journal_outcome(state)


if __name__ == "__main__":
  main()
