"""Run the user-authorized Shoulder80 failed-Prior PPO B1/B2 experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.prepare_shoulder80_ppo_override import Cfg as PreparationCfg
from scripts.prepare_shoulder80_ppo_override import prepare
from scripts.train_force_unloading_curriculum import (
  CONTACT_STAGES,
  CurriculumCfg,
  ResumeSource,
  _assert_finite_training_log,
  _find_new_run_dir,
  _run_process,
  _run_training_stage,
  build_evaluation_command,
  build_train_command,
  find_latest_checkpoint,
  validate_checkpoint,
)

from smp.pretrain.conditioning import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/conditional_shoulder80_ppo_v1_override"
PERSISTENT = Path(
  "/personal_fast_storage/smp_experiments/conditional_shoulder80_ppo_v1_override_20261112"
)
TASK = "Smp-Force-Unloading-Conditional-Shoulder80V1-G1"
EXPERIMENT = "smp_force_unloading_conditional_shoulder80_v1"
GPU_UUID = "GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771"
PRIOR = (
  ROOT / "logs/reproduction/conditional_shoulder80_prior_v1_expanded/"
  "prior/checkpoint_02000.pt"
)
PRIOR_SHA256 = "73d9b0409066650525c3bb619f8440af947c2a96f3697f8774ed23d859c90cbe"
GSI_PRIOR = ROOT / "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
GSI_PRIOR_SHA256 = "ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734"
CALIBRATION = OUTPUT / "calibration/runtime_calibration.pt"
MIGRATED = (
  OUTPUT / "rsl_rl/smp_force_unloading_conditional_shoulder80_v1/"
  "parent_a_migrated_gsi_v2/model_2999.pt"
)
MANIFEST = OUTPUT / "manifest.json"
B1, B2 = CONTACT_STAGES[:2]


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  os.replace(temporary, path)


def require_storage() -> dict[str, Any]:
  if not OUTPUT.is_symlink() or OUTPUT.resolve() != PERSISTENT.resolve():
    raise RuntimeError("PPO output must resolve to its dedicated persistent directory")
  free = shutil.disk_usage(OUTPUT).free
  if free < 2 * 1024**3:
    raise RuntimeError("Less than 2 GiB free on persistent experiment storage")
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
  for key, value in values.items():
    if key.endswith("DIR") or key.endswith("PATH"):
      Path(value).mkdir(parents=True, exist_ok=True)


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
    run_prefix="conditional_shoulder80_override",
    log_root=str(OUTPUT / "rsl_rl"),
    state_dir=str(OUTPUT / "state"),
    evaluation_dir=str(OUTPUT / "evaluation"),
    run_smoke=False,
    shoulder_split_evaluation=True,
    record_final_video=False,
  )


def initial_manifest(storage: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
  cfg = curriculum_cfg()
  return {
    "schema_version": 1,
    "experiment": "EXP-030",
    "status": "starting",
    "created_at_utc": now(),
    "updated_at_utc": now(),
    "storage": storage,
    "gpu": gpu,
    "task_id": TASK,
    "prior": {
      "path": str(PRIOR),
      "sha256": PRIOR_SHA256,
      "formal_gate_passed": False,
      "selected_by_original_protocol": False,
      "candidate_epoch": 2000,
    },
    "gsi_prior": {
      "path": str(GSI_PRIOR),
      "sha256": GSI_PRIOR_SHA256,
      "role": "stand-only initialization; not used for SMP reward",
    },
    "authorization": {
      "user_authorized_failed_prior_override": True,
      "authorized_at_utc": now(),
      "claim_prior_accepted": False,
      "claim_causal_ablation": False,
    },
    "score": {
      "mode": "shoulder_mirror_contrast_v1",
      "score_v4_used": False,
      "calibration": str(CALIBRATION),
      "known_limitation": "formal validation correct>stand gate failed",
    },
    "config": asdict(cfg),
    "stages": {},
  }


def update(state: dict[str, Any], **values: Any) -> None:
  state.update(values, updated_at_utc=now())
  atomic_json(MANIFEST, state)


def smoke_train(
  state: dict[str, Any], cfg: CurriculumCfg, parent: ResumeSource
) -> None:
  stage_name = "ppo_smoke_dual_gsi"
  record = state["stages"].get(stage_name, {})
  if record.get("status") == "complete":
    return
  experiment_root = Path(cfg.log_root) / cfg.experiment_name
  run_name = "conditional_shoulder80_override_dual_gsi_ppo_smoke_seed42"
  command = build_train_command(cfg, B1, run_name, parent, smoke=True)
  before = set(experiment_root.glob(f"*_{run_name}"))
  state["stages"][stage_name] = {
    "status": "running",
    "started_at_utc": now(),
    "command": command,
    "parent_checkpoint": str(parent.checkpoint),
  }
  update(state, current_phase=stage_name)

  def on_start(pid: int) -> None:
    state["stages"][stage_name]["pid"] = pid
    update(state)

  _run_process(
    command, OUTPUT / "logs/ppo_smoke_dual_gsi.log", dry_run=False, on_start=on_start
  )
  run_dir = _find_new_run_dir(experiment_root, run_name, before)
  checkpoint = find_latest_checkpoint(run_dir)
  if validate_checkpoint(checkpoint) != 3000:
    raise RuntimeError("PPO smoke did not end at model_3000")
  _assert_finite_training_log(run_dir)
  state["stages"][stage_name].update(
    status="complete",
    finished_at_utc=now(),
    run_dir=str(run_dir),
    checkpoint=str(checkpoint),
    checkpoint_sha256=sha256(checkpoint),
  )
  update(state)


def contact_smoke(
  state: dict[str, Any], cfg: CurriculumCfg, parent: ResumeSource, side: str
) -> None:
  name = f"contact_smoke_dual_gsi_{side}"
  if state["stages"].get(name, {}).get("status") == "complete":
    return
  target = (0.0, 1.0, 0.0, 0.0) if side == "left" else (0.0, 0.0, 1.0, 0.0)
  small = replace(cfg, evaluation_num_envs=64, evaluation_num_episodes=64)
  output = OUTPUT / f"evaluation/{name}.json"
  command = build_evaluation_command(
    small,
    B1,
    parent.checkpoint,
    output,
    target_probabilities=target,
  )
  state["stages"][name] = {
    "status": "running",
    "started_at_utc": now(),
    "command": command,
  }
  update(state, current_phase=name)
  _run_process(command, OUTPUT / f"logs/{name}.log", dry_run=False)
  report = json.loads(output.read_text())
  coverage = float(report["metrics"]["contact_coverage"]["mean"])
  classified = float(
    report["metrics"]["contact_condition_classification_correct"]["mean"]
  )
  if not report.get("finite") or coverage < 0.80 or classified < 0.80:
    raise RuntimeError(
      f"{name} failed: finite={report.get('finite')}, "
      f"coverage={coverage}, classification={classified}"
    )
  state["stages"][name].update(
    status="complete",
    finished_at_utc=now(),
    report=str(output),
    contact_coverage=coverage,
    classification_correct=classified,
  )
  update(state)


def run(args: argparse.Namespace) -> None:
  os.chdir(ROOT)
  storage = require_storage()
  gpu = require_gpu0_idle()
  configure_environment()
  if sha256(PRIOR) != PRIOR_SHA256:
    raise RuntimeError("Shoulder80 candidate Prior identity changed")
  if sha256(GSI_PRIOR) != GSI_PRIOR_SHA256:
    raise RuntimeError("Conditional-v2 GSI Prior identity changed")
  prepare(PreparationCfg())
  if validate_checkpoint(MIGRATED) != 2999:
    raise RuntimeError("Migrated A checkpoint iteration mismatch")
  cfg = curriculum_cfg()
  state = (
    json.loads(MANIFEST.read_text())
    if MANIFEST.is_file()
    else initial_manifest(storage, gpu)
  )
  state.update(
    storage=storage,
    gpu=gpu,
    pid=os.getpid(),
    status="running",
    gsi_prior={
      "path": str(GSI_PRIOR),
      "sha256": GSI_PRIOR_SHA256,
      "role": "stand-only initialization; not used for SMP reward",
    },
  )
  update(state, current_phase="preparation")
  parent = ResumeSource(MIGRATED.parent, MIGRATED)
  try:
    smoke_train(state, cfg, parent)
    contact_smoke(state, cfg, parent, "left")
    contact_smoke(state, cfg, parent, "right")
    if args.smoke_only:
      update(state, status="smoke_complete", current_phase=None)
      return
    state_dir = Path(cfg.state_dir)
    evaluation_dir = Path(cfg.evaluation_dir)
    experiment_root = Path(cfg.log_root) / cfg.experiment_name
    for path in (state_dir, evaluation_dir, experiment_root):
      path.mkdir(parents=True, exist_ok=True)
    source, _ = _run_training_stage(
      cfg, state, MANIFEST, state_dir, evaluation_dir, experiment_root, B1, parent
    )
    source, _ = _run_training_stage(
      cfg, state, MANIFEST, state_dir, evaluation_dir, experiment_root, B2, source
    )
    update(
      state,
      status="complete",
      current_phase=None,
      finished_at_utc=now(),
      final_checkpoint=str(source.checkpoint),
      final_checkpoint_sha256=sha256(source.checkpoint),
    )
  except Exception as error:
    update(state, status="failed", failed_at_utc=now(), error=str(error))
    raise


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--smoke-only", action="store_true")
  args = parser.parse_args()
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run(args)


if __name__ == "__main__":
  main()
