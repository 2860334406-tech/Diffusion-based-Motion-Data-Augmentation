"""EXP-037: user-authorized exploratory PPO with the failed EXP-036 Prior.

Reuse the mixed-source scratch A/B1/B2 curriculum, without rerunning or
publishing a Prior. The original mirror score and all runtime checks remain.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import scripts.run_g1_shoulder_mixed_prior_ppo_suite as suite
import torch
from scripts.train_force_unloading_curriculum import CurriculumCfg

from smp.data.shoulder_ik import atomic_json
from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_mixed import CONDITION_SCHEMA
from smp.rl.conditional_score import load_score_calibration

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/reproduction/g1_shoulder_mixed_p01p02_prior_v1"
OUTPUT = ROOT / "logs/reproduction/g1_shoulder_mixed_p01p02_ppo_v1_override"
PERSISTENT_ROOT = Path("/personal_fast_storage/smp_experiments")
GPU_UUID = "GPU-eb0fc39f-6914-9e1d-65af-570524e8e061"
PRIOR = SOURCE / "prior/checkpoint_04000.pt"
PRIOR_SHA = "4df6197499cda4026839f596d21f89361114d232e239a2ec3101349c8fd75043"
SOURCE_CALIBRATION = SOURCE / "prior/validation/epoch_04000/calibration.pt"
CALIBRATION_SHA = "90894d01a2ea4fecdc59cc7fb07898dab1434928c5b24ccd99b78caa2e6d2139"
SOURCE_REPORT = SOURCE / "prior/validation/epoch_04000/report.json"
REPORT_SHA = "f4023fc830df8c809a18b758e7ffb801430a385d8753b4ecdedb263c821b4b62"
DATA_SHA = "41c2b9f9c466f8a12d86cb812157958b83ccd990533f416d173aaa053b2902f0"
CALIBRATION = OUTPUT / "calibration/runtime.pt"
MANIFEST = OUTPUT / "manifest.json"
PPO_MANIFEST = OUTPUT / "ppo/manifest.json"
EXPERIMENT = "smp_force_unloading_conditional_shoulder_p01p02_override"


def now():
  return datetime.now(timezone.utc).isoformat()


def storage():
  if not Path("/personal_fast_storage").is_mount():
    raise RuntimeError("Persistent storage is missing; no root-disk fallback")
  if OUTPUT.is_symlink():
    physical = OUTPUT.resolve(strict=True)
  elif OUTPUT.exists():
    raise RuntimeError("Output must be an isolated persistent-storage symlink")
  else:
    physical = PERSISTENT_ROOT / (
      OUTPUT.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    physical.mkdir()
    OUTPUT.symlink_to(physical, target_is_directory=True)
  if physical.parent != PERSISTENT_ROOT or not physical.name.startswith(OUTPUT.name):
    raise RuntimeError("Unexpected physical output directory")
  free = shutil.disk_usage(physical).free
  if free < 10 * 1024**3:
    raise RuntimeError("Less than 10 GiB free on persistent storage")
  probe = physical / ".write_probe"
  with probe.open("xb") as stream:
    stream.write(b"EXP-037")
    stream.flush()
    os.fsync(stream.fileno())
  probe.unlink()
  return {"logical": str(OUTPUT), "physical": str(physical), "free_bytes": free}


def gpu_idle():
  info = subprocess.check_output(
    ["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
    text=True,
  ).strip()
  index, uuid, name = [value.strip() for value in info.split(",", 2)]
  if index != "0" or uuid != GPU_UUID:
    raise RuntimeError(f"GPU0 identity changed: {info}")
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
    raise RuntimeError(f"GPU0 already in use; no process will be stopped: {active}")
  return {"physical_index": 0, "uuid": uuid, "name": name, "process_device": "cuda:0"}


def configure_environment():
  values = {
    "CUDA_VISIBLE_DEVICES": "0",
    "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "SMP_PHYSICAL_GPU_UUID": GPU_UUID,
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "WANDB_DIR": str(OUTPUT),
  }
  for name, relative in {
    "WANDB_CACHE_DIR": "cache/wandb",
    "XDG_CACHE_HOME": "cache/xdg",
    "TMPDIR": "tmp",
    "WARP_CACHE_PATH": "cache/warp",
    "MPLCONFIGDIR": "cache/matplotlib",
    "CUDA_CACHE_PATH": "cache/cuda",
    "TRITON_CACHE_DIR": "cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
  }.items():
    values[name] = str(OUTPUT / relative)
    Path(values[name]).mkdir(parents=True, exist_ok=True)
  os.environ.update(values)


def validate_sources():
  for path, expected in (
    (PRIOR, PRIOR_SHA),
    (SOURCE_CALIBRATION, CALIBRATION_SHA),
    (SOURCE_REPORT, REPORT_SHA),
    (SOURCE / "data/manifest.json", DATA_SHA),
  ):
    if sha256(path) != expected:
      raise ValueError(f"Frozen source identity mismatch: {path}")
  report = json.loads(SOURCE_REPORT.read_text())
  if report["epoch"] != 4000 or report["gate"]["passed"]:
    raise ValueError("Expected the explicitly unaccepted epoch-4000 candidate")
  payload = torch.load(PRIOR, map_location="cpu", weights_only=False)
  if payload["condition_schema"] != CONDITION_SCHEMA or payload["epoch"] != 4000:
    raise ValueError("Unexpected Prior/schema")
  if payload["data_sha256"] != DATA_SHA:
    raise ValueError("Prior training data identity mismatch")
  for name, value in {
    **payload["model"],
    "q_low": payload["q_low"],
    "q_high": payload["q_high"],
  }.items():
    if not torch.isfinite(torch.as_tensor(value)).all():
      raise FloatingPointError(f"Nonfinite Prior state: {name}")
  calibration, identity = load_score_calibration(
    SOURCE_CALIBRATION,
    "cpu",
    expected_prior_sha256=PRIOR_SHA,
    expected_num_conditions=3,
    expected_condition_schema=CONDITION_SCHEMA,
  )
  if calibration["data_manifest_sha256"] != DATA_SHA:
    raise ValueError("Calibration data identity mismatch")
  if calibration["normalization_sha256"] != payload["normalization_sha256"]:
    raise ValueError("Calibration normalization identity mismatch")
  if CALIBRATION.exists():
    if sha256(CALIBRATION) != CALIBRATION_SHA:
      raise ValueError("Isolated runtime calibration changed")
  else:
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = CALIBRATION.with_suffix(".tmp")
    shutil.copyfile(SOURCE_CALIBRATION, temporary)
    temporary.replace(CALIBRATION)
  return {
    "source_report": str(SOURCE_REPORT),
    "source_report_sha256": REPORT_SHA,
    "failed_checks": report["gate"]["failed_checks"],
    "calibration_identity": identity,
  }


def curriculum_cfg():
  return CurriculumCfg(
    task_id=suite.TASK,
    prior_checkpoint=str(PRIOR),
    prior_sha256=PRIOR_SHA,
    conditional_score_calibration=str(CALIBRATION),
    seed=42,
    gpu_id=0,
    num_envs=4096,
    smoke_num_envs=64,
    evaluation_num_envs=256,
    evaluation_num_episodes=1024,
    action_std=0.10,
    learning_rate=1e-4,
    save_interval=1000,
    gsi_max_root_planar_speed=0.15,
    stationary_speed_scale=10.0,
    sphere_radius=0.10,
    target_probabilities=(0.0, 0.5, 0.5, 0.0),
    push_direction_mode="front_to_back",
    experiment_name=EXPERIMENT,
    wandb_project="smp",
    run_prefix="g1_shoulder_p01p02_override",
    log_root=str(OUTPUT / "ppo/rsl_rl"),
    state_dir=str(OUTPUT / "ppo/state"),
    evaluation_dir=str(OUTPUT / "ppo/evaluation"),
    run_smoke=True,
    shoulder_split_evaluation=True,
    record_final_video=False,
  )


def update(state, **values):
  state.update(values, updated_at_utc=now())
  atomic_json(MANIFEST, state)


def freeze_code(state):
  paths = [
    Path(__file__).resolve().relative_to(ROOT),
    Path("scripts/run_g1_shoulder_mixed_prior_ppo_suite.py"),
    Path("scripts/train_force_unloading_curriculum.py"),
    Path("scripts/train.py"),
    Path("scripts/evaluate_shoulder_mixed.py"),
    Path("scripts/evaluate_force_unloading.py"),
    Path("src/smp/rl/tasks/force_unloading/conditional_shoulder_mixed.py"),
    Path("src/smp/rl/tasks/force_unloading/conditional.py"),
    Path("src/smp/rl/tasks/force_unloading/mdp/controller.py"),
    Path("src/smp/rl/rewards.py"),
    Path("src/smp/rl/events.py"),
    Path("src/smp/rl/conditional_score.py"),
    Path("src/smp/rl/rl_cfg.py"),
    Path("src/smp/rl/utils.py"),
    Path("src/smp/pretrain/schema.py"),
    Path("src/smp/pretrain/model.py"),
    Path("src/smp/pretrain/conditioning.py"),
  ]
  identity = {str(path): sha256(path) for path in paths}
  if state.get("code_identity", identity) != identity:
    raise RuntimeError("Frozen experiment code changed; investigate before resume")
  for path in paths:
    destination = OUTPUT / "audit/code" / path
    if destination.exists():
      if sha256(destination) != identity[str(path)]:
        raise ValueError(f"Code snapshot changed: {path}")
    else:
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(path, destination)
  state["code_identity"] = identity


def install_curriculum():
  # Module-local bindings affect this new process only; historical scripts and
  # task defaults stay untouched. The legacy name PUBLISHED_PRIOR is only the
  # curriculum's input path: no publishing function is called by run_ppo.
  suite.OUTPUT = OUTPUT
  suite.PPO_MANIFEST = PPO_MANIFEST
  suite.PUBLISHED_PRIOR = PRIOR
  suite.CALIBRATION = CALIBRATION
  suite.EXPERIMENT = EXPERIMENT
  suite.curriculum_cfg = curriculum_cfg
  suite.update = update

  def check_before_child():
    for path, expected in ((PRIOR, PRIOR_SHA), (CALIBRATION, CALIBRATION_SHA)):
      if sha256(path) != expected:
        raise ValueError(f"Frozen runtime identity changed: {path}")
    return gpu_idle()

  suite.require_gpu0_idle = check_before_child


def journal(state):
  ppo = json.loads(PPO_MANIFEST.read_text()) if PPO_MANIFEST.exists() else {}
  stages = {
    name: {
      key: record.get(key)
      for key in (
        "status",
        "checkpoint",
        "checkpoint_iteration",
        "gate_passed",
        "wandb_url",
      )
    }
    for name, record in ppo.get("stages", {}).items()
  }
  atomic_json(
    OUTPUT / "outcome.json",
    {
      "experiment": "EXP-037",
      "recorded_at_utc": now(),
      "status": state["status"],
      "stages": stages,
      "prior_gate_passed": False,
      "user_authorized_exploration": True,
    },
  )
  text = (
    f"\n- EXP-037 驱动状态（UTC {now()}）：status={state['status']}；"
    f"Prior仍未通过验收，按用户授权探索PPO。阶段={json.dumps(stages, ensure_ascii=False)}。"
  )
  if state.get("error"):
    text += f" 原因：{state['error']}。"
  with (ROOT / "docs/FORCE_UNLOADING_EXPERIMENT_LOG.md").open("a") as stream:
    stream.write(text + "\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--prepare-only", action="store_true")
  args = parser.parse_args()
  os.chdir(ROOT)
  disk = storage()
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = gpu_idle()
    configure_environment()
    inputs = validate_sources()
    if MANIFEST.exists():
      state = json.loads(MANIFEST.read_text())
      if not args.resume and state["status"] != "prepared":
        raise RuntimeError("Existing driver state requires explicit --resume")
      if state["prior"]["sha256"] != PRIOR_SHA:
        raise ValueError("Driver manifest Prior mismatch")
    else:
      state = {
        "schema_version": 1,
        "experiment": "EXP-037",
        "created_at_utc": now(),
        "status": "preparing",
        "prior": {
          "path": str(PRIOR),
          "sha256": PRIOR_SHA,
          "epoch": 4000,
          "formal_gate_passed": False,
          "published": False,
          "selection": "lowest validation loss, diagnostic candidate only",
        },
        "data_manifest_sha256": DATA_SHA,
        "authorization": {
          "user_authorized_failed_prior_override": True,
          "request": "先启动PPO训练看一下吧",
          "recorded_at_utc": now(),
          "claim_prior_accepted": False,
        },
        "score": {
          "mode": "shoulder_mirror_contrast_v1",
          "changed": False,
          "stand_contrast_added": False,
        },
        "ppo": {
          "policy_initialization": "scratch",
          "actor_critic_normalizers_optimizer": "scratch",
          "stages": ["a", "a_extension_if_needed", "b1", "b2"],
          "stage_c": False,
          "old_a_checkpoint_loaded": False,
        },
        "history": [],
      }
    freeze_code(state)
    update(
      state,
      storage=disk,
      gpu=gpu,
      inputs=inputs,
      config=asdict(curriculum_cfg()),
      driver_pid=os.getpid(),
      resume_requested=args.resume,
    )
    if args.prepare_only:
      update(state, status="prepared")
      print(json.dumps({"status": "prepared", "storage": disk, "gpu": gpu}))
      return
    install_curriculum()
    try:
      suite.run_ppo(state)
    except BaseException as error:
      if not str(state.get("status", "")).startswith("blocked"):
        update(
          state,
          status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
          error_type=type(error).__name__,
          error=str(error),
          failed_at_utc=now(),
        )
      raise
    finally:
      journal(state)


if __name__ == "__main__":
  main()
