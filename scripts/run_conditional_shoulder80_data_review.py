"""EXP-027 stage 1: isolated persistent storage and a hard data-review gate.

This entry point never starts Prior training or PPO. A successful review is a
prerequisite for implementing/running the remaining training stages; failure
requires an explicit data decision, not relaxed filters or automatic retries.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from scripts.curate_conditional_shoulder80 import atomic_json, now

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder80 import CONDITION_SCHEMA, SAMPLING, SPLIT_SUBJECTS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/conditional_shoulder80_prior_v1"
PERSISTENT = Path(
  "/personal_fast_storage/smp_experiments/conditional_shoulder80_prior_v1_20261112"
)
GPU_UUID = "GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771"
FORMAL_BUDGET = 10000
OLD_PRIOR = ROOT / "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
OLD_PRIOR_SHA = "ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734"
OLD_DATA = ROOT / "logs/reproduction/conditional_contact_v2/data/manifest.json"
OLD_DATA_SHA = "4e4a8226e40b43c3c00447c745ce75eea2ed6d065c55366650218e96f44ff57d"


def require_gpu():
  actual = subprocess.check_output(
    [
      "nvidia-smi",
      "-i",
      "0",
      "--query-gpu=uuid",
      "--format=csv,noheader",
    ],
    text=True,
  ).strip()
  if actual != GPU_UUID:
    raise RuntimeError(f"GPU0 UUID mismatch: {actual}; no fallback")
  active = subprocess.check_output(
    [
      "nvidia-smi",
      "-i",
      "0",
      "--query-compute-apps=pid",
      "--format=csv,noheader",
    ],
    text=True,
  ).strip()
  if active:
    raise RuntimeError(f"GPU0 has other compute processes: {active}; no preemption")
  return {
    "physical_index": 0,
    "uuid": actual,
    "process_device": "cuda:0",
    "review_device": "cpu",
    "compute_processes_at_check": [],
  }


def require_storage():
  mount = Path("/personal_fast_storage")
  if not mount.is_mount():
    raise RuntimeError("Persistent volume is not mounted; no system-disk fallback")
  if OUTPUT.is_symlink():
    if OUTPUT.resolve() != PERSISTENT:
      raise RuntimeError("Experiment symlink points to unexpected storage")
  elif OUTPUT.exists():
    raise RuntimeError("Refusing to overwrite a pre-existing experiment directory")
  else:
    if PERSISTENT.exists():
      raise RuntimeError(
        "Persistent experiment path already exists without its expected workspace link"
      )
    PERSISTENT.mkdir(parents=True)
    OUTPUT.symlink_to(PERSISTENT, target_is_directory=True)
  if not OUTPUT.is_dir():
    raise RuntimeError("Dedicated persistent experiment directory is unavailable")
  available = shutil.disk_usage(OUTPUT).free
  if available < 5 * 1024**3:
    raise RuntimeError("Less than 5 GiB filesystem free space; no automatic cleanup")
  probe = OUTPUT / ".storage_write_probe"
  with probe.open("xb") as stream:
    stream.write(bytes(1024**2))
    stream.flush()
    os.fsync(stream.fileno())
  probe.unlink()
  tools = {name: shutil.which(name) for name in ("quota", "mmlsquota", "lfs")}
  return {
    "logical_path": str(OUTPUT),
    "physical_path": str(PERSISTENT),
    "filesystem_free_bytes": available,
    "fsync_write_probe_bytes": 1024**2,
    "quota_tools": tools,
    "personal_quota_verified": False,
    "quota_note": "Shared filesystem free space is not a personal quota guarantee; stop on I/O/quota errors.",
  }


def review_environment():
  env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
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
    Path(env[key]).mkdir(parents=True, exist_ok=True)
  return env


def main():
  os.chdir(ROOT)
  gpu = require_gpu()
  if sha256(OLD_PRIOR) != OLD_PRIOR_SHA or sha256(OLD_DATA) != OLD_DATA_SHA:
    raise RuntimeError("Historical prior/data identity mismatch; no files modified")
  storage = require_storage()
  with (OUTPUT / "data_review.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUTPUT / "manifest.json"
    if path.exists():
      existing = json.loads(path.read_text())
      if existing["status"] == "blocked_data_gate":
        raise RuntimeError(
          "Data review is blocked; do not automatically retry or relax thresholds"
        )
      if existing["status"] == "data_ready":
        print(json.dumps({"status": "data_ready", "manifest": str(path)}), flush=True)
        return
      raise RuntimeError(
        "Previous review is incomplete; inspect its state before recovery"
      )
    state = {
      "experiment": "EXP-027",
      "name": "conditional_shoulder80_prior_v1",
      "created_at_utc": now(),
      "status": "data_review_running",
      "gpu": gpu,
      "storage": storage,
      "pid": os.getpid(),
      "condition_schema": CONDITION_SCHEMA,
      "sampling": SAMPLING,
      "split_subjects": SPLIT_SUBJECTS,
      "approved_formal_prior_epochs": FORMAL_BUDGET,
      "completed_formal_prior_epochs": 0,
      "prior_initialization": "scratch",
      "ppo_authorized": False,
      "prior_published": False,
      "formal_configuration": {
        "batch_size": 1024,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "num_timesteps": 50,
        "num_noise_samples": 10,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "condition_dropout": 0.1,
        "guidance_scale": 1.0,
        "ema": False,
        "seed": 42,
        "checkpoint_interval": 100,
        "save_interval": 1000,
        "upload_model": False,
      },
      "historical_identities": {
        str(OLD_PRIOR): OLD_PRIOR_SHA,
        str(OLD_DATA): OLD_DATA_SHA,
      },
      "stages": {
        "data_review": "running",
        "prior_smoke": "not_started",
        "prior_training": "not_started",
        "prior_acceptance": "not_started",
        "publication": "not_started",
      },
      "training_pipeline": "pending_data_acceptance",
    }
    atomic_json(path, state)
    child = None

    def stop(signum, _frame):
      if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=30)
      state.update(
        status="data_review_interrupted", signal=signum, updated_at_utc=now()
      )
      atomic_json(path, state)
      raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
      environment = review_environment()
      command = [sys.executable, "-u", "-m", "scripts.curate_conditional_shoulder80"]
      state["data_command"] = command
      atomic_json(path, state)
      with (OUTPUT / "data_review.log").open("x", buffering=1) as stream:
        child = subprocess.Popen(
          command,
          cwd=ROOT,
          env=environment,
          stdout=stream,
          stderr=subprocess.STDOUT,
          start_new_session=True,
        )
        state["child_pid"] = child.pid
        atomic_json(path, state)
        code = child.wait()
      report_path = OUTPUT / "data/manifest.json"
      if code not in (0, 2) or not report_path.is_file():
        raise RuntimeError(
          f"Data review process exited {code}; inspect data_review.log"
        )
      report = json.loads(report_path.read_text())
      if (code == 0) != report["gate"]["passed"]:
        raise RuntimeError("Data review exit status disagrees with saved gate")
      state.update(
        status="data_ready" if report["gate"]["passed"] else "blocked_data_gate",
        data_manifest=str(report_path),
        data_manifest_sha256=sha256(report_path),
        data_gate=report["gate"],
        data_counts=report["counts"],
        data_phase_counts=report["phase_counts"],
        data_returncode=code,
        updated_at_utc=now(),
      )
      state["stages"]["data_review"] = (
        "passed" if report["gate"]["passed"] else "blocked"
      )
      for historical, digest in state["historical_identities"].items():
        if sha256(historical) != digest:
          raise RuntimeError("Historical artifact changed during review")
      atomic_json(path, state)
      print(
        json.dumps(
          {
            "status": state["status"],
            "counts": report["counts"],
            "failed_checks": report["gate"]["failed_checks"],
            "manifest": str(path),
          }
        ),
        flush=True,
      )
    except Exception as error:
      state.update(status="data_review_failed", error=str(error), updated_at_utc=now())
      atomic_json(path, state)
      raise


if __name__ == "__main__":
  main()
