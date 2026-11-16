"""Persistent GPU0 driver: data verification -> isolated smoke -> fresh 10000 epochs.

Never starts PPO, never increases budget, and never deletes historical artifacts.
Resume is explicit and keeps the full optimizer/RNG/sampler progress.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from scripts.curate_conditional_shoulder80 import atomic_json, now
from scripts.run_conditional_shoulder80_data_review import GPU_UUID, require_gpu

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder80 import CONDITION_SCHEMA, SAMPLING
from smp.pretrain.shoulder80_dataset import DATA_DIR, DATA_SHA

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/conditional_shoulder80_prior_v1_expanded"
PERSISTENT = Path(
  "/personal_fast_storage/smp_experiments/conditional_shoulder80_prior_v1_expanded_20261112"
)


def prepare():
  if not Path("/personal_fast_storage").is_mount():
    raise RuntimeError("Persistent storage absent; no root-disk fallback")
  if OUTPUT.is_symlink():
    if OUTPUT.resolve() != PERSISTENT:
      raise RuntimeError("Unexpected experiment destination")
  elif OUTPUT.exists() or PERSISTENT.exists():
    raise RuntimeError("Refusing to replace pre-existing experiment files")
  else:
    PERSISTENT.mkdir(parents=True)
    OUTPUT.symlink_to(PERSISTENT, target_is_directory=True)
  if shutil.disk_usage(OUTPUT).free < 5 * 1024**3:
    raise RuntimeError("Less than 5 GiB shared storage free")
  probe = OUTPUT / ".write_probe"
  with probe.open("xb") as stream:
    stream.write(bytes(1024**2))
    stream.flush()
    os.fsync(stream.fileno())
  probe.unlink()
  if sha256(DATA_DIR / "manifest.json") != DATA_SHA:
    raise ValueError("Approved data identity changed")
  data = json.loads((DATA_DIR / "manifest.json").read_text())
  if not data["gate"]["passed"]:
    raise ValueError("Data gate failed")
  environment = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "0",
    "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "SMP_PHYSICAL_GPU_UUID": GPU_UUID,
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "PYTHONUNBUFFERED": "1",
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
    Path(environment[key]).mkdir(parents=True, exist_ok=True)
  return environment


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prepare", action="store_true")
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  os.chdir(ROOT)
  environment = prepare()
  if args.prepare:
    print(
      json.dumps(
        {"logical": str(OUTPUT), "physical": str(PERSISTENT), "data_gate": True}
      )
    )
    return
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = OUTPUT / "manifest.json"
    if state_path.exists() and not args.resume:
      raise RuntimeError("Existing experiment; explicit --resume required")
    if args.resume and not state_path.exists():
      raise RuntimeError("Cannot resume missing experiment")
    state = (
      json.loads(state_path.read_text())
      if args.resume
      else {
        "experiment": "EXP-029",
        "created_at_utc": now(),
        "data_sha256": DATA_SHA,
        "data_manifest": str(DATA_DIR / "manifest.json"),
        "condition_schema": CONDITION_SCHEMA,
        "sampling": SAMPLING,
        "approved_epochs": 10000,
        "initialization": "scratch",
        "physical_storage": str(PERSISTENT),
        "personal_quota_verified": False,
        "quota_note": "Shared filesystem space is not a personal quota guarantee; stop on write errors.",
        "ppo_authorized": False,
        "prior_published": False,
        "history": [],
      }
    )
    if state["data_sha256"] != DATA_SHA or state["approved_epochs"] != 10000:
      raise ValueError("Driver resume identity mismatch")
    state.update(driver_pid=os.getpid(), updated_at_utc=now())
    child = None

    def stop(signum, _frame):
      if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
      state.update(status="interrupt_requested", signal=signum, updated_at_utc=now())
      atomic_json(state_path, state)
      # Let child finish its epoch and save; no forced kill or restart.

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def stage(name, module, extra):
      nonlocal child
      state["gpu"] = require_gpu()
      command = [sys.executable, "-u", "-m", module, *extra]
      log = OUTPUT / f"{name}.log"
      state.update(status=f"{name}_running", updated_at_utc=now())
      attempt = {
        "stage": name,
        "started_at_utc": now(),
        "command": command,
        "resume": "--resume" in extra,
      }
      state["history"].append(attempt)
      with log.open("a", buffering=1) as stream:
        child = subprocess.Popen(
          command,
          cwd=ROOT,
          env=environment,
          stdout=stream,
          stderr=subprocess.STDOUT,
          start_new_session=True,
        )
        attempt["pid"] = child.pid
        atomic_json(state_path, state)
        code = child.wait()
      attempt.update(returncode=code, ended_at_utc=now())
      atomic_json(state_path, state)
      if code:
        raise RuntimeError(f"{name} exited {code}; preserved state, inspect {log}")

    try:
      smoke_path = OUTPUT / "smoke/run.json"
      smoke_status = (
        json.loads(smoke_path.read_text())["status"] if smoke_path.exists() else None
      )
      if smoke_status != "smoke_passed":
        extra = ["--output", str(OUTPUT / "smoke"), "--smoke"]
        if args.resume and (OUTPUT / "smoke/latest.pt").exists():
          extra.append("--resume")
        stage("smoke", "scripts.pretrain_shoulder80", extra)
      smoke = json.loads((OUTPUT / "smoke/smoke_report.json").read_text())
      if not all(smoke[key] for key in ("finite_loss", "finite_score", "finite_ddpm")):
        raise RuntimeError("Prior smoke failed")
      state["smoke"] = smoke
      extra = ["--output", str(OUTPUT / "prior")]
      if args.resume and (OUTPUT / "prior/latest.pt").exists():
        extra.append("--resume")
      stage("prior", "scripts.pretrain_shoulder80", extra)
      training = json.loads((OUTPUT / "prior/run.json").read_text())
      state.update(
        completed_epochs=training["completed_epochs"],
        wandb_url=training["wandb_url"],
        selected=training["selected"],
        status=training["status"],
        updated_at_utc=now(),
      )
      atomic_json(state_path, state)
      # A selected candidate is locked, but never published merely on validation.
      # Test/replays require their independent acceptance stage, not an auto-pass.
      print(
        json.dumps({"status": state["status"], "epochs": state["completed_epochs"]}),
        flush=True,
      )
    except BaseException as error:
      state.update(
        status="stopped_error",
        error_type=type(error).__name__,
        error=str(error),
        updated_at_utc=now(),
      )
      atomic_json(state_path, state)
      raise


if __name__ == "__main__":
  main()
