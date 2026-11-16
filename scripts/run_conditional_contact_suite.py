"""Persistent GPU1-only conditional prior -> gated PPO experiment driver.

Run via nohup/setsid, never a terminal pipe. A failed prior gate cannot start PPO.
Restarting resumes an interrupted prior from its full latest checkpoint; numeric,
quality and checkpoint failures require diagnosis, not blind automatic retries.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/conditional_contact_v1"
PRIOR = ROOT / "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v1.pt"
TASK = "Smp-Force-Unloading-Conditional-G1"


def now():
  return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
  temporary = path.with_suffix(".json.tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
  temporary.replace(path)


def sha256(path):
  import hashlib
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def curriculum_command(prior_sha):
  return [
    sys.executable, "-m", "scripts.train_force_unloading_curriculum",
    "--task-id", TASK, "--all-contact-evaluation",
    "--prior-checkpoint", str(PRIOR), "--prior-sha256", prior_sha,
    "--gpu-id", "1", "--gsi-max-root-planar-speed", "0.15",
    "--stationary-speed-scale", "10", "--sphere-radius", "0.10",
    "--target-probabilities", "0.25", "0.25", "0.25", "0.25",
    "--push-direction-mode", "front_to_back",
    "--experiment-name", "smp_force_unloading_conditional_g1",
    "--run-prefix", "conditional_contact_v1",
    "--log-root", str(OUTPUT / "rsl_rl"),
    "--state-dir", str(OUTPUT / "curriculum"),
    "--evaluation-dir", str(OUTPUT / "evaluation"),
  ]


def main():
  os.chdir(ROOT)
  OUTPUT.mkdir(parents=True, exist_ok=True)
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_locked()


def run_locked():
  info = subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-gpu=index,uuid,name",
    "--format=csv,noheader",
  ], text=True).strip().split(", ")
  gpu = {"index": int(info[0]), "uuid": info[1], "name": info[2]}
  if gpu["index"] != 1 or "A100" not in gpu["name"]:
    raise RuntimeError(f"Expected physical A100 GPU1: {gpu}")
  path = OUTPUT / "manifest.json"
  state = json.loads(path.read_text()) if path.exists() else {
    "schema_version": 1, "created_at_utc": now(), "gpu": gpu, "phases": {},
    "fixed_config": {
      "conditions": ["stand", "left_shoulder", "right_shoulder", "chest", "back"],
      "null_id": 5, "prior_epochs": 10000, "samples_per_epoch": 3360,
      "guidance_scale": 1, "gsi_condition": 0, "gsi_max_root_planar_speed": .15,
      "stationary_speed_scale": 10., "sphere_radius": .10, "sphere_mass": 5.,
      "initial_surface_gap": .02, "target_probabilities": [.25]*4,
      "push_direction_mode": "front_to_back_with_back_reversed",
      "seed": 42, "ppo_num_envs": 4096, "ppo_save_interval": 1000,
      "ppo_budget": [3000, 2000, 4000, 4000, 19000],
      "ppo_a_extension": "only_if_initial_A_gate_fails", "scratch_prior_and_actor": True,
      "upload_models": False,
    },
  }
  if state["gpu"]["uuid"] != gpu["uuid"]:
    raise RuntimeError("GPU1 UUID changed; refuse to silently switch devices")
  if state.get("status") in {"failed", "blocked_prior_gate"}:
    if "--resume-after-diagnosis" not in sys.argv:
      raise RuntimeError("Previous hard failure: diagnose first, then use --resume-after-diagnosis")
  state.update(status="running", pid=os.getpid(), updated_at_utc=now())
  atomic_json(path, state)
  env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "MUJOCO_GL": "egl",
         "MUJOCO_EGL_DEVICE_ID": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
         "SMP_PHYSICAL_GPU_UUID": gpu["uuid"], "WANDB_DIR": str(OUTPUT),
         "WANDB_CACHE_DIR": str(OUTPUT / "wandb_cache"),
         "MPLCONFIGDIR": str(OUTPUT / "matplotlib")}
  child = None

  def interrupted(signum, _frame):
    state.update(status="interrupted", signal=signum, updated_at_utc=now())
    atomic_json(path, state)
    if child is not None and child.poll() is None:
      os.killpg(child.pid, signal.SIGTERM)
      child.wait(timeout=60)
    raise SystemExit(128+signum)

  signal.signal(signal.SIGINT, interrupted)
  signal.signal(signal.SIGTERM, interrupted)

  def phase(name, command):
    nonlocal child
    record = state["phases"].setdefault(name, {})
    if record.get("status") == "complete":
      return
    if record:
      record.setdefault("recovery_history", []).append({
        "time": now(), "previous_status": record.get("status"),
        "previous_pid": record.get("pid"),
        "prior_resume_checkpoint": str(OUTPUT / "prior/latest.pt") if name == "prior" else None,
      })
    record.update(status="running", command=command, started_at_utc=now())
    state["current_phase"] = name
    atomic_json(path, state)
    with (OUTPUT / f"{name}.log").open("a", buffering=1) as log:
      print(f"[{now()}] START {name}", flush=True)
      child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
      record["pid"] = child.pid
      atomic_json(path, state)
      returncode = child.wait()
    record.update(returncode=returncode, finished_at_utc=now())
    if returncode:
      record["status"] = "failed"
      raise RuntimeError(f"{name} exited {returncode}; inspect {OUTPUT / (name+'.log')}")
    record["status"] = "complete"
    atomic_json(path, state)
    print(f"[{now()}] COMPLETE {name}", flush=True)

  try:
    phase("data", [sys.executable, "-m", "scripts.curate_conditional_contact"])
    state["data_sha256"] = sha256(OUTPUT / "data/manifest.json")
    phase("prior_smoke", [sys.executable, "-m", "scripts.pretrain_conditional_contact",
                         "--output-dir", "logs/reproduction/conditional_contact_v1/prior_smoke",
                         "--num-epochs", "1", "--no-use-wandb"])
    phase("contact_smoke", [sys.executable, "-m", "scripts.smoke_conditional_contact"])
    phase("prior", [sys.executable, "-u", "-m", "scripts.pretrain_conditional_contact"])
    state["prior_training"] = json.loads((OUTPUT / "prior/run.json").read_text())
    phase("prior_validation", [sys.executable, "-m", "scripts.validate_conditional_contact"])
    report = json.loads((OUTPUT / "prior_validation/report.json").read_text())
    if not report["gate"]["passed"]:
      state["status"] = "blocked_prior_gate"
      raise RuntimeError("Conditional prior failed acceptance; formal PPO prohibited")
    candidate = OUTPUT / "prior/pretrained.pt"
    digest = sha256(candidate)
    if PRIOR.exists():
      if sha256(PRIOR) != digest:
        raise RuntimeError("Refusing to overwrite an existing promoted conditional prior")
    else:
      temporary = PRIOR.with_suffix(".pt.pending")
      shutil.copyfile(candidate, temporary)
      temporary.replace(PRIOR)
    state["prior"] = {"path": str(PRIOR), "sha256": digest,
                      "data_sha256": state["data_sha256"], "guidance_scale": 1}
    atomic_json(path, state)
    free = shutil.disk_usage(OUTPUT).free
    if free < 2 * 1024**3:
      state["storage"] = {"free_bytes": free, "minimum_free_bytes": 2 * 1024**3}
      raise RuntimeError("Less than 2 GiB free before long PPO training; provide storage, "
                         "historical artifacts will not be deleted automatically")
    phase("ppo_curriculum", curriculum_command(digest))
    state["curriculum"] = str(OUTPUT / "curriculum/curriculum_state.json")
    state.update(status="complete", finished_at_utc=now())
  except Exception as exc:
    if state.get("current_phase") == "prior_validation":
      state["status"] = "blocked_prior_gate"
    elif state.get("status") != "interrupted":
      state["status"] = "failed"
    state.update(error=str(exc), failed_at_utc=now())
    raise
  finally:
    state["updated_at_utc"] = now()
    atomic_json(path, state)


if __name__ == "__main__":
  main()
