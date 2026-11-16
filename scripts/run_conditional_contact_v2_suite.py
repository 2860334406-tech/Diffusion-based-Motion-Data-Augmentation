"""GPU1 v2 experiment: stronger shoulder data, one new 10,000-epoch budget.

All new bulk artifacts/caches live on the persistent volume. The workspace
symlink preserves the expected logs/reproduction path; v1 is never rewritten.
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

from scripts.run_conditional_contact_suite import ROOT, TASK, atomic_json, now, sha256

OUTPUT = ROOT / "logs/reproduction/conditional_contact_v2"
PERSISTENT = Path("/personal_fast_storage/smp_experiments/conditional_contact_v2_20261111")
PRIOR = ROOT / "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
GPU_UUID = "GPU-4169da3c-33af-015e-5902-355f8e02a6c9"
BUDGET = 10000


def child_environment():
  env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "MUJOCO_GL": "egl",
         "MUJOCO_EGL_DEVICE_ID": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
         "SMP_PHYSICAL_GPU_UUID": GPU_UUID, "WANDB_DIR": str(OUTPUT),
         "WANDB_CACHE_DIR": str(OUTPUT / "cache/wandb"),
         "MPLCONFIGDIR": str(OUTPUT / "cache/matplotlib"),
         "WARP_CACHE_PATH": str(OUTPUT / "cache/warp"),
         "TORCHINDUCTOR_CACHE_DIR": str(OUTPUT / "cache/inductor"),
         "TRITON_CACHE_DIR": str(OUTPUT / "cache/triton"),
         "CUDA_CACHE_PATH": str(OUTPUT / "cache/cuda"),
         "XDG_CACHE_HOME": str(OUTPUT / "cache/xdg"),
         "TMPDIR": str(OUTPUT / "tmp")}
  return env


def prior_command(smoke=False):
  return [
    sys.executable, "-u", "-m", "scripts.pretrain_conditional_contact",
    "--data-dir", str(OUTPUT / "data"),
    "--output-dir", str(OUTPUT / ("prior_smoke" if smoke else "prior")),
    "--num-epochs", "1" if smoke else str(BUDGET),
    "--name", "conditional_contact_v2_prior_seed42_gpu1",
    *(["--no-use-wandb"] if smoke else []),
  ]


def curriculum_command(digest):
  return [
    sys.executable, "-m", "scripts.train_force_unloading_curriculum",
    "--task-id", TASK, "--all-contact-evaluation",
    "--prior-checkpoint", str(PRIOR), "--prior-sha256", digest,
    "--gpu-id", "1", "--gsi-max-root-planar-speed", "0.15",
    "--stationary-speed-scale", "10", "--sphere-radius", "0.10",
    "--target-probabilities", "0.25", "0.25", "0.25", "0.25",
    "--push-direction-mode", "front_to_back",
    "--experiment-name", "smp_force_unloading_conditional_g1",
    "--run-prefix", "conditional_contact_v2",
    "--log-root", str(OUTPUT / "rsl_rl"),
    "--state-dir", str(OUTPUT / "curriculum"),
    "--evaluation-dir", str(OUTPUT / "evaluation"),
  ]


def require_storage():
  if not OUTPUT.is_symlink() or OUTPUT.resolve() != PERSISTENT.resolve():
    raise RuntimeError("v2 output must use the dedicated persistent volume, not the full system disk")
  free = shutil.disk_usage(OUTPUT).free
  if free < 2*1024**3:
    raise RuntimeError("Less than 2 GiB free on experiment storage; no automatic cleanup")
  return {"logical_path": str(OUTPUT), "physical_path": str(OUTPUT.resolve()), "free_bytes": free}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--resume-after-diagnosis", action="store_true")
  args = parser.parse_args()
  os.chdir(ROOT)
  require_storage()
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run(args)


def run(args):
  storage = require_storage()
  actual = subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader",
  ], text=True).strip()
  if actual != GPU_UUID:
    raise RuntimeError("GPU1 identity changed")
  active = subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader",
  ], text=True).strip()
  if active:
    raise RuntimeError(f"GPU1 already has compute processes: {active}; no processes will be stopped")
  path = OUTPUT / "manifest.json"
  state = json.loads(path.read_text()) if path.exists() else {
    "schema_version": 2, "created_at_utc": now(), "phases": {},
    "experiment": "conditional_contact_v2", "parent_experiment": "EXP-024",
    "approved_additional_prior_epochs": BUDGET,
    "gpu": {"index": 1, "uuid": GPU_UUID, "process_device": "cuda:0"},
    "fixed_config": {
      "prior_initialization": "scratch", "actor_initialization": "scratch",
      "sampling": "5 x 672; each shoulder 336 entering + 336 holding",
      "guidance_scale": 1, "gsi_condition": 0, "gsi_speed": .15, "stationary_scale": 10.,
      "seed": 42, "prior_epochs": BUDGET, "upload_model": False,
      "ppo_num_envs": 4096, "ppo_save_interval": 1000,
      "ppo_stages": {"a": 3000, "a_extension_if_needed": 2000, "b1": 4000, "b2": 4000, "c": 19000},
      "ppo_target_probabilities": [.25]*4, "acceptance_thresholds": "unchanged from v1",
    },
  }
  if state["approved_additional_prior_epochs"] != BUDGET:
    raise RuntimeError("Approved training budget mismatch")
  if state.get("status") in {"failed", "blocked_prior_gate"} and not args.resume_after_diagnosis:
    raise RuntimeError("Prior failure requires diagnosis; no automatic budget reset or retries")
  state.update(status="running", pid=os.getpid(), storage=storage, updated_at_utc=now())
  atomic_json(path, state)
  env = child_environment()
  for name in ("TMPDIR", "WANDB_CACHE_DIR", "MPLCONFIGDIR", "WARP_CACHE_PATH",
               "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME"):
    Path(env[name]).mkdir(parents=True, exist_ok=True)
  child = None

  def stop(signum, _frame):
    state.update(status="interrupted", signal=signum, updated_at_utc=now())
    atomic_json(path, state)
    if child is not None and child.poll() is None:
      os.killpg(child.pid, signal.SIGTERM)
      child.wait(timeout=60)
    raise SystemExit(128+signum)

  signal.signal(signal.SIGTERM, stop)
  signal.signal(signal.SIGINT, stop)

  def phase(name, command):
    nonlocal child
    require_storage()
    record = state["phases"].setdefault(name, {})
    if record.get("status") == "complete":
      return
    if record:
      record.setdefault("recovery_history", []).append({
        "time": now(), "previous_status": record.get("status"),
        "previous_pid": record.get("pid"), "budget_ceiling": BUDGET if name == "prior" else None,
      })
      if name == "prior" and not (OUTPUT / "prior/latest.pt").is_file():
        raise RuntimeError("Interrupted prior has no full checkpoint; refusing to silently restart its budget")
    record.update(status="running", started_at_utc=now(), command=command)
    state["current_phase"] = name
    atomic_json(path, state)
    print(f"[{now()}] START {name}", flush=True)
    with (OUTPUT / f"{name}.log").open("a", buffering=1) as stream:
      child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
      record["pid"] = child.pid
      atomic_json(path, state)
      code = child.wait()
    record.update(returncode=code, finished_at_utc=now(), status="complete" if code == 0 else "failed")
    atomic_json(path, state)
    if code:
      raise RuntimeError(f"{name} exited {code}; see {OUTPUT / (name+'.log')}")
    print(f"[{now()}] COMPLETE {name}", flush=True)

  try:
    phase("data", [sys.executable, "-m", "scripts.curate_conditional_contact_v2"])
    data_sha = sha256(OUTPUT / "data/manifest.json")
    if state.get("data_sha256", data_sha) != data_sha:
      raise RuntimeError("Dataset changed during experiment")
    state["data_sha256"] = data_sha
    phase("prior_smoke", prior_command(smoke=True))
    phase("contact_smoke", [sys.executable, "-m", "scripts.smoke_conditional_contact",
                            "--output", str(OUTPUT / "contact_smoke.json")])
    phase("prior", prior_command())
    state["prior_training"] = json.loads((OUTPUT / "prior/run.json").read_text())
    if state["prior_training"]["completed_epochs"] != BUDGET:
      raise RuntimeError("Formal prior did not complete its exact approved budget")
    candidate = OUTPUT / "prior/pretrained.pt"
    phase("prior_validation", [
      sys.executable, "-m", "scripts.validate_conditional_contact",
      "--checkpoint", str(candidate), "--data-dir", str(OUTPUT / "data"),
      "--output-dir", str(OUTPUT / "prior_validation"),
    ])
    report = json.loads((OUTPUT / "prior_validation/report.json").read_text())
    if not report["gate"]["passed"]:
      raise RuntimeError("Conditional prior gate failed; PPO prohibited")
    digest = sha256(candidate)
    if PRIOR.exists() or PRIOR.is_symlink():
      if not PRIOR.is_file() or sha256(PRIOR) != digest:
        raise RuntimeError("Refusing to overwrite an existing promoted prior")
    else:
      PRIOR.symlink_to(candidate.resolve())
    state["prior"] = {"path": str(PRIOR), "sha256": digest, "data_sha256": data_sha,
                      "candidate_path": str(candidate.resolve())}
    atomic_json(path, state)
    phase("ppo_curriculum", curriculum_command(digest))
    state.update(status="complete", finished_at_utc=now(),
                 curriculum=str(OUTPUT / "curriculum/curriculum_state.json"))
  except Exception as error:
    state.update(status="blocked_prior_gate" if state.get("current_phase") == "prior_validation" else "failed",
                 error=str(error), failed_at_utc=now())
    report_path = OUTPUT / "prior_validation/report.json"
    if report_path.is_file():
      state["prior_gate"] = json.loads(report_path.read_text())["gate"]
      state["prior_validation_report"] = str(report_path)
    raise
  finally:
    state["updated_at_utc"] = now()
    atomic_json(path, state)


if __name__ == "__main__":
  main()
