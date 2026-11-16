"""EXP-038: frozen-split LAFAN90 stand-source Prior training, with no PPO stage."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from smp.data.shoulder_ik import atomic_json
from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_mixed_lafan90 import (
  NORM_SHA, PARENT_SHA, POLICY, SAMPLING, data_sha, prepare,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/g1_shoulder_mixed_lafan90_prior_v1"
GPU_UUID = "GPU-eb0fc39f-6914-9e1d-65af-570524e8e061"
TERMINAL = {"validation_candidate_selected", "training_complete_no_score_candidate"}


def now():
  return datetime.now(timezone.utc).isoformat()


def gpu_idle():
  info = subprocess.check_output(
    ["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
    text=True,
  ).strip()
  index, uuid, name = [v.strip() for v in info.split(",", 2)]
  if index != "0" or uuid != GPU_UUID:
    raise RuntimeError(f"GPU0 identity changed: {info}")
  processes = subprocess.check_output(
    [
      "nvidia-smi",
      "-i",
      "0",
      "--query-compute-apps=pid,process_name",
      "--format=csv,noheader",
    ],
    text=True,
  ).strip()
  if processes:
    raise RuntimeError(f"GPU0 busy; no task will be interrupted: {processes}")
  return {"physical_index": 0, "uuid": uuid, "name": name, "process_device": "cuda:0"}


def storage():
  mount = Path("/personal_fast_storage")
  if not mount.is_mount():
    raise RuntimeError("Persistent storage missing; local fallback is prohibited")
  if OUTPUT.is_symlink():
    physical = OUTPUT.resolve()
  elif OUTPUT.exists():
    raise RuntimeError("Expected an isolated persistent-storage symlink")
  else:
    physical = (
      mount
      / "smp_experiments"
      / (
        "g1_shoulder_mixed_lafan90_prior_v1_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
      )
    )
    physical.mkdir()
    OUTPUT.symlink_to(physical, target_is_directory=True)
  if shutil.disk_usage(physical).free < 10 * 1024**3:
    raise RuntimeError("Less than 10 GiB free for this independent experiment")
  probe = physical / ".write_probe"
  with probe.open("xb") as stream:
    stream.write(b"EXP-038")
    stream.flush()
    os.fsync(stream.fileno())
  probe.unlink()
  return {"logical": str(OUTPUT), "physical": str(physical)}


def environment():
  values = {
    "CUDA_VISIBLE_DEVICES": "0",
    "SMP_PHYSICAL_GPU_UUID": GPU_UUID,
    "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
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


def save(state, **values):
  state.update(values, updated_at_utc=now())
  atomic_json(OUTPUT / "manifest.json", state)


def freeze_code(state):
  paths = [
    Path(__file__).resolve().relative_to(ROOT),
    Path("scripts/pretrain_shoulder_mixed_lafan90.py"),
    Path("scripts/assess_shoulder_mixed_lafan90.py"),
    Path("src/smp/pretrain/shoulder_mixed_lafan90.py"),
    Path("src/smp/pretrain/shoulder_mixed_p01p02_dataset.py"),
    Path("src/smp/pretrain/shoulder_mixed_dataset.py"),
    Path("src/smp/pretrain/shoulder_mixed.py"),
    Path("src/smp/pretrain/shoulder_mixed_score.py"),
    Path("src/smp/pretrain/shoulder80_score.py"),
    Path("src/smp/rl/conditional_score.py"),
    Path("src/smp/pretrain/model.py"),
    Path("src/smp/pretrain/schema.py"),
    Path("src/smp/pretrain/conditioning.py"),
    Path("src/smp/pretrain/scheduler.py"),
    Path("scripts/pretrain.py"),
  ]
  identity = {str(p): sha256(p) for p in paths}
  if state.get("code_identity", identity) != identity:
    raise ValueError("Frozen experiment code identity changed")
  state["code_identity"] = identity
  for path in paths:
    destination = OUTPUT / "audit/code" / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
      if sha256(destination) != identity[str(path)]:
        raise ValueError("Frozen code snapshot mismatch")
    else:
      shutil.copyfile(path, destination)


def child(state, stage, command):
  gpu_idle()
  attempt = {
    "stage": stage,
    "command": command,
    "status": "running",
    "started_at_utc": now(),
  }
  state["history"].append(attempt)
  log = OUTPUT / "logs" / f"{stage}.log"
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("a", buffering=1) as stream:
    process = subprocess.Popen(
      command,
      cwd=ROOT,
      env=os.environ.copy(),
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
    attempt["pid"] = process.pid
    save(state, status=stage + "_running")

    def forward(signum, frame):
      if process.poll() is None:
        process.send_signal(signal.SIGTERM)
      save(state, status="interrupt_requested")

    previous = {
      sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
      code = process.wait()
    finally:
      for sig, handler in previous.items():
        signal.signal(sig, handler)
  attempt.update(
    status="complete" if code == 0 else "failed", returncode=code, finished_at_utc=now()
  )
  save(state)
  if code:
    raise RuntimeError(f"{stage} exited {code}; see {log}")


def train_stages(state, resume):
  for stage, smoke in (("prior_smoke", True), ("prior", False)):
    folder = OUTPUT / stage
    run_path = folder / "run.json"
    previous = json.loads(run_path.read_text()) if run_path.exists() else {}
    finished = (
      previous.get("status") == "smoke_passed"
      if smoke
      else previous.get("status") in TERMINAL
    )
    if not finished:
      command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.pretrain_shoulder_mixed_lafan90",
        "--output",
        str(folder),
      ]
      if smoke:
        command.append("--smoke")
      if (folder / "latest.pt").exists():
        if not resume:
          raise RuntimeError("Existing training state requires explicit --resume")
        command.append("--resume")
      child(state, stage, command)
    run = json.loads(run_path.read_text())
    if smoke:
      report = json.loads((folder / "smoke_report.json").read_text())
      if not all(report[k] for k in ("finite_loss", "finite_ddpm", "finite_score")):
        raise FloatingPointError("Prior smoke produced invalid values")
      state["smoke"] = report
    else:
      if run["completed_epochs"] != 10000 or run["optimizer_steps"] != 40000:
        raise RuntimeError("Formal training did not complete the fixed budget")
      diagnostic = min(
        run["evaluations"], key=lambda item: item["weighted_validation_loss"]
      )
      state["prior"].update(
        completed_epochs=run["completed_epochs"],
        optimizer_steps=run["optimizer_steps"],
        wandb_url=run["wandb_url"],
        selected_validation_candidate=run["selected"],
        diagnostic_lowest_loss=diagnostic,
        validation_passed=run["selected"] is not None,
      )
      status = (
        "complete_validation_passed"
        if run["selected"]
        else "complete_validation_failed"
      )
      save(
        state,
        status=status,
        finished_at_utc=now(),
        ppo_authorized=False,
        test_evaluated=False,
      )


def assess_stage(state):
  child(state, "assessment", [sys.executable, "-u", "-m",
        "scripts.assess_shoulder_mixed_lafan90", "--output", str(OUTPUT / "assessment")])
  report = json.loads((OUTPUT / "assessment/report.json").read_text())
  save(state, assessment=report, status=("complete_assessment_passed"
       if report["gate"]["passed"] else "complete_assessment_failed"),
       finished_at_utc=now(), ppo_authorized=False, test_evaluated=False)


def journal(state):
  path = OUTPUT / "prior/run.json"
  prior = json.loads(path.read_text()) if path.exists() else {}
  text = (
    f"\n- EXP-038 驱动结果（UTC {now()}）：status={state['status']}，"
    f"Prior实际{prior.get('completed_epochs', 0)}/10000epochs，W&B={prior.get('wandb_url')}；"
    f"selected_validation_candidate={prior.get('selected')}。本轮只训练Prior，P03未验收，PPO未启动。"
  )
  if state.get("error"):
    text += f" 原因：{state['error']}。"
  with (ROOT / "docs/FORCE_UNLOADING_EXPERIMENT_LOG.md").open("a") as stream:
    stream.write(text + "\n")
  atomic_json(
    OUTPUT / "outcome.json",
    {
      "status": state["status"],
      "prior": prior,
      "ppo_started": False,
      "test_evaluated": False,
      "assessment": state.get("assessment"),
      "at_utc": now(),
    },
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--prepare-only", action="store_true")
  args = parser.parse_args()
  os.chdir(ROOT)
  disk, gpu = storage(), gpu_idle()
  environment()
  with (OUTPUT / "driver.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = prepare()
    path = OUTPUT / "manifest.json"
    state = (
      json.loads(path.read_text())
      if path.exists()
      else {
        "experiment": "EXP-038",
        "schema_version": 1,
        "created_at_utc": now(),
        "status": "prepared",
        "prior_only": True,
        "ppo_authorized": False,
        "data_manifest_sha256": data_sha(),
        "split_policy": POLICY,
        "sampling": SAMPLING,
        "normalization_sha256": NORM_SHA,
        "normalization_recomputed": False,
        "parent_data_sha256": PARENT_SHA,
        "prior": {
          "initialization": "scratch",
          "budget_epochs": 10000,
          "published": False,
        },
        "history": [],
      }
    )
    if state["data_manifest_sha256"] != data_sha():
      raise ValueError("Dataset changed since the experiment was prepared")
    state.update(
      storage=disk, gpu=gpu, driver_pid=os.getpid(), data_shards=data["shards"]
    )
    freeze_code(state)
    save(state)
    if args.prepare_only:
      print(
        json.dumps(
          {"storage": disk, "gpu": gpu, "data_manifest_sha256": data_sha()}, indent=2
        )
      )
      return
    try:
      train_stages(state, args.resume)
      assess_stage(state)
    except BaseException as error:
      save(
        state,
        status="failed_preserved",
        error_type=type(error).__name__,
        error=str(error),
        failed_at_utc=now(),
      )
      raise
    finally:
      journal(state)


if __name__ == "__main__":
  main()
