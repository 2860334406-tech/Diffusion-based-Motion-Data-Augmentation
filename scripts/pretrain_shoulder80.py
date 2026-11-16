"""EXP-029: from-scratch three-condition Prior; exact 80/20 sampling and resume."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from scripts.curate_conditional_shoulder80 import atomic_json, now
from scripts.pretrain import _diffusion_loss
from scripts.run_conditional_shoulder80_data_review import GPU_UUID
from torch.utils.data import DataLoader

from smp.pretrain.conditioning import sample_condition, sha256
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.pretrain.shoulder80 import (
  CONDITION_NAMES,
  CONDITION_SCHEMA,
  SAMPLING,
  Shoulder80Sampler,
)
from smp.pretrain.shoulder80_dataset import DATA_DIR, DATA_SHA, Shoulder80Dataset
from smp.pretrain.shoulder80_score import calibrate, evaluate_scores, paired_scores
from smp.rl.conditional_score import FIXED_TIMESTEPS, SCORE_MODE, score_feature_weights
from smp.utils import seed_everything

FORMAL = dict(
  num_epochs=10000,
  batch_size=1024,
  d_model=128,
  nhead=4,
  num_layers=2,
  num_timesteps=50,
  num_noise_samples=10,
  lr=3e-4,
  weight_decay=1e-4,
  max_grad_norm=1.0,
  condition_dropout=0.1,
  seed=42,
  feature_dim=59,
  window_size=10,
  num_conditions=3,
  ema=False,
  checkpoint_interval=100,
  save_interval=1000,
  validation_score_repeats=64,
  guidance_scale=1.0,
)


def rng_state():
  return {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
  }


def restore_rng(state):
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"].cpu())
  if state["cuda"]:
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_checkpoint(path, payload):
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("wb") as stream:
    torch.save(payload, stream)
    stream.flush()
    os.fsync(stream.fileno())
  temporary.replace(path)
  atomic_json(
    path.with_suffix(".identity.json"),
    {"sha256": sha256(path), "epoch": payload["epoch"]},
  )


def verified_checkpoint(path):
  identity = json.loads(path.with_suffix(".identity.json").read_text())
  if sha256(path) != identity["sha256"]:
    raise ValueError("Checkpoint SHA-256 mismatch")
  payload = torch.load(path, weights_only=False, map_location="cpu")
  if payload["epoch"] != identity["epoch"]:
    raise ValueError("Checkpoint epoch identity mismatch")
  return payload


def restore_training(payload, model, optimizer, sampler, cfg, data_sha):
  if payload["cfg"] != cfg or payload["condition_schema"] != CONDITION_SCHEMA:
    raise ValueError("Resume configuration/schema mismatch")
  if payload["data_sha256"] != data_sha or payload["sampling"] != SAMPLING:
    raise ValueError("Resume data/sampling mismatch")
  if payload["initialization"] != "scratch":
    raise ValueError("Unexpected warm start")
  if not all(torch.isfinite(value).all() for value in payload["model"].values()):
    raise FloatingPointError("Nonfinite checkpoint parameters")
  model.load_state_dict(payload["model"])
  optimizer.load_state_dict(payload["optimizer"])
  sampler.load_state_dict(payload["sampler"])
  if (
    sampler.epoch != payload["epoch"]
    or payload["optimizer_steps"] != payload["epoch"] * 4
  ):
    raise ValueError("Resume sampler/update progress mismatch")
  for state in optimizer.state.values():
    for value in state.values():
      if torch.is_tensor(value) and not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite optimizer state")
  restore_rng(payload["rng"])


@torch.no_grad()
def validation_loss(model, scheduler, dataset, cfg):
  model.eval()
  losses = {}
  with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(90042)
    torch.cuda.manual_seed(90042)
    for condition, name in enumerate(CONDITION_NAMES):
      ids = (dataset.labels == condition).nonzero().flatten()
      total = 0.0
      for chunk in ids.split(256):
        total += float(
          _diffusion_loss(
            model,
            scheduler,
            dataset.windows[chunk].cuda(),
            cfg["num_noise_samples"],
            dataset.labels[chunk].cuda(),
          )
        ) * len(chunk)
      losses[name] = total / len(ids)
  losses["weighted_loss"] = (
    0.2 * losses["stand"]
    + 0.4 * losses["left_shoulder"]
    + 0.4 * losses["right_shoulder"]
  )
  if not all(math.isfinite(value) for value in losses.values()):
    raise FloatingPointError("Nonfinite validation loss")
  return losses


def train(output, *, smoke=False, resume=False):
  if (
    os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    or os.environ.get("SMP_PHYSICAL_GPU_UUID") != GPU_UUID
  ):
    raise RuntimeError("Require the approved physical GPU0, process cuda:0")
  torch.set_num_threads(2)
  seed_everything(42)
  cfg = {**FORMAL, "num_epochs": 1 if smoke else 10000, "smoke": smoke}
  output.mkdir(parents=True, exist_ok=True)
  latest = output / "latest.pt"
  if latest.exists() and not resume:
    raise RuntimeError(
      "Existing checkpoint; explicit resume required, never overwrite from scratch"
    )
  if resume and not latest.exists():
    raise RuntimeError("Resume requested without a complete checkpoint")
  if not resume and (output / "run.json").exists():
    raise RuntimeError("Existing run metadata; inspect interrupted startup first")
  train_data = Shoulder80Dataset(DATA_DIR, "train")
  validation = Shoulder80Dataset(DATA_DIR, "validation")
  sampler = Shoulder80Sampler(train_data.labels, train_data.phase_ids)
  loader = DataLoader(
    train_data, batch_size=1024, sampler=sampler, num_workers=0, pin_memory=True
  )
  model = DiffusionDenoiser(
    59, 10, d_model=128, nhead=4, num_layers=2, num_conditions=3
  ).cuda()
  model.condition_schema = CONDITION_SCHEMA
  scheduler = DDPMScheduler(50).cuda()
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
  completed, steps, evaluations, selected, wandb_id = 0, 0, [], None, None
  code_paths = [
    Path(__file__),
    Path("src/smp/pretrain/shoulder80_dataset.py"),
    Path("src/smp/pretrain/shoulder80_score.py"),
    Path("src/smp/pretrain/model.py"),
    Path("src/smp/pretrain/shoulder80.py"),
    Path("scripts/pretrain.py"),
  ]
  code_identity = {str(path.resolve()): sha256(path) for path in code_paths}
  restored = None
  if resume:
    restored = verified_checkpoint(latest)
    if restored["code_identity"] != code_identity:
      raise ValueError(
        "Training code changed since checkpoint; inspect before resuming"
      )
    restore_training(restored, model, optimizer, sampler, cfg, DATA_SHA)
    completed, steps = restored["epoch"], restored["optimizer_steps"]
    evaluations, selected, wandb_id = (
      restored["evaluations"],
      restored["selected"],
      restored["wandb_id"],
    )
  initial_model_hash = None
  if not resume:
    initial_path = output / "initial_state.pt"
    if initial_path.exists():
      raise RuntimeError("Refusing to overwrite initial model identity")
    torch.save(
      {"model": model.state_dict(), "seed": 42, "condition_schema": CONDITION_SCHEMA},
      initial_path,
    )
    initial_model_hash = sha256(initial_path)
  else:
    initial_model_hash = restored["initial_model_sha256"]
  training_rng = rng_state()
  run = None
  state = dict(
    experiment="EXP-029",
    status="starting",
    created_at_utc=now(),
    cfg=cfg,
    data_sha256=DATA_SHA,
    data_dir=str(DATA_DIR),
    condition_schema=CONDITION_SCHEMA,
    sampling=SAMPLING,
    initial_model_sha256=initial_model_hash,
    initialization="scratch",
    parent_prior=None,
    physical_gpu=0,
    gpu_uuid=GPU_UUID,
    completed_epochs=completed,
    optimizer_steps=steps,
    pid=os.getpid(),
    selection="score-gate first, then 0.2/0.4/0.4 validation L1 loss",
    ppo_authorized=False,
    prior_published=False,
    code_identity=code_identity,
    evaluations=evaluations,
    selected=selected,
    wandb_url=None,
  )
  previous_history = (
    json.loads((output / "run.json").read_text()).get("resume_history", [])
    if resume
    else []
  )
  state["resume_history"] = previous_history + (
    [{"at_utc": now(), "epoch": completed, "checkpoint_sha256": sha256(latest)}]
    if resume
    else []
  )
  stopped = False

  def on_signal(signum, _frame):
    nonlocal stopped
    stopped = True
    print(f"Signal {signum}: finishing current epoch and saving safely", flush=True)

  signal.signal(signal.SIGTERM, on_signal)
  signal.signal(signal.SIGINT, on_signal)

  def update_state(status):
    state.update(
      status=status,
      completed_epochs=completed,
      optimizer_steps=steps,
      updated_at_utc=now(),
      evaluations=evaluations,
      selected=selected,
    )
    atomic_json(output / "run.json", state)

  def save(path):
    atomic_checkpoint(
      path,
      dict(
        epoch=completed,
        optimizer_steps=steps,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        rng=rng_state(),
        sampler=sampler.state_dict(),
        cfg=cfg,
        condition_schema=CONDITION_SCHEMA,
        sampling=SAMPLING,
        data_sha256=DATA_SHA,
        data_manifest=str(DATA_DIR / "manifest.json"),
        q_low=train_data.q_low,
        q_high=train_data.q_high,
        normalization_sha256=train_data.manifest["norm_stats_sha256"],
        initialization="scratch",
        initial_model_sha256=initial_model_hash,
        code_identity=code_identity,
        wandb_id=wandb_id,
        evaluations=evaluations,
        selected=selected,
        gpu_uuid=GPU_UUID,
      ),
    )

  def evaluate_candidate():
    nonlocal selected
    candidate = output / f"checkpoint_{completed:05d}.pt"
    folder = output / "validation" / f"epoch_{completed:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    update_state("validation_running")
    with torch.random.fork_rng(devices=[0]):
      model.eval()
      losses = validation_loss(model, scheduler, validation, cfg)
      means = calibrate(model, scheduler, train_data)
      calibration = dict(
        prior_sha256=sha256(candidate),
        data_manifest_sha256=DATA_SHA,
        normalization_sha256=train_data.manifest["norm_stats_sha256"],
        score_mode=SCORE_MODE,
        condition_schema=CONDITION_SCHEMA,
        fixed_timesteps=list(FIXED_TIMESTEPS),
        feature_weights=score_feature_weights(),
        samples_per_condition=4096,
        seed=42,
        reference_mean=means,
      )
      calibration_path = folder / "calibration.pt"
      if calibration_path.exists():
        # A failed/incomplete evaluation may be rerun on the identical locked candidate.
        old = torch.load(calibration_path, weights_only=True)
        if old["prior_sha256"] != calibration["prior_sha256"] or not torch.equal(
          old["reference_mean"], means
        ):
          raise ValueError("Interrupted candidate calibration identity mismatch")
      else:
        torch.save(calibration, calibration_path)
      report, raw = evaluate_scores(model, scheduler, validation, means)
      report.update(
        losses=losses,
        epoch=completed,
        prior_sha256=sha256(candidate),
        calibration_sha256=sha256(calibration_path),
        data_sha256=DATA_SHA,
      )
      np.savez_compressed(folder / "paired_scores.npz", **raw)
      atomic_json(folder / "report.json", report)
    item = dict(
      epoch=completed,
      checkpoint=str(candidate),
      checkpoint_sha256=sha256(candidate),
      calibration=str(calibration_path),
      calibration_sha256=sha256(calibration_path),
      report=str(folder / "report.json"),
      passed=report["gate"]["passed"],
      weighted_validation_loss=losses["weighted_loss"],
    )
    evaluations.append(item)
    if item["passed"] and (
      selected is None
      or item["weighted_validation_loss"] < selected["weighted_validation_loss"]
    ):
      selected = item
    print(
      json.dumps(
        {"validation": item, "failed_checks": report["gate"]["failed_checks"]}
      ),
      flush=True,
    )
    if run:
      run.log(
        {
          "epoch": completed,
          "score_gate/passed": int(item["passed"]),
          **{f"validation/{k}": v for k, v in losses.items()},
          **{f"score_gate/{k}": int(v) for k, v in report["gate"]["checks"].items()},
        }
      )
    save(latest)
    update_state("training")

  try:
    if not smoke:
      import wandb

      run = wandb.init(
        project="smp",
        name="conditional_shoulder80_expanded_prior_v1_seed42_gpu0",
        id=wandb_id,
        resume="must" if wandb_id else None,
        mode="online",
        config={
          **cfg,
          "condition_schema": CONDITION_SCHEMA,
          "sampling": SAMPLING,
          "data_sha256": DATA_SHA,
          "gpu_uuid": GPU_UUID,
          "scratch_prior": True,
          "upload_model": False,
        },
      )
      wandb_id = run.id
      state["wandb_url"] = run.url
      # Network/log initialization must not perturb scratch or restored RNG.
      restore_rng(training_rng)
    update_state("training")
    if not resume:
      save(latest)  # Epoch 0 is recoverable even if startup is interrupted.
    if (
      completed
      and completed % 1000 == 0
      and not any(item["epoch"] == completed for item in evaluations)
    ):
      evaluate_candidate()
    started = time.monotonic()
    counts = torch.tensor(SAMPLING["per_condition"])
    for epoch in range(completed, cfg["num_epochs"]):
      model.train()
      total, seen = 0.0, 0
      counts = torch.zeros(3, dtype=torch.long)
      for x, labels in loader:
        counts += torch.bincount(labels, minlength=3)
        x, labels = x.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        conditions = labels.clone()
        conditions[torch.rand(len(labels), device="cuda:0") < 0.10] = 3
        optimizer.zero_grad(set_to_none=True)
        loss = _diffusion_loss(model, scheduler, x, 10, conditions)
        if not torch.isfinite(loss):
          raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
          model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
          raise FloatingPointError("Nonfinite trained parameters")
        steps += 1
        total += float(loss.detach()) * len(x)
        seen += len(x)
      completed = epoch + 1
      if counts.tolist() != [672, 1344, 1344] or sampler.epoch != completed:
        raise RuntimeError("Sampling quota/progress mismatch")
      if completed == 1 or completed % 25 == 0 or stopped:
        stats = dict(
          epoch=completed,
          train_loss=total / seen,
          grad_norm=float(grad),
          optimizer_steps=steps,
          elapsed_seconds=time.monotonic() - started,
        )
        print(json.dumps(stats), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
          stream.write(json.dumps(stats) + "\n")
        if run:
          run.log(stats)
        state["latest_metrics"] = stats
        update_state("training")
      if completed == 1 or completed % 100 == 0 or stopped:
        save(latest)
      if completed % 1000 == 0 or smoke:
        save(output / f"checkpoint_{completed:05d}.pt")
      if stopped:
        update_state("interrupted_recoverable")
        raise InterruptedError(f"Safely stopped at epoch {completed}")
      if completed % 1000 == 0:
        evaluate_candidate()
    if smoke:
      model.eval()
      with torch.random.fork_rng(devices=[0]), torch.no_grad():
        means = calibrate(model, scheduler, train_data, samples=16, batch_size=16)
        score = paired_scores(
          model, scheduler, validation.windows[:4], 1, means, repeats=2
        )
        bundle = (
          model,
          scheduler,
          torch.from_numpy(train_data.q_low).cuda(),
          torch.from_numpy(train_data.q_high).cuda(),
          59,
          10,
        )
        samples = [sample_condition(bundle, c, 4) for c in (0, 1, 2, "null")]
        smoke_report = dict(
          finite_loss=True,
          finite_score=bool(torch.isfinite(score["reward"]).all()),
          finite_ddpm=all(bool(torch.isfinite(x).all()) for x in samples),
          sampling_counts=counts.tolist(),
          optimizer_steps=steps,
          schema=CONDITION_SCHEMA,
          acceptance_not_evaluated=True,
        )
        atomic_json(output / "smoke_report.json", smoke_report)
      update_state("smoke_passed")
    else:
      save(output / "final_training_state.pt")
      atomic_json(
        output / "selection.json",
        {
          "selected": selected,
          "evaluations": evaluations,
          "test_used_for_selection": False,
          "budget_epochs": 10000,
          "status": "candidate_locked"
          if selected
          else "failed_no_score_passing_candidate",
        },
      )
      update_state(
        "candidate_locked_pending_test"
        if selected
        else "failed_no_score_passing_candidate"
      )
  except BaseException as error:
    state["failure"] = {
      "type": type(error).__name__,
      "message": str(error),
      "at_utc": now(),
    }
    update_state(
      "interrupted_recoverable"
      if isinstance(error, InterruptedError)
      else "failed_preserved_last_checkpoint"
    )
    raise
  finally:
    if run:
      run.finish(
        exit_code=0
        if state["status"]
        in ("candidate_locked_pending_test", "failed_no_score_passing_candidate")
        else 1
      )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  train(args.output, smoke=args.smoke, resume=args.resume)
