"""GPU1 conditional prior training with exact balance and restartable checkpoints."""
from __future__ import annotations

import json
import math
import os
import random
import signal
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from scripts.pretrain import _diffusion_loss
from torch.utils.data import DataLoader

from smp.pretrain.conditional_dataset import ConditionalMotionDataset
from smp.pretrain.conditioning import (
  CONDITION_NAMES,
  CONDITION_SCHEMA,
  NULL_CONDITION,
  BalancedConditionSampler,
  ShoulderPhaseBalancedSampler,
)
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.utils import seed_everything


@dataclass
class Cfg:
  data_dir: str = "logs/reproduction/conditional_contact_v1/data"
  output_dir: str = "logs/reproduction/conditional_contact_v1/prior"
  num_epochs: int = 10_000
  batch_size: int = 1024
  d_model: int = 128
  nhead: int = 4
  num_layers: int = 2
  num_timesteps: int = 50
  num_noise_samples: int = 10
  lr: float = 3e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0
  condition_dropout: float = .10
  samples_per_epoch: int = 3360
  seed: int = 42
  log_interval: int = 25
  checkpoint_interval: int = 100
  save_interval: int = 1000
  use_wandb: bool = True
  wandb_project: str = "smp"
  name: str = "conditional_contact_v1_prior_seed42_gpu1"
  device: str = "cuda:0"
  resume: bool = True


def atomic_save(value, path: Path):
  temporary = path.with_suffix(path.suffix + ".tmp")
  torch.save(value, temporary)
  temporary.replace(path)


def train(cfg: Cfg):
  if cfg.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise RuntimeError("This experiment requires CUDA_VISIBLE_DEVICES=1 before Python starts")
  torch.set_num_threads(2)
  seed_everything(cfg.seed)
  output = Path(cfg.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  dataset = ConditionalMotionDataset(cfg.data_dir, "train")
  validation = ConditionalMotionDataset(cfg.data_dir, "validation")
  sampler = (
    BalancedConditionSampler(dataset.labels, cfg.seed, cfg.samples_per_epoch)
    if dataset.phase_ids is None else
    ShoulderPhaseBalancedSampler(dataset.labels, dataset.phase_ids, cfg.seed, cfg.samples_per_epoch)
  )
  loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=0,
                      pin_memory=cfg.device.startswith("cuda"))
  model = DiffusionDenoiser(59, 10, d_model=cfg.d_model, nhead=cfg.nhead,
                           num_layers=cfg.num_layers, num_conditions=5).to(cfg.device)
  scheduler = DDPMScheduler(cfg.num_timesteps).to(cfg.device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
  completed, best_loss, wandb_id = 0, float("inf"), None
  latest = output / "latest.pt"
  if latest.exists():
    if not cfg.resume:
      raise RuntimeError("Refusing to overwrite an existing training run")
    ckpt = torch.load(latest, map_location=cfg.device, weights_only=False)
    expected = {**asdict(cfg), "feature_dim": 59, "window_size": 10, "num_conditions": 5}
    if ckpt["cfg"] != expected or ckpt["data_sha256"] != dataset.manifest_sha256:
      raise RuntimeError("Resume configuration/data checksum mismatch")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    completed, best_loss, wandb_id = ckpt["epoch"], ckpt["best_loss"], ckpt["wandb_id"]
    random.setstate(ckpt["rng"]["python"])
    np.random.set_state(ckpt["rng"]["numpy"])
    torch.set_rng_state(ckpt["rng"]["torch"].cpu())
    if cfg.device.startswith("cuda"):
      torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["rng"]["cuda"]])
    sampler.epoch = ckpt["sampler_epoch"]
    if sampler.epoch != completed:
      raise RuntimeError("Sampler/checkpoint epoch mismatch")
  stopped = False

  def on_signal(signum, _frame):
    nonlocal stopped
    stopped = True
    print(f"Received signal {signum}; saving after current epoch", flush=True)

  signal.signal(signal.SIGTERM, on_signal)
  signal.signal(signal.SIGINT, on_signal)
  run = None
  if cfg.use_wandb:
    import wandb
    run = wandb.init(project=cfg.wandb_project, name=cfg.name, id=wandb_id,
                     resume="allow" if wandb_id else None,
                     config={**asdict(cfg), "num_conditions": 5, "condition_schema": CONDITION_SCHEMA,
                             "data_sha256": dataset.manifest_sha256, "sampling": dataset.sampling,
                             "upload_model": False, "physical_gpu": 1, "scratch_prior": True})
    wandb_id = run.id
  metadata = {
    "config": asdict(cfg), "data_sha256": dataset.manifest_sha256,
    "conditions": CONDITION_SCHEMA, "sampling": dataset.sampling,
    "wandb_url": run.url if run else None,
    "physical_gpu": 1, "gpu_uuid": os.environ.get("SMP_PHYSICAL_GPU_UUID"),
    "initialization": "scratch",
  }
  (output / "run.json").write_text(json.dumps(metadata, indent=2)+"\n")

  def save(epoch, path):
    atomic_save({
      "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
      "q_low": dataset.q_low, "q_high": dataset.q_high,
      "cfg": {**asdict(cfg), "feature_dim": 59, "window_size": 10, "num_conditions": 5},
      "condition_schema": CONDITION_SCHEMA, "data_sha256": dataset.manifest_sha256,
      "sampling": dataset.sampling,
      "best_loss": best_loss, "wandb_id": wandb_id, "sampler_epoch": sampler.epoch,
      "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state(),
              "cuda": torch.cuda.get_rng_state_all() if cfg.device.startswith("cuda") else []},
    }, path)

  def validate(epoch):
    # Fixed noise across validation passes; validation never perturbs training RNG.
    devices = [0] if cfg.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(cfg.seed + 90_001)
      if devices:
        torch.cuda.manual_seed(cfg.seed + 90_001)
      model.eval()
      losses = {}
      with torch.no_grad():
        for condition, name in enumerate(CONDITION_NAMES):
          ids = (validation.labels == condition).nonzero().flatten()[:512]
          x = validation.windows[ids].to(cfg.device)
          c = validation.labels[ids].to(cfg.device)
          loss = _diffusion_loss(model, scheduler, x, cfg.num_noise_samples, c)
          losses[f"validation/{name}"] = float(loss)
      losses["validation/macro_loss"] = sum(losses.values()) / 5
      return losses

  print(f"Scratch={completed == 0}; completed_epochs={completed}; "
        f"train={len(dataset)}, validation={len(validation)}, "
        f"per_epoch={[cfg.samples_per_epoch//5]*5}; sampling={dataset.sampling}", flush=True)
  try:
    for epoch in range(completed, cfg.num_epochs):
      model.train()
      total, seen = 0.0, 0
      for x, conditions in loader:
        x, conditions = x.to(cfg.device), conditions.to(cfg.device)
        conditions = conditions.clone()
        conditions[torch.rand(len(conditions), device=cfg.device) < cfg.condition_dropout] = NULL_CONDITION
        loss = _diffusion_loss(model, scheduler, x, cfg.num_noise_samples, conditions)
        if not torch.isfinite(loss):
          raise FloatingPointError("Nonfinite conditional diffusion loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm,
                                              error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
          raise FloatingPointError("Nonfinite conditional model parameters")
        total += float(loss.detach()) * len(x)
        seen += len(x)
      completed = epoch + 1
      stats = {"epoch": completed, "train/loss": total / seen, "train/grad_norm": float(norm)}
      if completed == 1 or completed % cfg.log_interval == 0 or completed == cfg.num_epochs:
        stats.update(validate(completed))
        macro = stats["validation/macro_loss"]
        if not math.isfinite(macro):
          raise FloatingPointError("Nonfinite validation loss")
        if macro < best_loss:
          best_loss = macro
          save(completed, output / "best.pt")
        print(json.dumps(stats), flush=True)
        if run:
          run.log(stats, step=completed)
      if completed == 1 or completed % cfg.checkpoint_interval == 0 or stopped or completed == cfg.num_epochs:
        save(completed, latest)
      if completed % cfg.save_interval == 0 or completed == cfg.num_epochs:
        save(completed, output / f"checkpoint_{completed:05d}.pt")
      if stopped:
        raise InterruptedError(f"Training interrupted safely at epoch {completed}")
    save(completed, output / "pretrained.pt")
    metadata.update(status="complete", completed_epochs=completed, best_loss=best_loss)
    (output / "run.json").write_text(json.dumps(metadata, indent=2)+"\n")
  finally:
    if run:
      run.finish()


if __name__ == "__main__":
  train(tyro.cli(Cfg))
