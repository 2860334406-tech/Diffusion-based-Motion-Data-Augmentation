"""Diffusion model pretraining loop."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Sampler, Subset, random_split

from smp.pretrain.dataset import MotionWindowDataset
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.pretrain_cfg import PretrainCfg
from smp.pretrain.scheduler import DDPMScheduler
from smp.utils import count_parameters, seed_everything


class _Ema:
  """Exponential moving average shadow of a model.

  Standard formula: ``θ_ema ← decay·θ_ema + (1−decay)·θ``, applied in-place
  over every entry of ``state_dict()`` (covers params and buffers).
  """

  def __init__(self, model: torch.nn.Module, decay: float) -> None:
    self.decay = decay
    self.shadow = copy.deepcopy(model)
    self.shadow.eval()
    for p in self.shadow.parameters():
      p.requires_grad_(False)

  @torch.no_grad()
  def update(self, model: torch.nn.Module) -> None:
    src = model.state_dict()
    dst = self.shadow.state_dict()
    for k, v_src in src.items():
      v_dst = dst[k]
      if v_dst.is_floating_point():
        v_dst.mul_(self.decay).add_(v_src.detach(), alpha=1.0 - self.decay)
      else:
        v_dst.copy_(v_src)


class ExactCategorySampler(Sampler[int]):
  """Draw an exact, reproducible number of windows from each category."""

  def __init__(
    self,
    dataset: MotionWindowDataset,
    manifest_file: str | Path,
    seed: int,
  ) -> None:
    manifest_path = Path(manifest_file)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
      raise ValueError("Sampling manifest schema_version must be 1")
    self.samples_per_epoch = int(payload["samples_per_epoch"])
    if self.samples_per_epoch <= 0:
      raise ValueError("samples_per_epoch must be positive")

    fractions = payload["category_fractions"]
    if not fractions or any(float(value) <= 0.0 for value in fractions.values()):
      raise ValueError("category_fractions must contain positive values")
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1.0e-9:
      raise ValueError("category_fractions must sum to 1")

    files = payload["files"]
    if set(files) != set(dataset.file_slices):
      missing = sorted(set(dataset.file_slices).difference(files))
      extra = sorted(set(files).difference(dataset.file_slices))
      raise ValueError(
        f"Sampling manifest file mismatch: missing={missing}, extra={extra}"
      )

    pools: dict[str, list[int]] = {category: [] for category in fractions}
    for filename, category in files.items():
      if category not in pools:
        raise ValueError(f"Unknown category {category!r} for {filename}")
      file_slice = dataset.file_slices[filename]
      pools[category].extend(range(file_slice.start, file_slice.stop))
    if any(not pool for pool in pools.values()):
      raise ValueError("Every sampling category must contain at least one window")

    categories = list(fractions)
    counts: dict[str, int] = {}
    assigned = 0
    for category in categories[:-1]:
      count = round(self.samples_per_epoch * float(fractions[category]))
      counts[category] = count
      assigned += count
    counts[categories[-1]] = self.samples_per_epoch - assigned
    if any(count <= 0 for count in counts.values()):
      raise ValueError("Every category must receive at least one sample per epoch")

    self.category_fractions = {
      category: float(fractions[category]) for category in categories
    }
    self.category_counts = counts
    self.category_pools = {
      category: torch.tensor(pool, dtype=torch.long) for category, pool in pools.items()
    }
    self.seed = seed
    self.epoch = 0
    self.manifest_path = manifest_path
    self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

  def __len__(self) -> int:
    return self.samples_per_epoch

  def __iter__(self) -> Iterator[int]:
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    sampled: list[torch.Tensor] = []
    for category, count in self.category_counts.items():
      pool = self.category_pools[category]
      positions = torch.randint(len(pool), (count,), generator=generator)
      sampled.append(pool[positions])
    indices = torch.cat(sampled)
    indices = indices[torch.randperm(len(indices), generator=generator)]
    self.epoch += 1
    return iter(indices.tolist())

  def metadata(self) -> dict[str, Any]:
    return {
      "manifest": str(self.manifest_path),
      "manifest_sha256": self.manifest_sha256,
      "samples_per_epoch": self.samples_per_epoch,
      "category_fractions": self.category_fractions,
      "category_counts": self.category_counts,
      "seed": self.seed,
    }


def _maybe_upload_checkpoint(
  wandb_run: Any | None,
  path: Path,
  base_path: Path,
  upload_model: bool,
) -> None:
  """Upload a checkpoint only when artifact upload was explicitly enabled."""
  if wandb_run is not None and upload_model:
    wandb_run.save(str(path), base_path=str(base_path))


def _diffusion_loss(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  x_0: torch.Tensor,
  num_noise_samples: int,
  condition_id: torch.Tensor | None = None,
) -> torch.Tensor:
  """DDPM ε-prediction L1 loss with multiple noise samples per data point.

  Each sample in the batch is paired with ``num_noise_samples`` random
  (timestep, noise) draws, giving lower-variance gradients than a single
  draw without the cost of exhausting all T timesteps.
  """
  B = x_0.shape[0]
  K = num_noise_samples
  # (B, W, F) → (B*K, W, F)
  x_0_exp = x_0[:, None].expand(B, K, *x_0.shape[1:]).reshape(B * K, *x_0.shape[1:])
  t = scheduler.sample_timesteps(B * K, x_0.device)
  noise = torch.randn_like(x_0_exp)
  x_t = scheduler.add_noise(x_0_exp, noise, t)
  predicted = (
    model(x_t, t) if condition_id is None
    else model(x_t, t, condition_id.repeat_interleave(K))
  )
  return F.l1_loss(predicted, noise)


def _save_checkpoint(
  path: Path,
  epoch: int,
  model: DiffusionDenoiser,
  dataset: MotionWindowDataset,
  feature_dim: int,
  cfg: PretrainCfg,
  optimizer: torch.optim.Optimizer | None = None,
  ema: _Ema | None = None,
  sampling_metadata: dict[str, Any] | None = None,
) -> None:
  data: dict[str, Any] = {
    "epoch": epoch,
    "model": model.state_dict(),
    "q_low": dataset.q_low,
    "q_high": dataset.q_high,
    "cfg": {
      **vars(cfg),
      "feature_dim": feature_dim,
      "window_size": dataset.window_size,
    },
  }
  if optimizer is not None:
    data["optimizer"] = optimizer.state_dict()
  if ema is not None:
    data["model_ema"] = ema.shadow.state_dict()
  if sampling_metadata is not None:
    data["sampling"] = sampling_metadata
  torch.save(data, path)


def pretrain(cfg: PretrainCfg) -> Path:
  """Run diffusion pretraining."""
  seed_everything(cfg.seed)
  print(f"[INFO] seed={cfg.seed}")
  device = torch.device(cfg.device)

  dataset = MotionWindowDataset(cfg.data_dir, norm_stats_file=cfg.norm_stats_file)
  feature_dim = dataset.feature_dim
  window_size = dataset.window_size

  n_train = int(len(dataset) * cfg.train_split)
  n_val = len(dataset) - n_train
  print(
    f"Dataset: {len(dataset)} windows, n_train={n_train}, n_val={n_val}, "
    f"feature_dim={feature_dim}, window_size={window_size}"
  )

  sampler: ExactCategorySampler | None = None
  if cfg.sampling_manifest_file:
    if cfg.train_split != 1.0:
      raise ValueError("Exact category sampling requires train_split=1.0")
    train_set = dataset
    val_set = Subset(dataset, [])
    sampler = ExactCategorySampler(dataset, cfg.sampling_manifest_file, cfg.seed)
    sampling_metadata = sampler.metadata()
    print(f"Exact category sampling: {json.dumps(sampling_metadata, sort_keys=True)}")
  else:
    train_set, val_set = random_split(dataset, [n_train, n_val])
    sampling_metadata = None
  pin_memory = device.type == "cuda"
  train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=sampler is None,
    sampler=sampler,
    pin_memory=pin_memory,
  )
  val_loader = DataLoader(
    val_set, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory
  )

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout,
  ).to(device)
  scheduler = DDPMScheduler(
    num_timesteps=cfg.num_timesteps,
  ).to(device)
  print(f"Denoiser: {count_parameters(model):,} params")

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )

  ema = _Ema(model, decay=cfg.ema_decay) if cfg.use_ema else None
  if ema is not None:
    print(f"EMA enabled (decay={cfg.ema_decay})")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir = Path(cfg.log_dir) / cfg.name / timestamp
  save_dir.mkdir(parents=True, exist_ok=True)

  wandb_run = None
  if cfg.use_wandb:
    import wandb

    wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.name, config=vars(cfg))

    if sampling_metadata is not None:
      wandb_run.config.update({"sampling": sampling_metadata})

  for epoch in range(cfg.num_epochs):
    model.train()
    epoch_loss = torch.zeros((), device=device)
    n_batches = 0

    for batch in train_loader:
      x_0 = batch.to(device, non_blocking=pin_memory)
      loss = _diffusion_loss(model, scheduler, x_0, cfg.num_noise_samples)

      optimizer.zero_grad()
      loss.backward()
      if cfg.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
      optimizer.step()
      if ema is not None:
        ema.update(model)

      epoch_loss += loss.detach()
      n_batches += 1

    avg_loss = (epoch_loss / max(n_batches, 1)).item()

    if epoch % cfg.log_interval == 0:
      eval_model = ema.shadow if ema is not None else model
      val_loss = _validate(
        eval_model, scheduler, val_loader, device, pin_memory, cfg.num_noise_samples
      )
      print(f"Epoch {epoch:4d} | train={avg_loss:.6f} | val={val_loss:.6f}")
      if wandb_run is not None:
        wandb_run.log({"epoch": epoch, "train/loss": avg_loss, "val/loss": val_loss})

    if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1:
      ckpt_path = save_dir / f"checkpoint_{epoch:05d}.pt"
      _save_checkpoint(
        ckpt_path,
        epoch,
        model,
        dataset,
        feature_dim,
        cfg,
        optimizer,
        ema,
        sampling_metadata,
      )
      _maybe_upload_checkpoint(wandb_run, ckpt_path, save_dir, cfg.upload_model)

  final_path = save_dir / "pretrained.pt"
  _save_checkpoint(
    final_path,
    cfg.num_epochs,
    model,
    dataset,
    feature_dim,
    cfg,
    ema=ema,
    sampling_metadata=sampling_metadata,
  )
  print(f"Saved final checkpoint to {final_path}")

  if wandb_run is not None:
    _maybe_upload_checkpoint(wandb_run, final_path, save_dir, cfg.upload_model)
    wandb_run.finish()

  return final_path


@torch.no_grad()
def _validate(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  val_loader: DataLoader[torch.Tensor],
  device: torch.device,
  pin_memory: bool,
  num_noise_samples: int,
) -> float:
  model.eval()
  total = torch.zeros((), device=device)
  n = 0
  for batch in val_loader:
    x_0 = batch.to(device, non_blocking=pin_memory)
    total += _diffusion_loss(model, scheduler, x_0, num_noise_samples)
    n += 1
  return (total / max(n, 1)).item()


if __name__ == "__main__":
  pretrain(tyro.cli(PretrainCfg))
