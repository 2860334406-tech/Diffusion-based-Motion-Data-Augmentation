"""Train the trajectory-level conditional diffusion model for waist augmentation.

Differs from GTA in one respect: the condition is an 18-dim attribute vector
computed from the trajectory (amplitude, entry speed, arm/knee posture, ...)
rather than a discounted return. Everything else follows the paper: epsilon
prediction, condition dropout for classifier-free guidance, and a frozen
normaliser stored in the checkpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader, Subset

from smp.data.waist_aug_specs import HORIZON_FRAMES
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.pretrain.waist_aug_dataset import (
  COND_DIM,
  FEATURE_DIM,
  Clip,
  WaistAugDataset,
  load_csv_clip,
  load_npz_clip,
  normalise,
)


@dataclass
class Cfg:
  anchor_dir: str = "logs/reproduction/waist_aug_v1/anchors"
  real_csv_dir: str = "data/real"
  out_dir: str = "logs/reproduction/waist_aug_v1/prior"
  window: int = HORIZON_FRAMES
  stride: int = 32
  d_model: int = 256
  nhead: int = 4
  num_layers: int = 6
  dropout: float = 0.1
  num_timesteps: int = 100
  num_noise_samples: int = 4
  x0_loss_weight: float = 0.5
  condition_dropout: float = 0.1
  batch_size: int = 256
  num_epochs: int = 3000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0
  log_interval: int = 50
  save_interval: int = 500
  val_fraction: float = 0.1
  seed: int = 42
  device: str = ""
  # Oversample real captures so the model learns human style instead of being
  # dominated by the 1000+ procedural anchors (they are ~99% of the data).
  real_weight: float = 10.0


def _now() -> str:
  return datetime.now(timezone.utc).isoformat()


def load_anchors(anchor_dir: str | Path) -> list[Clip]:
  root = Path(anchor_dir) / "clips" / "accepted"
  if not root.exists():
    raise FileNotFoundError(f"No accepted anchors under {root}")
  paths = sorted(root.rglob("*.npz"))
  paths = [p for p in paths if not p.name.endswith(".reference.npz")]
  if not paths:
    raise FileNotFoundError(f"No execution npz under {root}")
  return [load_npz_clip(p) for p in paths]


def load_real(csv_dir: str | Path) -> list[Clip]:
  paths = sorted(Path(csv_dir).glob("P0*.csv"))
  if not paths:
    raise FileNotFoundError(f"No P0*.csv under {csv_dir}")
  return [load_csv_clip(p, fps=30.0) for p in paths]


def split_indices(ds: WaistAugDataset, val_fraction: float, seed: int):
  """Real clips are split by subject; anchors are split over clips as a pool."""
  rng = np.random.default_rng(seed)
  anchor_idx: list[int] = []
  train: list[int] = []
  val: list[int] = []
  for i, src in enumerate(ds.sources):
    name = Path(src).name
    if name.startswith("P0") and name.endswith(".csv"):
      subject = name.split("_")[0]
      (val if subject == "P03" else train).append(i)
    else:
      anchor_idx.append(i)

  anchor_idx = np.array(anchor_idx)
  rng.shuffle(anchor_idx)
  cut = max(1, int(len(anchor_idx) * val_fraction))
  val.extend(anchor_idx[:cut].tolist())
  train.extend(anchor_idx[cut:].tolist())
  return sorted(train), sorted(val)


def diffusion_loss(model, scheduler, x0, cond, cfg: Cfg, device):
  b, w, f = x0.shape
  k = cfg.num_noise_samples
  x0r = x0.repeat_interleave(k, dim=0)
  cr = cond.repeat_interleave(k, dim=0)
  t = scheduler.sample_timesteps(x0r.shape[0], device)
  noise = torch.randn_like(x0r)
  x_t = scheduler.add_noise(x0r, noise, t)
  drop = torch.rand(x0r.shape[0], device=device) < cfg.condition_dropout
  # The null branch must go through ``condition=None`` so that the dedicated
  # ``null_condition`` embedding is actually trained and matches sampling time.
  if drop.any() and not drop.all():
    eps = torch.empty_like(x0r)
    keep = ~drop
    eps[keep] = model(x_t[keep], t[keep], condition=cr[keep])
    eps[drop] = model(x_t[drop], t[drop], condition=None)
  elif drop.all():
    eps = model(x_t, t, condition=None)
  else:
    eps = model(x_t, t, condition=cr)
  abar = scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1)
  x0_hat = (x_t - torch.sqrt(1 - abar) * eps) / torch.sqrt(abar)
  loss = torch.nn.functional.l1_loss(eps, noise)
  loss = loss + cfg.x0_loss_weight * torch.nn.functional.l1_loss(x0_hat, x0r)
  return loss


def main(cfg: Cfg) -> None:
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

  anchors = load_anchors(cfg.anchor_dir)
  real = load_real(cfg.real_csv_dir)
  print(json.dumps({"anchors": len(anchors), "real": len(real)}), flush=True)

  ds = WaistAugDataset(anchors + real, window=cfg.window, stride=cfg.stride)
  lo_np, hi_np = ds.compute_norm_stats()
  lo = torch.from_numpy(lo_np)
  hi = torch.from_numpy(hi_np)
  train_idx, val_idx = split_indices(ds, cfg.val_fraction, cfg.seed)
  print(json.dumps({
    "windows": len(ds), "train": len(train_idx), "val": len(val_idx),
  }), flush=True)

  x_all = torch.from_numpy(ds.windows.numpy())
  c_all = torch.from_numpy(ds.conditions.numpy())

  class _View(torch.utils.data.Dataset):
    def __init__(self, idx):
      self.idx = idx

    def __len__(self):
      return len(self.idx)

    def __getitem__(self, i):
      j = self.idx[i]
      return normalise(x_all[j], lo, hi), c_all[j]

  sample_w = torch.ones(len(train_idx), dtype=torch.double)
  for k, j in enumerate(train_idx):
    name = Path(ds.sources[j]).name
    if name.startswith("P0") and name.endswith(".csv"):
      sample_w[k] = cfg.real_weight
  n_real = int((sample_w > 1).sum())
  print(json.dumps({"real_train_windows": n_real,
                    "real_weight": cfg.real_weight}), flush=True)
  sampler = torch.utils.data.WeightedRandomSampler(
    sample_w, num_samples=len(train_idx), replacement=True,
    generator=torch.Generator().manual_seed(cfg.seed),
  )
  train_loader = DataLoader(
    _View(train_idx), batch_size=cfg.batch_size, sampler=sampler, drop_last=True
  )
  val_loader = DataLoader(_View(val_idx), batch_size=cfg.batch_size, shuffle=False)

  model = DiffusionDenoiser(
    feature_dim=FEATURE_DIM,
    window_size=cfg.window,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout,
    condition_dim=COND_DIM,
  ).to(device)
  scheduler = DDPMScheduler(num_timesteps=cfg.num_timesteps).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

  out = Path(cfg.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
  out.mkdir(parents=True, exist_ok=True)
  (out / "config.json").write_text(json.dumps({**asdict(cfg), "started_at_utc": _now()}, indent=1))

  def save(path: Path, epoch: int):
    torch.save({
      "epoch": epoch,
      "model": model.state_dict(),
      "q_low": lo_np,
      "q_high": hi_np,
      "cfg": {**asdict(cfg), "feature_dim": FEATURE_DIM, "condition_dim": COND_DIM,
              "num_conditions": 0},
    }, path)

  step = 0
  start = time.time()
  for epoch in range(1, cfg.num_epochs + 1):
    model.train()
    running = 0.0
    for x, c in train_loader:
      x, c = x.to(device), c.to(device)
      loss = diffusion_loss(model, scheduler, x, c, cfg, device)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
      opt.step()
      running += float(loss.detach())
      step += 1

    if epoch % cfg.log_interval == 0 or epoch == 1:
      model.eval()
      val = 0.0
      n = 0
      with torch.no_grad():
        for x, c in val_loader:
          x, c = x.to(device), c.to(device)
          val += float(diffusion_loss(model, scheduler, x, c, cfg, device))
          n += 1
      print(json.dumps({
        "epoch": epoch, "step": step,
        "train": round(running / max(1, len(train_loader)), 5),
        "val": round(val / max(1, n), 5),
        "elapsed_s": round(time.time() - start, 1),
      }), flush=True)

    if epoch % cfg.save_interval == 0:
      save(out / f"checkpoint_{epoch:05d}.pt", epoch)

  save(out / "pretrained.pt", cfg.num_epochs)
  print(json.dumps({"done": str(out / "pretrained.pt"), "elapsed_s": round(time.time() - start, 1)}))


if __name__ == "__main__":
  tyro.cli(main)