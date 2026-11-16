"""Validate a low-speed SMP prior and compare it with the locomotion prior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv

from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)
from smp.rl.utils import load_denoiser
from smp.utils import detect_device, seed_everything


@dataclass
class Cfg:
  ckpt_path: str = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_balance.pt"
  reference_ckpt_path: str = "datasets/pretrain_ckpt/pretrained_loco.pt"
  output: str = "logs/reproduction/data/low_speed_prior_validation.json"
  num_samples: int = 4096
  batch_size: int = 256
  seed: int = 42
  device: str = ""
  check_gsi: bool = True
  gsi_num_envs: int = 64
  gsi_buffer_size: int = 1024
  max_root_planar_speed: float = 0.5
  min_gsi_acceptance: float = 0.5
  min_p95_reduction: float = 0.5
  enforce_thresholds: bool = True


@torch.no_grad()
def sample_speed_stats(
  ckpt_path: str,
  num_samples: int,
  batch_size: int,
  device: str,
) -> dict[str, float | int | bool]:
  model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(
    ckpt_path, device
  )
  speeds: list[torch.Tensor] = []
  finite = True
  for start in range(0, num_samples, batch_size):
    count = min(batch_size, num_samples - start)
    x_t = torch.randn(count, window_size, feature_dim, device=device)
    for timestep in reversed(range(scheduler.num_timesteps)):
      t = torch.full((count,), timestep, dtype=torch.long, device=device)
      eps = model(x_t, t)
      finite = finite and bool(torch.isfinite(eps).all())
      x_t = scheduler.step(eps, x_t, timestep)
    windows = (x_t + 1.0) * 0.5 * (q_high - q_low) + q_low
    finite = finite and bool(torch.isfinite(windows).all())
    speeds.append(torch.linalg.vector_norm(windows[:, -1, 53:55], dim=-1).cpu())
  speed = torch.cat(speeds)
  quantiles = torch.quantile(speed, torch.tensor([0.5, 0.9, 0.95, 0.99]))
  return {
    "samples": num_samples,
    "finite": finite,
    "speed_mean_m_s": float(speed.mean()),
    "speed_p50_m_s": float(quantiles[0]),
    "speed_p90_m_s": float(quantiles[1]),
    "speed_p95_m_s": float(quantiles[2]),
    "speed_p99_m_s": float(quantiles[3]),
    "speed_max_m_s": float(speed.max()),
  }


def gsi_stats(cfg: Cfg, device: str) -> dict[str, int | float]:
  env_cfg = g1_force_unloading_smp_env_cfg()
  env_cfg.scene.num_envs = cfg.gsi_num_envs
  params = env_cfg.events["init_smp_state"].params
  params.update(
    {
      "ckpt_path": cfg.ckpt_path,
      "gsi_buffer_size": cfg.gsi_buffer_size,
      "gsi_batch_size": min(cfg.batch_size, cfg.gsi_buffer_size),
      "compile_model": False,
      "gsi_max_root_planar_speed": cfg.max_root_planar_speed,
      "gsi_max_sampling_multiplier": 20,
    }
  )
  env_cfg.events["gsi_refresh"].params["num_samples"] = min(64, cfg.gsi_buffer_size)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    return dict(env._smp_gsi_safety_stats)  # type: ignore[attr-defined]
  finally:
    env.close()


def validate(cfg: Cfg) -> dict[str, object]:
  device = cfg.device or detect_device()
  seed_everything(cfg.seed)
  candidate = sample_speed_stats(cfg.ckpt_path, cfg.num_samples, cfg.batch_size, device)
  seed_everything(cfg.seed)
  reference = sample_speed_stats(
    cfg.reference_ckpt_path, cfg.num_samples, cfg.batch_size, device
  )
  reference_p95 = float(reference["speed_p95_m_s"])
  candidate_p95 = float(candidate["speed_p95_m_s"])
  reduction = 1.0 - candidate_p95 / reference_p95
  result: dict[str, object] = {
    "checkpoint": cfg.ckpt_path,
    "reference_checkpoint": cfg.reference_ckpt_path,
    "candidate": candidate,
    "reference": reference,
    "p95_reduction_fraction": reduction,
  }
  if cfg.check_gsi:
    result["gsi"] = gsi_stats(cfg, device)

  failures: list[str] = []
  if not bool(candidate["finite"]):
    failures.append("candidate DDPM forward/sample contains NaN or Inf")
  if candidate_p95 > cfg.max_root_planar_speed:
    failures.append(
      f"candidate speed p95 {candidate_p95:.4f} > {cfg.max_root_planar_speed:.4f}"
    )
  if reduction < cfg.min_p95_reduction:
    failures.append(f"p95 reduction {reduction:.2%} < {cfg.min_p95_reduction:.2%}")
  if cfg.check_gsi:
    acceptance = float(result["gsi"]["acceptance_rate"])  # type: ignore[index]
    if acceptance < cfg.min_gsi_acceptance:
      failures.append(f"GSI acceptance {acceptance:.2%} < {cfg.min_gsi_acceptance:.2%}")
  result["failures"] = failures
  output = Path(cfg.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2))
  if failures and cfg.enforce_thresholds:
    raise RuntimeError("; ".join(failures))
  return result


if __name__ == "__main__":
  validate(tyro.cli(Cfg))
