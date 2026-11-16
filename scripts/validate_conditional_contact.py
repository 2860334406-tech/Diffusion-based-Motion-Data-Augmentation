"""Paired-noise five-condition sampling, GSI gate, and headless sample previews."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from smp.pretrain.conditional_dataset import ConditionalMotionDataset
from smp.pretrain.conditioning import (
  CONDITION_NAMES,
  CONDITION_SCHEMA,
  sample_condition,
  sha256,
)
from smp.rl.utils import load_denoiser


@dataclass
class Cfg:
  checkpoint: str = "logs/reproduction/conditional_contact_v1/prior/pretrained.pt"
  output_dir: str = "logs/reproduction/conditional_contact_v1/prior_validation"
  data_dir: str = "logs/reproduction/conditional_contact_v1/data"
  samples_per_condition: int = 4096
  batch_size: int = 256
  device: str = "cuda:0"
  seed: int = 42
  check_gsi: bool = True
  record_video: bool = True
  enforce_thresholds: bool = True


def motion_statistics(windows, condition):
  speed = torch.linalg.vector_norm(windows[..., 53:55], dim=-1).max(-1).values
  vx = windows[..., 53].mean(-1)
  dx = windows[:, -1, 0]-windows[:, 0, 0]
  waist = windows[..., 21]
  delta = waist[:, -1]-waist[:, 0]
  # The first 6D column is the heading-relative x-axis.
  yaw = torch.atan2(windows[..., 4], windows[..., 3])
  yaw_delta = torch.atan2(torch.sin(yaw[:, -1]-yaw[:, 0]), torch.cos(yaw[:, -1]-yaw[:, 0]))
  if condition == 0:
    valid = (speed <= .15) & (waist.abs().max(-1).values <= .08)
  elif condition in (1, 2):
    sign = 1 if condition == 1 else -1
    signed_q, signed_d = sign*waist[:, -1], sign*delta
    entering = (signed_d >= .01) & (signed_q >= 0)
    holding = (signed_q >= .08) & (signed_d >= -.02)
    valid = (entering | holding) & (signed_q <= .35) & (speed <= .30) & (yaw_delta.abs() <= .05)
  else:
    sign = -1 if condition == 3 else 1
    valid = (sign*vx >= .03) & (sign*dx > 0)
  stats = {
    "samples": len(windows), "finite": bool(torch.isfinite(windows).all()),
    "speed_max_in_window_p95": float(torch.quantile(speed, .95)),
    "mean_vx": float(vx.mean()), "mean_x_displacement": float(dx.mean()),
    "mean_waist_last_rad": float(waist[:, -1].mean()),
    "mean_waist_delta_rad": float(delta.mean()),
    "pelvis_yaw_delta_abs_p95": float(torch.quantile(yaw_delta.abs(), .95)),
    "condition_success_rate": float(valid.float().mean()),
    "feature_diversity_mean_std": float(windows.std(dim=0).mean()),
  }
  if condition in (1, 2):
    stats.update(
      entering_rate=float(entering.float().mean()), holding_rate=float(holding.float().mean()),
      entering_or_holding_rate=float((entering | holding).float().mean()),
      signed_waist_positive_rate=float((signed_q > 0).float().mean()),
      pelvis_turn_within_limit_rate=float((yaw_delta.abs() <= .05).float().mean()),
      planar_speed_within_limit_rate=float((speed <= .30).float().mean()),
      waist_amplitude_within_limit_rate=float((signed_q <= .35).float().mean()),
    )
  return stats


@torch.no_grad()
def score_confusion(bundle, data_dir, seed):
  model, scheduler, *_ = bundle
  validation = ConditionalMotionDataset(data_dir, "validation")
  device = next(model.parameters()).device
  torch.manual_seed(seed)
  if device.type == "cuda":
    torch.cuda.manual_seed(seed)
  matrix = []
  for actual in range(5):
    ids = (validation.labels == actual).nonzero().flatten()[:64]
    x = validation.windows[ids].to(device)
    errors = torch.zeros(len(ids), 5, device=device)
    for timestep in (8, 15, 22):
      t = torch.full((len(ids),), timestep, device=device, dtype=torch.long)
      noise = torch.randn_like(x)
      noisy = scheduler.add_noise(x, noise, t)
      for candidate in range(5):
        eps = model(noisy, t, torch.full_like(t, candidate))
        errors[:, candidate] += (eps-noise).square().mean((-1, -2))
    predicted = errors.argmin(-1)
    matrix.append((torch.bincount(predicted, minlength=5).float()/len(ids)).cpu().tolist())
  return matrix


def gsi_and_video(cfg, bundle, samples, output):
  import mediapy
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer import OffscreenRenderer, ViewerConfig

  from smp.rl.events import _prime_sim_and_buffer
  from smp.rl.tasks.force_unloading.conditional import conditional_env_cfg

  config = conditional_env_cfg(play=True)
  config.scene.num_envs = 64
  config.seed = cfg.seed
  config.events["init_smp_state"].params.update(
    ckpt_path=cfg.checkpoint, compile_model=False, gsi_buffer_size=1024,
    gsi_batch_size=256, gsi_max_root_planar_speed=.15,
  )
  if "gsi_refresh" in config.events:
    config.events["gsi_refresh"].params["num_samples"] = 64
  env = ManagerBasedRlEnv(config, device=cfg.device)
  renderer = None
  try:
    stats = dict(env._smp_gsi_safety_stats)
    if cfg.record_video:
      render_cfg = ViewerConfig(height=480, width=640, distance=2.5,
                                elevation=-8., azimuth=35.,
                                origin_type=ViewerConfig.OriginType.ASSET_ROOT,
                                entity_name="robot")
      renderer = OffscreenRenderer(model=env.sim.mj_model, cfg=render_cfg, scene=env.scene)
      renderer.initialize()
      for name, windows in samples.items():
        frames = []
        for window in windows[:4].to(cfg.device):
          for frame in range(10):
            pose = window[frame].view(1, 1, 59).expand(1, 10, 59).contiguous()
            _prime_sim_and_buffer(env, torch.tensor([0], device=cfg.device), pose)
            env.sim.forward()
            env.scene.update(env.step_dt)
            renderer.update(env.sim.data)
            pixels = renderer.render().copy()
            frames.extend([pixels]*5)  # explicitly 5x slowed FK preview, no physics rollout
          frames.extend([np.zeros_like(pixels)]*10)
        mediapy.write_video(output / f"{name}_fk_samples_5x_slow.mp4", frames, fps=50)
    return stats
  finally:
    if renderer is not None:
      renderer.close()
    env.close()


def sync_validation_summary(report, checkpoint):
  """Update only scalars/configuration in the existing training run, never artifacts."""
  metadata_path = Path(checkpoint).parent / "run.json"
  if not metadata_path.is_file():
    return {"status": "not_configured"}
  metadata = json.loads(metadata_path.read_text())
  url = metadata.get("wandb_url")
  if not url:
    return {"status": "not_configured"}
  try:
    import wandb
    run = wandb.Api(timeout=30).run(url.split("wandb.ai/", 1)[1].replace("/runs/", "/"))
    summary = {"acceptance/passed": report["gate"]["passed"],
               "acceptance/checks": report["gate"]["checks"],
               "acceptance/checkpoint_sha256": report["checkpoint_sha256"],
               "acceptance/epoch": report["checkpoint_epoch"]}
    for name, stats in report["conditions"].items():
      for key, value in stats.items():
        summary[f"acceptance/{name}/{key}"] = value
    summary["acceptance/gsi_rate"] = report.get("gsi", {}).get("acceptance_rate", 0.)
    run.summary.update(summary)
    return {"status": "synced", "run_url": url}
  except Exception as error:
    return {"status": "pending", "error_type": type(error).__name__, "run_url": url}


def validate(cfg):
  if cfg.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise RuntimeError("Conditional experiment must run on physical GPU1")
  torch.set_num_threads(2)
  output = Path(cfg.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  payload = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
  if payload.get("condition_schema") != CONDITION_SCHEMA:
    raise ValueError("Checkpoint condition dictionary is incompatible")
  bundle = load_denoiser(cfg.checkpoint, cfg.device)
  samples, statistics = {}, {}
  for condition, name in enumerate(CONDITION_NAMES):
    chunks = []
    for start in range(0, cfg.samples_per_condition, cfg.batch_size):
      # Reset both initial and ancestral noise for a paired comparison.
      torch.manual_seed(cfg.seed + start)
      if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed(cfg.seed + start)
      count = min(cfg.batch_size, cfg.samples_per_condition-start)
      chunks.append(sample_condition(bundle, condition, count).cpu())
    windows = torch.cat(chunks)
    samples[name] = windows
    statistics[name] = motion_statistics(windows, condition)
    np.savez_compressed(output / f"{name}_samples.npz", windows=windows.numpy(),
                        condition_id=np.full(len(windows), condition))
    print(json.dumps({"condition": name, **statistics[name]}), flush=True)
  report = {
    "config": asdict(cfg), "checkpoint_sha256": sha256(cfg.checkpoint),
    "checkpoint_epoch": payload["epoch"], "condition_schema": CONDITION_SCHEMA,
    "conditions": statistics,
    "score_confusion_matrix": score_confusion(bundle, cfg.data_dir, cfg.seed),
    "physical_gpu": 1,
  }
  if cfg.check_gsi:
    try:
      report["gsi"] = gsi_and_video(cfg, bundle, samples, output)
    except Exception as error:
      report["gsi"] = {"error": repr(error), "acceptance_rate": 0.0}
  checks = {
    "all_samples_finite": all(s["finite"] for s in statistics.values()),
    "stand_speed_p95_le_0.15": statistics["stand"]["speed_max_in_window_p95"] <= .15,
    "chest_mean_vx_le_minus_0.03": statistics["chest"]["mean_vx"] <= -.03,
    "back_mean_vx_ge_0.03": statistics["back"]["mean_vx"] >= .03,
    **{f"{name}_direction_rate_ge_0.80": statistics[name]["condition_success_rate"] >= .8
       for name in CONDITION_NAMES[1:]},
  }
  if cfg.check_gsi:
    checks["stand_gsi_acceptance_ge_0.50"] = report["gsi"]["acceptance_rate"] >= .50
    checks["gsi_and_preview_valid"] = "error" not in report["gsi"]
  report["gate"] = {"passed": all(checks.values()), "checks": checks}
  if cfg.enforce_thresholds:
    report["wandb_sync"] = sync_validation_summary(report, cfg.checkpoint)
  (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  for name, windows in samples.items():
    axes[0].hist(windows[..., 53].mean(-1).numpy(), bins=60, density=True,
                 histtype="step", label=name)
    axes[1].hist(windows[:, -1, 21].numpy(), bins=60, density=True,
                 histtype="step", label=name)
  axes[0].set_xlabel("Heading-local mean root vx (m/s)")
  axes[1].set_xlabel("Last waist yaw (rad)")
  for axis in axes:
    axis.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(output / "condition_distributions.png", dpi=160)
  plt.close(fig)
  print(json.dumps(report["gate"]), flush=True)
  if cfg.enforce_thresholds and not report["gate"]["passed"]:
    raise RuntimeError("Conditional Prior failed acceptance; PPO must not start")
  return report


if __name__ == "__main__":
  validate(tyro.cli(Cfg))
