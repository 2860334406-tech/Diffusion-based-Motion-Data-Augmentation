"""Build and validate the frozen ShoulderScoreV3 energy calibration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import tyro

from smp.pretrain.conditional_dataset import ConditionalMotionDataset
from smp.pretrain.conditioning import CONDITION_NAMES, sha256
from smp.rl.conditional_score import (
  FIXED_TIMESTEPS,
  SCORE_MODE,
  SCORE_SCHEMA_VERSION,
  calibration_identity,
  score_feature_weights,
  weighted_denoising_mse,
)
from smp.rl.utils import load_denoiser


@dataclass(frozen=True)
class Cfg:
  checkpoint: str = "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
  data_dir: str = "logs/reproduction/conditional_contact_v2/data"
  output: str = (
    "logs/reproduction/conditional_contact_shoulder_score_v3/"
    "calibration/shoulder_score_calibration.pt"
  )
  report: str = (
    "logs/reproduction/conditional_contact_shoulder_score_v3/calibration/report.json"
  )
  samples_per_condition: int = 4096
  batch_size: int = 256
  validation_noise_repeats: int = 16
  seed: int = 42
  device: str = "cuda:0"
  enforce_thresholds: bool = True


def _paired_margins(
  model,
  scheduler,
  windows: torch.Tensor,
  correct_condition: int,
  opposite_condition: int,
  weights: torch.Tensor,
  *,
  repeats: int,
  batch_size: int,
  seed: int,
  device: torch.device,
) -> torch.Tensor:
  margins = []
  for repeat in range(repeats):
    generator = torch.Generator(device=device).manual_seed(seed + 1009 * repeat)
    for start in range(0, len(windows), batch_size):
      x = windows[start : start + batch_size].to(device)
      count = len(x)
      correct_id = torch.full(
        (count,), correct_condition, device=device, dtype=torch.long
      )
      opposite_id = torch.full_like(correct_id, opposite_condition)
      per_timestep = []
      for timestep in FIXED_TIMESTEPS:
        t = torch.full((count,), timestep, device=device, dtype=torch.long)
        noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
        x_t = scheduler.add_noise(x, noise, t)
        prediction = model(
          torch.cat((x_t, x_t)),
          torch.cat((t, t)),
          torch.cat((correct_id, opposite_id)),
        )
        correct_prediction, opposite_prediction = prediction.chunk(2)
        correct_energy = weighted_denoising_mse(correct_prediction, noise, weights)
        opposite_energy = weighted_denoising_mse(opposite_prediction, noise, weights)
        per_timestep.append(
          (opposite_energy - correct_energy)
          / (0.5 * (opposite_energy + correct_energy) + 1.0e-6)
        )
      margins.append(torch.stack(per_timestep, -1).mean(-1).cpu())
  return torch.cat(margins)


@torch.no_grad()
def calibrate(cfg: Cfg) -> dict:
  if cfg.samples_per_condition <= 0 or cfg.batch_size <= 0:
    raise ValueError("Calibration sample and batch counts must be positive")
  if cfg.validation_noise_repeats <= 0:
    raise ValueError("validation_noise_repeats must be positive")
  if cfg.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

  device = torch.device(cfg.device)
  prior_sha = sha256(cfg.checkpoint)
  data_manifest = Path(cfg.data_dir) / "manifest.json"
  data_sha = sha256(data_manifest)
  train = ConditionalMotionDataset(cfg.data_dir, "train")
  validation = ConditionalMotionDataset(cfg.data_dir, "validation")
  if train.manifest_sha256 != data_sha or validation.manifest_sha256 != data_sha:
    raise RuntimeError("Dataset manifest identity changed while loading")

  model, scheduler, *_ = load_denoiser(cfg.checkpoint, device)
  model.eval()
  weights = score_feature_weights(device)
  reference = torch.zeros((5, len(FIXED_TIMESTEPS)), dtype=torch.float64)
  counts = torch.zeros(5, dtype=torch.long)
  index_generator = torch.Generator().manual_seed(cfg.seed)
  noise_generator = torch.Generator(device=device).manual_seed(cfg.seed)

  for condition in range(5):
    available = (train.labels == condition).nonzero(as_tuple=False).flatten()
    sampled = available[
      torch.randint(
        len(available),
        (cfg.samples_per_condition,),
        generator=index_generator,
      )
    ]
    for start in range(0, cfg.samples_per_condition, cfg.batch_size):
      x = train.windows[sampled[start : start + cfg.batch_size]].to(device)
      condition_id = torch.full((len(x),), condition, device=device, dtype=torch.long)
      for column, timestep in enumerate(FIXED_TIMESTEPS):
        t = torch.full((len(x),), timestep, device=device, dtype=torch.long)
        noise = torch.randn(
          x.shape, generator=noise_generator, device=device, dtype=x.dtype
        )
        x_t = scheduler.add_noise(x, noise, t)
        prediction = model(x_t, t, condition_id)
        energy = weighted_denoising_mse(prediction, noise, weights)
        reference[condition, column] += energy.double().sum().cpu()
      counts[condition] += len(x)
  reference /= counts[:, None]
  reference = reference.float()
  if not torch.isfinite(reference).all() or (reference <= 0).any():
    raise FloatingPointError("Calibration reference table contains invalid values")

  shoulder_reports = {}
  checks = {}
  for condition, opposite in ((1, 2), (2, 1)):
    windows = validation.windows[validation.labels == condition]
    margins = _paired_margins(
      model,
      scheduler,
      windows,
      condition,
      opposite,
      weights,
      repeats=cfg.validation_noise_repeats,
      batch_size=cfg.batch_size,
      seed=cfg.seed + 10_000 * condition,
      device=device,
    )
    name = CONDITION_NAMES[condition]
    shoulder_reports[name] = {
      "windows": int(len(windows)),
      "noise_repeats": cfg.validation_noise_repeats,
      "comparisons": int(len(margins)),
      "correct_better_rate": float((margins > 0).float().mean()),
      "median_margin": float(margins.median()),
      "mean_margin": float(margins.mean()),
      "finite": bool(torch.isfinite(margins).all()),
    }
    checks[f"{name}_correct_better_ge_0.95"] = (
      shoulder_reports[name]["correct_better_rate"] >= 0.95
    )
    checks[f"{name}_median_margin_ge_0.5"] = (
      shoulder_reports[name]["median_margin"] >= 0.5
    )
    checks[f"{name}_finite"] = shoulder_reports[name]["finite"]

  stand_windows = validation.windows[validation.labels == 0]
  stand_margins = _paired_margins(
    model,
    scheduler,
    stand_windows,
    1,
    2,
    weights,
    repeats=cfg.validation_noise_repeats,
    batch_size=cfg.batch_size,
    seed=cfg.seed + 90_000,
    device=device,
  )
  left_preference = float((stand_margins > 0).float().mean())
  stand_report = {
    "windows": int(len(stand_windows)),
    "noise_repeats": cfg.validation_noise_repeats,
    "comparisons": int(len(stand_margins)),
    "left_preference_rate": left_preference,
    "mean_margin": float(stand_margins.mean()),
    "mean_absolute_margin": float(stand_margins.abs().mean()),
    "finite": bool(torch.isfinite(stand_margins).all()),
  }
  checks.update(
    {
      "stand_preference_35_to_65_percent": 0.35 <= left_preference <= 0.65,
      "stand_mean_absolute_margin_le_0.2": stand_report["mean_absolute_margin"] <= 0.2,
      "stand_margins_finite": stand_report["finite"],
      "reference_table_finite": bool(torch.isfinite(reference).all()),
    }
  )

  payload = {
    "score_schema_version": SCORE_SCHEMA_VERSION,
    "mode": SCORE_MODE,
    "prior_sha256": prior_sha,
    "data_manifest_sha256": data_sha,
    "fixed_timesteps": list(FIXED_TIMESTEPS),
    "feature_weights": weights.cpu(),
    "reference_mean": reference,
    "samples_per_condition": cfg.samples_per_condition,
    "seed": cfg.seed,
  }
  output = Path(cfg.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix(output.suffix + ".tmp")
  torch.save(payload, temporary)
  os.replace(temporary, output)

  report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "config": asdict(cfg),
    "prior_sha256": prior_sha,
    "data_manifest_sha256": data_sha,
    "calibration_path": str(output),
    "calibration_sha256": sha256(output),
    "calibration_identity": calibration_identity(output, payload),
    "reference_mean": reference.tolist(),
    "counts": counts.tolist(),
    "shoulder_validation": shoulder_reports,
    "stand_bias_validation": stand_report,
    "gate": {"passed": all(checks.values()), "checks": checks},
  }
  report_path = Path(cfg.report)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
  report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  os.replace(report_tmp, report_path)
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)
  if cfg.enforce_thresholds and not report["gate"]["passed"]:
    raise RuntimeError("Frozen ShoulderScoreV3 calibration failed acceptance")
  return report


if __name__ == "__main__":
  calibrate(tyro.cli(Cfg))
