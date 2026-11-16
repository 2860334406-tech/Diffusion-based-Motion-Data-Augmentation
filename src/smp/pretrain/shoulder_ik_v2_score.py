"""Clip-balanced calibration and overall score gates for G1 shoulder IK-v2."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from smp.pretrain.shoulder80_score import paired_scores
from smp.pretrain.shoulder_ik_v2 import CONDITION_NAMES
from smp.rl.conditional_score import (
  FIXED_TIMESTEPS,
  score_feature_weights,
  weighted_denoising_mse,
)


def clip_first_indices(
  dataset, condition: int, samples: int, seed: int
) -> torch.Tensor:
  """Deterministically sample clips first; shoulder clips are amplitude-balanced."""
  if samples <= 0 or condition not in (0, 1, 2):
    raise ValueError("Invalid clip-first sample request")
  generator = torch.Generator().manual_seed(seed)
  selected = (dataset.labels == condition).nonzero().flatten()
  if not len(selected):
    raise ValueError(f"Condition {condition} has no samples")

  def draw(pool: torch.Tensor, count: int) -> list[torch.Tensor]:
    clip_ids = torch.unique(dataset.clip_indices[pool], sorted=True)
    chosen_clips = clip_ids[torch.randint(len(clip_ids), (count,), generator=generator)]
    result = []
    for clip in chosen_clips.tolist():
      candidates = pool[dataset.clip_indices[pool] == clip]
      offset = int(torch.randint(len(candidates), (1,), generator=generator))
      result.append(candidates[offset])
    return result

  if condition == 0:
    result = draw(selected, samples)
  else:
    edges = torch.tensor(tuple(float(value) for value in range(20, 70, 5)))
    bins = torch.bucketize(dataset.target_amplitudes_deg[selected], edges)
    base, remainder = divmod(samples, 11)
    result = []
    for bin_id in range(11):
      pool = selected[bins == bin_id]
      if not len(pool):
        raise ValueError(f"Condition {condition} amplitude bin {bin_id} is empty")
      result.extend(draw(pool, base + int(bin_id < remainder)))
  order = torch.randperm(len(result), generator=generator)
  return torch.stack(result)[order]


@torch.no_grad()
def calibrate(model, scheduler, dataset, *, samples=4096, seed=42, batch_size=256):
  device = next(model.parameters()).device
  means = torch.zeros(3, len(FIXED_TIMESTEPS), dtype=torch.float64)
  weights = score_feature_weights(device)
  for condition in range(3):
    chosen = clip_first_indices(dataset, condition, samples, seed + condition * 1000)
    noise_generator = torch.Generator(device=device).manual_seed(seed + condition)
    for start in range(0, samples, batch_size):
      x = dataset.windows[chosen[start : start + batch_size]].to(device)
      labels = torch.full((len(x),), condition, device=device, dtype=torch.long)
      for k, timestep in enumerate(FIXED_TIMESTEPS):
        t = torch.full_like(labels, timestep)
        noise = torch.randn(x.shape, device=device, generator=noise_generator)
        prediction = model(scheduler.add_noise(x, noise, t), t, labels)
        means[condition, k] += (
          weighted_denoising_mse(prediction, noise, weights).double().sum().cpu()
        )
  means = (means / samples).float()
  if not torch.isfinite(means).all() or bool((means <= 0).any()):
    raise FloatingPointError("Invalid IK-v2 calibration means")
  return means


def _clip_values(values, rows):
  by_clip = defaultdict(list)
  for value, row in zip(np.asarray(values, dtype=float), rows, strict=True):
    by_clip[row["clip_id"]].append(float(value))
  return np.array([np.mean(by_clip[key]) for key in sorted(by_clip)]), by_clip


def clip_rate(values, rows, seed=42):
  per_clip, by_clip = _clip_values(values, rows)
  generator = np.random.default_rng(seed)
  draws = generator.integers(len(per_clip), size=(2000, len(per_clip)))
  bootstrap = per_clip[draws].mean(1)
  return {
    "rate": float(per_clip.mean()),
    "windows": len(values),
    "clips": len(per_clip),
    "clip_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    "per_clip": {
      key: {"windows": len(items), "rate": float(np.mean(items))}
      for key, items in sorted(by_clip.items())
    },
    "unit": "equal-weight mean of per-clip window rates",
  }


def _clip_quantile(values, rows, quantile):
  per_clip, _ = _clip_values(values, rows)
  return float(np.quantile(per_clip, quantile))


def _group_metrics(reward, stand_reward, reverse_reward, margin, rows):
  return {
    "correct_gt_stand": clip_rate((reward > stand_reward).numpy(), rows),
    "correct_gt_reverse": clip_rate((reward > reverse_reward).numpy(), rows),
    "correct_condition_gt_opposite": clip_rate((margin > 0).numpy(), rows),
    "correct_reward_p05": _clip_quantile(reward.numpy(), rows, 0.05),
    "correct_reward_mean": _clip_quantile(reward.numpy(), rows, 0.50),
    "stand_reward_mean": _clip_quantile(stand_reward.numpy(), rows, 0.50),
    "reverse_reward_mean": _clip_quantile(reverse_reward.numpy(), rows, 0.50),
    "median_margin": _clip_quantile(margin.numpy(), rows, 0.50),
  }


def score_gate_report(dataset, correct, stationary, reverse, static_left):
  checks, shoulders, amplitude_diagnostics = {}, {}, {}
  for condition in (1, 2):
    selected = (dataset.labels == condition).nonzero().flatten()
    rows = [dataset.rows[int(index)] for index in selected]
    c, s, r = correct[condition], stationary[condition], reverse[condition]
    reward, stand_reward, reverse_reward = (
      values["reward"].mean(-1) for values in (c, s, r)
    )
    margin = c["contrast_margin"].mean(-1)
    name = CONDITION_NAMES[condition]
    metrics = _group_metrics(reward, stand_reward, reverse_reward, margin, rows)
    shoulders[name] = metrics
    checks[f"{name}/correct_gt_stand_ge_0.95"] = (
      metrics["correct_gt_stand"]["rate"] >= 0.95
    )
    checks[f"{name}/correct_gt_reverse_ge_0.95"] = (
      metrics["correct_gt_reverse"]["rate"] >= 0.95
    )
    checks[f"{name}/condition_better_ge_0.95"] = (
      metrics["correct_condition_gt_opposite"]["rate"] >= 0.95
    )
    checks[f"{name}/correct_reward_p05_ge_0.05"] = metrics["correct_reward_p05"] >= 0.05
    checks[f"{name}/median_margin_ge_0.5"] = metrics["median_margin"] >= 0.5

    amplitudes = dataset.target_amplitudes_deg[selected]
    for low, high in ((15, 30), (30, 45), (45, 60), (60, 70.001)):
      mask = (amplitudes >= low) & (amplitudes < high)
      diagnostic_rows = [
        row for row, keep in zip(rows, mask.tolist(), strict=True) if keep
      ]
      amplitude_diagnostics[f"{name}/{low:g}-{min(high, 70):g}deg"] = _group_metrics(
        reward[mask],
        stand_reward[mask],
        reverse_reward[mask],
        margin[mask],
        diagnostic_rows,
      )

  stand_ids = (dataset.labels == 0).nonzero().flatten()
  stand_rows = [dataset.rows[int(index)] for index in stand_ids]
  static_margin = static_left["contrast_margin"].mean(-1)
  per_clip_margin, _ = _clip_values(static_margin.numpy(), stand_rows)
  preference = float((per_clip_margin > 0).mean())
  absolute = float(np.abs(per_clip_margin).mean())
  checks["static_preference_35_to_65_percent"] = 0.35 <= preference <= 0.65
  checks["static_mean_absolute_margin_le_0.2"] = absolute <= 0.2
  return {
    "split": dataset.split,
    "paired_noise_repeats": int(static_left["contrast_margin"].shape[-1]),
    "gate_granularity": "per-shoulder overall; amplitude bands are diagnostic only",
    "rate_unit": "clip-balanced; noise repetitions are averaged before comparison",
    "shoulders": shoulders,
    "amplitude_diagnostics": amplitude_diagnostics,
    "stand_bias": {
      "left_preference_rate": preference,
      "mean_absolute_margin": absolute,
      "clips": len(per_clip_margin),
    },
    "gate": {
      "passed": all(checks.values()),
      "checks": checks,
      "failed_checks": [key for key, passed in checks.items() if not passed],
    },
  }


@torch.no_grad()
def evaluate_scores(model, scheduler, dataset, means, *, repeats=64, valid_frames=10):
  mirror = dataset.mirror_indices()
  stand_ids = (dataset.labels == 0).nonzero().flatten()
  correct, stationary, reverse, raw = {}, {}, {}, {}
  for condition in (1, 2):
    ids = (dataset.labels == condition).nonzero().flatten()
    matched_stand = clip_first_indices(dataset, 0, len(ids), seed=42000 + condition)
    options = {
      "repeats": repeats,
      "seed": 42 + condition * 10000,
      "valid_frames": valid_frames,
    }
    for name, indices, target in (
      ("correct", ids, correct),
      ("stand", matched_stand, stationary),
      ("reverse", mirror[ids], reverse),
    ):
      target[condition] = paired_scores(
        model, scheduler, dataset.windows[indices], condition, means, **options
      )
      for key, value in target[condition].items():
        raw[f"c{condition}_{name}_{key}"] = value.numpy()
      raw[f"c{condition}_{name}_indices"] = indices.numpy()
  static = paired_scores(
    model,
    scheduler,
    dataset.windows[stand_ids],
    1,
    means,
    repeats=repeats,
    seed=90042,
    valid_frames=valid_frames,
  )
  for key, value in static.items():
    raw[f"static_left_{key}"] = value.numpy()
  report = score_gate_report(dataset, correct, stationary, reverse, static)
  report["valid_history_frames"] = valid_frames
  return report, raw
