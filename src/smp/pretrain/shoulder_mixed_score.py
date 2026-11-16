"""Source-weighted final-score gates; paired noise is never an independent clip."""

from __future__ import annotations

import numpy as np
import torch

from smp.pretrain.shoulder80_score import paired_scores
from smp.pretrain.shoulder_mixed import CONDITION_NAMES
from smp.rl.conditional_score import (
  FIXED_TIMESTEPS,
  score_feature_weights,
  weighted_denoising_mse,
)


def clip_first_indices(dataset, condition, samples, seed):
  return dataset.sample_indices(condition, samples, seed)


@torch.no_grad()
def calibrate(model, scheduler, dataset, *, samples=4096, seed=42, batch_size=256):
  device = next(model.parameters()).device
  means = torch.zeros(3, 3, dtype=torch.float64)
  weights = score_feature_weights(device)
  for condition in range(3):
    ids = dataset.sample_indices(condition, samples, seed + condition * 1000)
    generator = torch.Generator(device=device).manual_seed(seed + condition)
    for offset in range(0, samples, batch_size):
      x = dataset.windows[ids[offset : offset + batch_size]].to(device)
      labels = torch.full((len(x),), condition, device=device, dtype=torch.long)
      for k, timestep in enumerate(FIXED_TIMESTEPS):
        t = torch.full_like(labels, timestep)
        noise = torch.randn(x.shape, device=device, generator=generator)
        prediction = model(scheduler.add_noise(x, noise, t), t, labels)
        means[condition, k] += (
          weighted_denoising_mse(prediction, noise, weights).double().sum().cpu()
        )
  means = (means / samples).float()
  if not torch.isfinite(means).all() or (means <= 0).any():
    raise FloatingPointError("Invalid mixed calibration")
  return means


def _weighted_quantile(values, weights, q):
  order = np.argsort(values, kind="stable")
  cdf = np.cumsum(weights[order])
  return float(values[order[min(np.searchsorted(cdf / cdf[-1], q), len(order) - 1)]])


def _metrics(dataset, condition, ids, correct, stand, reverse):
  # Average the 64 paired noises first, then compute a rate within each clip.
  values = {
    "correct_gt_stand": (correct["reward"].mean(-1) > stand["reward"].mean(-1)).numpy(),
    "correct_gt_reverse": (
      correct["reward"].mean(-1) > reverse["reward"].mean(-1)
    ).numpy(),
    "condition_better": (correct["contrast_margin"].mean(-1) > 0).numpy(),
    "reward": correct["reward"].mean(-1).numpy(),
    "margin": correct["contrast_margin"].mean(-1).numpy(),
  }
  row_weights = dataset.clip_equal_weights(condition)[ids].numpy()

  def summarize(mask):
    positions = np.flatnonzero(mask)
    if not len(positions):
      return {"available": False, "clips": 0}
    groups = {}
    for j in positions:
      groups.setdefault(dataset.rows[int(ids[j])]["clip_id"], []).append(j)
    clip_weights = np.array([row_weights[v].sum() for v in groups.values()])
    clip_weights /= clip_weights.sum()
    averages = {
      name: np.array(
        [np.average(array[v], weights=row_weights[v]) for v in groups.values()]
      )
      for name, array in values.items()
    }
    metrics = {
      name: float(np.dot(clip_weights, averages[name]))
      for name in ("correct_gt_stand", "correct_gt_reverse", "condition_better")
    }
    metrics.update(
      available=True,
      windows=len(positions),
      clips=len(groups),
      correct_reward_p05=_weighted_quantile(averages["reward"], clip_weights, 0.05),
      median_margin=_weighted_quantile(averages["margin"], clip_weights, 0.5),
    )
    # Resample original clips within source, keeping the configured source mass.
    rng = np.random.default_rng(42)
    clip_sources = [dataset.rows[int(ids[v[0]])]["dataset"] for v in groups.values()]
    intervals = {}
    for name in ("correct_gt_stand", "correct_gt_reverse", "condition_better"):
      draws = np.zeros(2000)
      for source in sorted(set(clip_sources)):
        selected = np.flatnonzero(np.array(clip_sources) == source)
        mass = clip_weights[selected].sum()
        choices = rng.choice(
          selected,
          size=(2000, len(selected)),
          replace=True,
          p=clip_weights[selected] / mass,
        )
        draws += mass * averages[name][choices].mean(-1)
      intervals[name] = np.quantile(draws, [0.025, 0.975]).tolist()
    metrics["clip_bootstrap_ci95"] = intervals
    return metrics

  overall = summarize(np.ones(len(ids), dtype=bool))
  overall["by_source"] = {
    source: summarize(
      np.array([dataset.rows[int(i)]["dataset"] == source for i in ids])
    )
    for source in ("ik", "personal")
  }
  return overall


@torch.no_grad()
def evaluate_scores(model, scheduler, dataset, means, *, repeats=64, valid_frames=10):
  mirror = dataset.mirror_indices()
  shoulders, checks, raw, diagnostics = {}, {}, {}, {}
  for condition in (1, 2):
    ids = (dataset.labels == condition).nonzero().flatten()
    stand = dataset.sample_indices(0, len(ids), 42000 + condition)
    results = {}
    for name, indices in (("correct", ids), ("stand", stand), ("reverse", mirror[ids])):
      parts = paired_scores(
        model,
        scheduler,
        dataset.windows[indices],
        condition,
        means,
        repeats=repeats,
        seed=42 + condition * 10000,
        valid_frames=valid_frames,
      )
      results[name] = parts
      for k, v in parts.items():
        raw[f"c{condition}_{name}_{k}"] = v.numpy()
      raw[f"c{condition}_{name}_indices"] = indices.numpy()
    m = _metrics(dataset, condition, ids, **results)
    name = CONDITION_NAMES[condition]
    shoulders[name] = m
    for field in ("correct_gt_stand", "correct_gt_reverse", "condition_better"):
      checks[f"{name}/{field}_ge_0.95"] = m[field] >= 0.95
    checks[f"{name}/median_margin_ge_0.5"] = m["median_margin"] >= 0.5
    checks[f"{name}/correct_reward_p05_ge_0.05"] = m["correct_reward_p05"] >= 0.05
    for a, b in ((15, 30), (30, 45), (45, 60), (60, 70.001)):
      selected = torch.tensor(
        [
          r["dataset"] == "ik" and a <= (r["target_yaw_amplitude_deg"] or 0) < b
          for r in (dataset.rows[int(i)] for i in ids)
        ]
      )
      if selected.any():
        diagnostics[f"{name}/{a}-{min(b, 70)}deg"] = _metrics(
          dataset,
          condition,
          ids[selected],
          **{k: {f: v[selected] for f, v in val.items()} for k, val in results.items()},
        )
  ids = (dataset.labels == 0).nonzero().flatten()
  static = paired_scores(
    model,
    scheduler,
    dataset.windows[ids],
    1,
    means,
    repeats=repeats,
    seed=90042,
    valid_frames=valid_frames,
  )
  margin = static["contrast_margin"].mean(-1).numpy()
  weights = dataset.clip_equal_weights(0)[ids].numpy()
  groups = {}
  for j, i in enumerate(ids.tolist()):
    groups.setdefault(dataset.rows[i]["clip_id"], []).append(j)
  clip_margin = np.array(
    [np.average(margin[v], weights=weights[v]) for v in groups.values()]
  )
  clip_weights = np.array([weights[v].sum() for v in groups.values()])
  preference = float(np.average(clip_margin > 0, weights=clip_weights))
  absolute = float(np.average(np.abs(clip_margin), weights=clip_weights))
  checks["static_preference_35_to_65_percent"] = 0.35 <= preference <= 0.65
  checks["static_mean_absolute_margin_le_0.2"] = absolute <= 0.2
  for k, v in static.items():
    raw[f"static_left_{k}"] = v.numpy()
  report = {
    "split": dataset.split,
    "paired_noise_repeats": repeats,
    "valid_history_frames": valid_frames,
    "shoulders": shoulders,
    "amplitude_diagnostics": diagnostics,
    "rate_unit": "source-weighted clip rates; noise repeats averaged first",
    "stand_bias": {
      "left_preference_rate": preference,
      "mean_absolute_margin": absolute,
      "clips": len(groups),
    },
    "gate": {
      "passed": all(checks.values()),
      "checks": checks,
      "failed_checks": [k for k, v in checks.items() if not v],
    },
  }
  return report, raw
