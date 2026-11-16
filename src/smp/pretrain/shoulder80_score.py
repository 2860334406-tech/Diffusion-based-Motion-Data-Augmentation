"""Frozen V3 final-score validation; noise repeats are not independent motions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from smp.pretrain.shoulder80 import CONDITION_NAMES, PHASE_NAMES
from smp.rl.conditional_score import (
  FIXED_TIMESTEPS,
  contrastive_score_from_energies,
  paired_condition_predictions,
  score_feature_weights,
  weighted_denoising_mse,
)


@torch.no_grad()
def calibrate(model, scheduler, dataset, *, samples=4096, seed=42, batch_size=256):
  device = next(model.parameters()).device
  generator = torch.Generator().manual_seed(seed)
  noise_generator = torch.Generator(device=device).manual_seed(seed)
  means = torch.zeros(3, len(FIXED_TIMESTEPS), dtype=torch.float64)
  weights = score_feature_weights(device)
  for condition in range(3):
    pool = (dataset.labels == condition).nonzero().flatten()
    chosen = pool[torch.randint(len(pool), (samples,), generator=generator)]
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
  if not torch.isfinite(means).all() or (means <= 0).any():
    raise FloatingPointError("Invalid frozen calibration means")
  return means


@torch.no_grad()
def paired_scores(
  model,
  scheduler,
  windows,
  condition,
  means,
  *,
  repeats=64,
  seed=42,
  batch_size=256,
  valid_frames=10,
):
  """Per-window/per-noise scores. Reset seed to pair different motion pools too."""
  if not 1 <= valid_frames <= 10 or repeats < 1 or condition not in (0, 1, 2):
    raise ValueError("Invalid score request")
  device = next(model.parameters()).device
  weights = score_feature_weights(device)
  generator = torch.Generator(device=device).manual_seed(seed)
  result = defaultdict(list)
  for start in range(0, len(windows), batch_size):
    x = windows[start : start + batch_size].to(device)
    condition_ids = torch.full((len(x),), condition, device=device, dtype=torch.long)
    opposite_ids = torch.full_like(condition_ids, {0: 0, 1: 2, 2: 1}[condition])
    mask = torch.zeros(len(x), 10, device=device)
    mask[:, -valid_frames:] = 1
    accumulated = defaultdict(list)
    for _ in range(repeats):
      correct, opposite = [], []
      for timestep in FIXED_TIMESTEPS:
        t = torch.full_like(condition_ids, timestep)
        noise = torch.randn(x.shape, generator=generator, device=device)
        x_t = scheduler.add_noise(x, noise, t)
        pc, po = paired_condition_predictions(
          model, x_t, t, condition_ids, opposite_ids
        )
        correct.append(weighted_denoising_mse(pc, noise, weights, mask))
        opposite.append(weighted_denoising_mse(po, noise, weights, mask))
      ec, eo = torch.stack(correct, -1), torch.stack(opposite, -1)
      reward, parts = contrastive_score_from_energies(
        ec, eo, means[condition].to(device).expand_as(ec), condition_ids != 0
      )
      for name, value in {"reward": reward, **parts}.items():
        if not torch.isfinite(value).all():
          raise FloatingPointError(f"Nonfinite score component: {name}")
        accumulated[name].append(value.cpu())
    for name, values in accumulated.items():
      result[name].append(torch.stack(values, -1))
  return {name: torch.cat(values) for name, values in result.items()}


def clustered_rate(values, rows, seed=42):
  """Source-file block bootstrap; keep all overlaps/mirrors/noise within a block."""
  values = np.asarray(values, dtype=float)
  by_source = defaultdict(list)
  for value, row in zip(values, rows, strict=True):
    by_source[row["source_id"]].append(float(value))
  sources = sorted(by_source)
  sums = np.array([sum(by_source[source]) for source in sources])
  counts = np.array([len(by_source[source]) for source in sources])
  generator = np.random.default_rng(seed)
  draws = generator.integers(len(sources), size=(2000, len(sources)))
  bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
  return {
    "rate": float(values.mean()),
    "windows_including_overlap_and_mirrors": len(values),
    "original_files": len(sources),
    "source_block_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    "per_original_file": {
      source: {"windows": len(v), "rate": float(np.mean(v))}
      for source, v in sorted(by_source.items())
    },
    "independence_note": "Blocks are source files, not windows or repeated noises; actor dependence can remain.",
  }


def score_gate_report(dataset, correct, stationary, reverse, static_left):
  checks, phases = {}, {}
  for condition in (1, 2):
    selected = (dataset.labels == condition).nonzero().flatten()
    c, s, r = correct[condition], stationary[condition], reverse[condition]
    reward, stand_reward, reverse_reward = (v["reward"].mean(-1) for v in (c, s, r))
    margin = c["contrast_margin"].mean(-1)
    for phase in (1, 2, 3):
      mask = dataset.phase_ids[selected] == phase
      name = f"{CONDITION_NAMES[condition]}/{PHASE_NAMES[phase]}"
      rows = [dataset.rows[int(i)] for i in selected[mask]]
      greater_stand = clustered_rate((reward[mask] > stand_reward[mask]).numpy(), rows)
      greater_reverse = clustered_rate(
        (reward[mask] > reverse_reward[mask]).numpy(), rows
      )
      condition_better = clustered_rate((margin[mask] > 0).numpy(), rows)
      p05 = float(torch.quantile(reward[mask], 0.05))
      median_margin = float(margin[mask].median())
      phases[name] = {
        "correct_gt_stand": greater_stand,
        "correct_gt_reverse": greater_reverse,
        "correct_condition_gt_opposite": condition_better,
        "correct_reward_p05": p05,
        "correct_reward_mean": float(reward[mask].mean()),
        "stand_reward_mean": float(stand_reward[mask].mean()),
        "reverse_reward_mean": float(reverse_reward[mask].mean()),
        "median_margin": median_margin,
      }
      checks[f"{name}/correct_gt_stand_ge_0.95"] = greater_stand["rate"] >= 0.95
      checks[f"{name}/correct_gt_reverse_ge_0.95"] = greater_reverse["rate"] >= 0.95
      checks[f"{name}/correct_reward_p05_ge_0.05"] = p05 >= 0.05
      if phase != 1:
        checks[f"{name}/condition_better_ge_0.95"] = condition_better["rate"] >= 0.95
        checks[f"{name}/median_margin_ge_0.5"] = median_margin >= 0.5
  static_margins = static_left["contrast_margin"]
  preference = float((static_margins > 0).float().mean())
  absolute = float(static_margins.abs().mean())
  checks["static_preference_35_to_65_percent"] = 0.35 <= preference <= 0.65
  checks["static_mean_absolute_margin_le_0.2"] = absolute <= 0.2
  return {
    "split": dataset.split,
    "paired_noise_repeats": static_margins.shape[-1],
    "rate_unit": "window mean final reward over 64 noises; source-file block uncertainty",
    "phases": phases,
    "stand_bias": {
      "left_preference_rate": preference,
      "mean_absolute_margin": absolute,
      "windows": len(static_margins),
    },
    "gate": {
      "passed": all(checks.values()),
      "checks": checks,
      "failed_checks": [name for name, passed in checks.items() if not passed],
    },
  }


@torch.no_grad()
def evaluate_scores(model, scheduler, dataset, means, *, repeats=64, valid_frames=10):
  mirror = dataset.mirror_indices()
  stand_ids = (dataset.labels == 0).nonzero().flatten()
  correct, stationary, reverse, raw = {}, {}, {}, {}
  generator = torch.Generator().manual_seed(42)
  for condition in (1, 2):
    ids = (dataset.labels == condition).nonzero().flatten()
    matched_stand = stand_ids[
      torch.randint(len(stand_ids), (len(ids),), generator=generator)
    ]
    options = dict(
      repeats=repeats, seed=42 + condition * 10000, valid_frames=valid_frames
    )
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
