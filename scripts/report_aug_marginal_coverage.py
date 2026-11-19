"""Quantify the marginal coverage added by augmentation over the anchors alone.

The anchors already form a dense grid, so a raw coverage count is not evidence
that augmentation helped. What matters is the set difference: parameter cells
reached by generated samples but NOT by any anchor. This script reports that.

Cells are defined on a fine grid of the four parameters that matter for the
"expected data" table: waist amplitude, entry speed, shoulder abduction, knee.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from smp.pretrain.waist_aug_dataset import (
  COND_SCALE,
  compute_conditions,
  load_csv_clip,
  load_npz_clip,
  resample_to_fps,
  to_38d,
)

AMP_STEP = 5.0      # degrees
SPEED_STEP = 2.0    # deg/s
ABD_STEP = 10.0     # degrees
KNEE_STEP = 5.0     # degrees


@dataclass
class Cfg:
  anchor_dir: str = "logs/reproduction/waist_aug_v1/anchors"
  generated_dir: str = "logs/reproduction/waist_aug_v1/full_grid"
  out: str = "logs/reproduction/waist_aug_v1/marginal_coverage.json"


def _cell(amp: float, speed: float, abd: float, knee: float) -> tuple[int, ...]:
  return (
    int(round(amp / AMP_STEP)),
    int(round(speed / SPEED_STEP)),
    int(round(abd / ABD_STEP)),
    int(round(knee / KNEE_STEP)),
  )


def _anchor_cells(anchor_dir: str) -> tuple[set[tuple[int, ...]], int]:
  paths = [
    p for p in sorted(Path(anchor_dir).rglob("*.npz"))
    if not p.name.endswith(".reference.npz")
  ]
  cells: set[tuple[int, ...]] = set()
  for p in paths:
    clip = resample_to_fps(load_npz_clip(p), 30.0)
    c = compute_conditions(to_38d(clip))
    cells.add(_cell(
      float(c[0] * COND_SCALE[0]) * 57.2957795,
      float(c[1] * COND_SCALE[1]),
      abs(float(c[6] * COND_SCALE[6])) * 57.2957795,
      float(c[4] * COND_SCALE[4]) * 57.2957795,
    ))
  return cells, len(paths)


def _generated_cells(generated_dir: str) -> tuple[set[tuple[int, ...]], int, int]:
  """Cells from the FINAL exported CSVs (post-IK), not the sampler summary.

  The summary records an intended target, not the realised motion; the CSVs are
  the actual deliverable, so coverage must be computed from them.
  """
  cells: set[tuple[int, ...]] = set()
  paths = sorted(Path(generated_dir).glob("*/clips/accepted/**/*.csv"))
  for p in paths:
    # Exported CSVs are 50 Hz. Using a 30 Hz time base here halves the measured
    # entry speed, which is silently wrong rather than obviously broken.
    clip = load_csv_clip(p, fps=50.0)
    c = compute_conditions(to_38d(clip), fps=50.0)
    cells.add(_cell(
      float(c[0] * COND_SCALE[0]) * 57.2957795,
      float(c[1] * COND_SCALE[1]),
      abs(float(c[6] * COND_SCALE[6])) * 57.2957795,
      float(c[4] * COND_SCALE[4]) * 57.2957795,
    ))
  return cells, len(paths), len(paths)


def main(cfg: Cfg) -> None:
  anchor_cells, n_anchor = _anchor_cells(cfg.anchor_dir)
  gen_cells, total, passed = _generated_cells(cfg.generated_dir)
  new_cells = gen_cells - anchor_cells
  union = anchor_cells | gen_cells
  report = {
    "grid_steps": {"amp_deg": AMP_STEP, "speed_deg_s": SPEED_STEP,
                   "abd_deg": ABD_STEP, "knee_deg": KNEE_STEP},
    "anchors": {"files": n_anchor, "cells": len(anchor_cells)},
    "generated": {"samples": total, "passed": passed, "cells": len(gen_cells)},
    "union_cells": len(union),
    "marginal_cells_added_by_generation": len(new_cells),
    "marginal_fraction": round(len(new_cells) / max(1, len(gen_cells)), 3),
    "margin_examples": sorted(new_cells)[:20],
  }
  Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
  Path(cfg.out).write_text(json.dumps(report, indent=1))
  print(json.dumps(report, indent=1))


if __name__ == "__main__":
  tyro.cli(main)