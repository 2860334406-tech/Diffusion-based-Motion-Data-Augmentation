"""Coverage report for augmented waist trajectories.

Maps the generated samples onto the project's "expected data" table, one row per
requirement, so we can see which bins are covered and which are still empty.
"""

from __future__ import annotations

import glob
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

# Expected-data table bins (degrees).
ABDUCTION_BINS = [(10, 20), (25, 40), (50, 60)]
KNEE_BINS = [(15, 22), (28, 38)]
WAIST_BINS = [(15, 30), (30, 45), (45, 60), (60, 70)]
WRIST_ROLL_MIN_SPREAD = 5.0   # a sample counts if its wrist roll moves >= 5 deg
WAIST_TILT_MAX = 3.0
ROOT_YAW_MAX = 5.0
HIP_YAW_MAX = 8.0


@dataclass
class Cfg:
  sample_dir: str = "logs/reproduction/waist_aug_v1/generated"
  out: str = "logs/reproduction/waist_aug_v1/coverage.json"


def _bin(value: float, bins: list[tuple[float, float]]) -> str | None:
  for lo, hi in bins:
    if lo <= value < hi:
      return f"{lo:g}-{hi:g}"
  return None


def main(cfg: Cfg) -> None:
  paths = sorted(glob.glob(str(Path(cfg.sample_dir) / "*" / "summary.json")))
  if not paths:
    raise FileNotFoundError(f"No summary.json under {cfg.sample_dir}")
  records: list[dict] = []
  for p in paths:
    records.extend(json.load(open(p))["records"])

  passed = [r for r in records if r.get("passed")]
  total = len(records)

  def _vals(key: str, rows: list[dict]) -> np.ndarray:
    return np.array([r[key] for r in rows if key in r], dtype=float)

  abd = _vals("abduction_deg", passed)
  knee = _vals("knee_deg", passed)
  amp = _vals("waist_amp_deg", passed)
  speed = _vals("speed_deg_s", passed)
  root_yaw = _vals("root_yaw_p95_deg", passed)
  hip_yaw = _vals("hip_yaw_p95_deg", passed)
  tilt = _vals("waist_pitch_p95_deg", passed)

  def _count_bins(v: np.ndarray, bins) -> dict[str, int]:
    out: Counter = Counter()
    for x in v:
      b = _bin(float(x), bins)
      out[b or "out"] += 1
    return dict(out)

  report = {
    "samples_total": total,
    "samples_passed": len(passed),
    "pass_rate": round(len(passed) / max(1, total), 3),
    "expected_table_checklist": {
      "1_shoulder_roll_abduction": {
        "bins": {f"{a}-{b}": 0 for a, b in ABDUCTION_BINS},
        "observed": _count_bins(abd, ABDUCTION_BINS),
        "covered_bins": sum(1 for a, b in ABDUCTION_BINS
                            if any(a <= x < b for x in abd)),
        "bins_total": len(ABDUCTION_BINS),
      },
      "2_6_knee_stance": {
        "bins": {f"{a}-{b}": 0 for a, b in KNEE_BINS},
        "observed": _count_bins(knee, KNEE_BINS),
        "covered_bins": sum(1 for a, b in KNEE_BINS
                            if any(a <= x < b for x in knee)),
        "bins_total": len(KNEE_BINS),
      },
      "10_waist_yaw_range": {
        "bins": {f"{a}-{b}": 0 for a, b in WAIST_BINS},
        "observed": _count_bins(amp, WAIST_BINS),
        "covered_bins": sum(1 for a, b in WAIST_BINS
                            if any(a <= x < b for x in amp)),
        "bins_total": len(WAIST_BINS),
      },
      "10_entry_speed_spread_deg_s": {
        "min": round(float(speed.min()), 2) if len(speed) else None,
        "median": round(float(np.median(speed)), 2) if len(speed) else None,
        "max": round(float(speed.max()), 2) if len(speed) else None,
      },
    },
    "constraint_observed": {
      "root_yaw_p95_max_deg": round(float(root_yaw.max()), 2) if len(root_yaw) else None,
      "hip_yaw_p95_max_deg": round(float(hip_yaw.max()), 2) if len(hip_yaw) else None,
      "waist_pitch_p95_max_deg": round(float(tilt.max()), 2) if len(tilt) else None,
      "root_yaw_threshold": ROOT_YAW_MAX,
      "hip_yaw_threshold": HIP_YAW_MAX,
      "waist_tilt_threshold": WAIST_TILT_MAX,
    },
    "not_covered_by_this_run": [
      "3_shoulder_pitch_elbow", "4_wrist_roll", "5_wrist_pitch_yaw",
      "7_hip_yaw_initial", "8_hip_roll_ankle_weight_shift",
      "9_waist_pitch_roll_excursion",
    ],
    "note": (
      "Rows 3-9 require per-joint spread statistics that this sampler does not "
      "report directly; compute them from the exported CSVs if needed."
    ),
  }
  Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
  Path(cfg.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
  print(json.dumps({k: v for k, v in report.items()
                    if k != "expected_table_checklist"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
  tyro.cli(main)