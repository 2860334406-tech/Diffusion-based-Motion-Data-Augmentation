"""Compare a frozen EXP-022 force candidate with frozen V3.92 on locked runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  root = args.root
  acceptance = json.loads((root / "acceptance_pretest.json").read_text())
  candidate = root / "candidate_both_v2"
  if (
    sha256(candidate / "manifest.json")
    != acceptance["frozen_candidate_manifest_sha256"]
  ):
    raise ValueError("Candidate manifest changed after test lock")
  if (
    sha256(candidate / "candidate.joblib")
    != acceptance["frozen_candidate_model_sha256"]
  ):
    raise ValueError("Candidate weights changed after test lock")
  candidate_metrics = json.loads((root / "test_frozen/metrics.json").read_text())
  if candidate_metrics["model_sha256"] != acceptance["frozen_candidate_model_sha256"]:
    raise ValueError("Candidate test metrics used different weights")
  selected = candidate_metrics["per_episode"]
  quality = json.loads((root / "quality_selection.json").read_text())
  expected_names = set(quality["selected_episodes"])
  if len(expected_names) != acceptance["expected_episodes"]:
    raise ValueError("Quality repair did not preserve test episode count")
  if {row["episode"] for row in selected} != expected_names:
    raise ValueError("Candidate test set is incomplete")
  baseline_rows = []
  for name in sorted(expected_names):
    with np.load(
      root / "test_selected" / name / "shadow_inputs.npz", allow_pickle=False
    ) as z:
      step = z["step"]
      truth = z["true_force_body_n"]
      geometries = z["contact_geometries"]
    with np.load(
      root / "test_v392_predictions" / f"{name}.npz", allow_pickle=False
    ) as z:
      if not np.array_equal(step, z["step"]):
        raise ValueError(f"Step mismatch: {name}")
      force = z["estimated_force_body_n"]
    true_n = np.linalg.norm(truth, axis=1)
    estimate_n = np.linalg.norm(force, axis=1)
    first = np.flatnonzero(geometries != "")
    precontact = np.arange(len(step)) < (int(first[0]) if len(first) else len(step))
    loaded = true_n >= 10
    peak = int(np.argmax(true_n))
    baseline_rows.append(
      {
        "episode": name,
        "precontact_max_est_n": float(estimate_n[precontact].max(initial=0)),
        "precontact_fp10_frames": int(np.count_nonzero(estimate_n[precontact] >= 10)),
        "loaded_vector_mae_n": float(
          np.linalg.norm(truth[loaded] - force[loaded], axis=1).mean()
        ),
        "true_peak_magnitude_abs_error_n": float(abs(true_n[peak] - estimate_n[peak])),
      }
    )
  baseline = {
    "precontact_fp_episodes": sum(
      row["precontact_fp10_frames"] > 0 for row in baseline_rows
    ),
    "precontact_worst_n": max(row["precontact_max_est_n"] for row in baseline_rows),
    "median_run_loaded_vector_mae_n": float(
      np.median([row["loaded_vector_mae_n"] for row in baseline_rows])
    ),
    "worst_run_loaded_vector_mae_n": max(
      row["loaded_vector_mae_n"] for row in baseline_rows
    ),
    "median_peak_magnitude_abs_error_n": float(
      np.median([row["true_peak_magnitude_abs_error_n"] for row in baseline_rows])
    ),
  }
  summary = candidate_metrics["summary"]
  rows = selected
  thresholds = acceptance["thresholds"]
  checks = {
    "episode_count": len(rows) == acceptance["expected_episodes"],
    "precontact_fp_episodes": summary["precontact_fp_episodes"]
    <= thresholds["precontact_fp10_episodes_max"],
    "precontact_worst": summary["precontact_worst_n"]
    <= thresholds["precontact_worst_n_max"],
    "median_loaded_vector": summary["median_run_vector_mae_n"]
    <= thresholds["median_run_loaded_vector_mae_n_max"],
    "worst_run_loaded_vector": summary["worst_run_vector_mae_n"]
    <= thresholds["worst_run_loaded_vector_mae_n_max"],
    "worst_single_loaded_vector": summary["worst_single_loaded_vector_n"]
    <= thresholds["worst_single_loaded_vector_n_max"],
    "median_peak_magnitude": summary["median_peak_magnitude_abs_error_n"]
    <= thresholds["median_peak_magnitude_abs_error_n_max"],
    "p90_peak_magnitude": np.percentile(
      [row["true_peak_magnitude_abs_error_n"] for row in rows], 90
    )
    <= thresholds["p90_peak_magnitude_abs_error_n_max"],
    "median_direction": summary["median_detected_direction_deg"]
    <= thresholds["median_detected_direction_deg_max"],
    "first_loaded_misses": summary["first_loaded_misses"]
    <= thresholds["first_loaded_misses_max"],
    "median_first_loaded_delay": summary["median_first_loaded_delay_frames"]
    <= thresholds["median_first_loaded_delay_frames_max"],
    "loaded_miss_fraction": summary["loaded_miss10_frames"] / summary["loaded_frames"]
    <= thresholds["loaded_miss10_fraction_max"],
    "baseline_vector_reduction": 1
    - summary["median_run_vector_mae_n"] / baseline["median_run_loaded_vector_mae_n"]
    >= acceptance["baseline_comparison"]["median_run_loaded_vector_mae_reduction_min"],
    "baseline_precontact_reduction": 1
    - summary["precontact_worst_n"] / baseline["precontact_worst_n"]
    >= acceptance["baseline_comparison"]["precontact_worst_reduction_min"],
  }
  report = {
    "candidate_summary": summary,
    "candidate_p90_peak_magnitude_abs_error_n": float(
      np.percentile([row["true_peak_magnitude_abs_error_n"] for row in rows], 90)
    ),
    "baseline_summary": baseline,
    "baseline_per_episode": baseline_rows,
    "checks": {key: bool(value) for key, value in checks.items()},
    "pass_for_mainline_step1": all(checks.values()),
    "acceptance_pretest_sha256": sha256(root / "acceptance_pretest.json"),
    "quality_selection_sha256": sha256(root / "quality_selection.json"),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2) + "\n")
  print(
    json.dumps(
      {
        key: report[key]
        for key in (
          "candidate_summary",
          "baseline_summary",
          "checks",
          "pass_for_mainline_step1",
        )
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
