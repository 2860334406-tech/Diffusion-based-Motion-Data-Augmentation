"""Evaluate frozen EXP-022 V2 force acceptance against V3.92 on the same runs."""

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
  plan_file = root / "v2_acceptance_pretest.json"
  plan = json.loads(plan_file.read_text())
  candidate = root / "candidate_v2"
  if sha256(candidate / "manifest.json") != plan["candidate_manifest_sha256"]:
    raise ValueError("V2 candidate manifest changed after test lock")
  if sha256(candidate / "candidate.joblib") != plan["force_model_sha256"]:
    raise ValueError("V2 force model changed after test lock")
  candidate_result = json.loads((root / "v2_test_frozen/metrics.json").read_text())
  if candidate_result["model_sha256"] != plan["force_model_sha256"]:
    raise ValueError("V2 test used a different force model")
  first_seed, last_seed = plan["fresh_seeds"]
  expected = {
    f"{seed}_{mode}"
    for seed in range(first_seed, last_seed + 1)
    for mode in ("oracle", "masked")
  }
  candidate_rows = candidate_result["per_episode"]
  if {row["episode"] for row in candidate_rows} != expected:
    raise ValueError("V2 locked test episode set differs from pretest plan")
  baseline_rows = []
  for name in sorted(expected):
    with np.load(
      root / "v2_test" / name / "shadow_inputs.npz", allow_pickle=False
    ) as z:
      truth = z["true_force_body_n"]
      step = z["step"]
      geometry = z["contact_geometries"]
    with np.load(
      root / "v2_test_v392_predictions" / f"{name}.npz", allow_pickle=False
    ) as z:
      if not np.array_equal(step, z["step"]):
        raise ValueError(f"Baseline step mismatch: {name}")
      estimate = z["estimated_force_body_n"]
    true_n = np.linalg.norm(truth, axis=1)
    estimate_n = np.linalg.norm(estimate, axis=1)
    first_contact = int(np.flatnonzero(geometry != "")[0])
    pre = np.arange(len(step)) < first_contact
    loaded = true_n >= 10
    peak = int(np.argmax(true_n))
    baseline_rows.append(
      {
        "episode": name,
        "precontact_max_est_n": float(estimate_n[pre].max(initial=0)),
        "loaded_vector_mae_n": float(
          np.linalg.norm(truth[loaded] - estimate[loaded], axis=1).mean()
        ),
        "true_peak_magnitude_abs_error_n": float(abs(true_n[peak] - estimate_n[peak])),
      }
    )
  baseline = {
    "precontact_worst_n": max(row["precontact_max_est_n"] for row in baseline_rows),
    "median_run_loaded_vector_mae_n": float(
      np.median([row["loaded_vector_mae_n"] for row in baseline_rows])
    ),
    "worst_run_loaded_vector_mae_n": max(
      row["loaded_vector_mae_n"] for row in baseline_rows
    ),
  }
  selected = candidate_result["summary"]
  limits = plan["thresholds"]
  checks = {
    "episode_count": selected["episodes"] == len(expected),
    "precontact_actor": selected["precontact_actor_active_episodes"]
    <= limits["precontact_false_active_episodes_max"],
    "precontact_force": selected["precontact_worst_n"]
    <= limits["force_precontact_max_n_max"],
    "median_vector": selected["median_run_vector_mae_n"]
    <= limits["force_median_run_loaded_vector_mae_n_max"],
    "worst_run_vector": selected["worst_run_vector_mae_n"]
    <= limits["force_worst_run_loaded_vector_mae_n_max"],
    "first_loaded_misses": selected["first_loaded_misses"]
    <= limits["force_first_loaded_misses_max"],
    "median_peak_error": selected["median_peak_magnitude_abs_error_n"]
    <= limits["force_median_peak_magnitude_abs_error_n_max"],
    "median_direction": selected["median_detected_direction_deg"]
    <= limits["force_median_detected_direction_deg_max"],
    "vector_improvement": 1
    - selected["median_run_vector_mae_n"] / baseline["median_run_loaded_vector_mae_n"]
    >= plan["baseline_comparison"]["force_median_run_loaded_vector_mae_reduction_min"],
    "precontact_improvement": 1
    - selected["precontact_worst_n"] / baseline["precontact_worst_n"]
    >= plan["baseline_comparison"]["force_precontact_worst_reduction_min"],
  }
  result = {
    "candidate_summary": selected,
    "baseline_summary": baseline,
    "baseline_per_episode": baseline_rows,
    "checks": {key: bool(value) for key, value in checks.items()},
    "pass_force_gate": all(checks.values()),
    "pretest_plan_sha256": sha256(plan_file),
  }
  if args.out.exists():
    raise FileExistsError(args.out)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(
    json.dumps(
      {
        key: result[key]
        for key in (
          "candidate_summary",
          "baseline_summary",
          "checks",
          "pass_force_gate",
        )
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
