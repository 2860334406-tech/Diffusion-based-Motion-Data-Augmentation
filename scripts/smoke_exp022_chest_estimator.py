"""Verify the EXP-022 chest bundle and optionally replay a saved episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smp.force_estimation import load_exp022_chest_estimator


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-dir", type=Path)
  parser.add_argument("--capture", type=Path)
  parser.add_argument("--predictions", type=Path)
  args = parser.parse_args()
  if (args.capture is None) != (args.predictions is None):
    raise ValueError("Provide both --capture and --predictions for replay")
  model = (
    load_exp022_chest_estimator(args.model_dir)
    if args.model_dir is not None
    else load_exp022_chest_estimator()
  )
  report = {"bundle_loaded": True, "model_version": "exp022_chest_v1"}
  if args.capture is not None:
    with np.load(args.capture, allow_pickle=False) as z:
      x = z["x"]
      tau = z["tau_hist"]
      step = z["step"]
    with np.load(args.predictions, allow_pickle=False) as z:
      if not np.array_equal(step, z["step"]):
        raise ValueError("Capture and prediction steps differ")
      reference = z["estimated_force_body_n"]
      reference_probability = z["contact_probability"]
    model.reset()
    output = [model.step(x[i], tau[i]) for i in range(len(x))]
    force = np.stack([row["force_body_n"] for row in output])
    probability = np.asarray([row["contact_probability"] for row in output])
    max_force_difference = float(np.max(np.abs(force - reference)))
    max_probability_difference = float(
      np.max(np.abs(probability - reference_probability))
    )
    if max_force_difference > 1e-4 or max_probability_difference > 1e-5:
      raise ValueError("Repository bundle does not reproduce frozen predictions")
    report.update(
      frames=len(x),
      max_force_difference_n=max_force_difference,
      max_contact_probability_difference=max_probability_difference,
    )
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
