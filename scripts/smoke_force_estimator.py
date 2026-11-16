"""Verify the bundled force estimator and optionally replay one held-out episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smp.force_estimation import load_force_estimator


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--version", choices=("v357", "v382", "v392"), default="v392")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--cache", type=Path)
  parser.add_argument("--predictions", type=Path)
  parser.add_argument("--episode", help="Episode prefix without its final :step suffix")
  args = parser.parse_args()
  if (args.cache is None) != (args.predictions is None):
    parser.error("--cache and --predictions must be supplied together")

  model = load_force_estimator(version=args.version, device=args.device)
  print(
    f"Loaded {args.version}: {len(model.manifest['sha256'])} verified artifacts on {args.device}"
  )
  if args.cache is None:
    return

  with (
    np.load(args.cache, allow_pickle=False) as cache,
    np.load(args.predictions, allow_pickle=False) as expected,
  ):
    if not np.array_equal(cache["run_id"], expected["run_id"]):
      raise ValueError("Cache and predictions are not aligned")
    episodes = np.asarray(
      [item.rsplit(":", 1)[0] for item in cache["run_id"].astype(str)]
    )
    name = args.episode or str(episodes[0])
    indices = np.flatnonzero(episodes == name)
    if not len(indices):
      raise ValueError(f"Episode not present: {name}")
    model.reset()
    forces, points, probabilities = [], [], []
    for index in indices:
      output = model.step(
        cache["x"][index],
        cache["tau_hist"][index],
        target_site_id=int(cache["site_id"][index]),
      )
      forces.append(output["force_body_n"])
      points.append(output["point_body_m"])
      probabilities.append(output["contact_probability"])
    force_diff = np.max(
      np.abs(np.asarray(forces) - expected["prediction_force_body_n"][indices])
    )
    point_diff = np.max(
      np.abs(np.asarray(points) - expected["prediction_point_body_m"][indices])
    )
    probability_diff = np.max(
      np.abs(np.asarray(probabilities) - expected["contact_probability"][indices])
    )
    print(f"Episode {name}: {len(indices)} frames")
    print(
      f"Max difference: force {force_diff:.6g} N, point {point_diff:.6g} m, probability {probability_diff:.6g}"
    )
    if force_diff > 1e-3 or point_diff > 1e-5 or probability_diff > 1e-4:
      raise SystemExit("Frozen prediction parity failed")


if __name__ == "__main__":
  main()
