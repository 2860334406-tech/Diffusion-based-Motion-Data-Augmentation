"""Check complete-run split, force labels and paired EXP-022 contact timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collect_exp022_estimator_adaptation import SPLITS, condition


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--split", choices=SPLITS, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  split_dir = args.root / args.split
  expected = {
    f"{seed}_{mode}" for seed in SPLITS[args.split] for mode in ("oracle", "masked")
  }
  actual = {path.name for path in split_dir.iterdir() if path.is_dir()}
  if actual != expected:
    raise ValueError(
      f"Split incomplete: missing={expected - actual}, extra={actual - expected}"
    )
  rows = []
  paired_onset_differences = []
  for seed in SPLITS[args.split]:
    pair = []
    for mode in ("oracle", "masked"):
      path = split_dir / f"{seed}_{mode}"
      audit = json.loads((path / "audit.json").read_text())
      with np.load(path / "shadow_inputs.npz", allow_pickle=False) as z:
        n = len(z["step"])
        if (
          z["x"].shape != (n, 930)
          or z["tau_hist"].shape != (n, 10, 29)
          or z["true_force_body_n"].shape != (n, 3)
          or not np.array_equal(z["step"], np.arange(1, n + 1))
          or not np.isfinite(z["x"]).all()
          or not np.isfinite(z["tau_hist"]).all()
          or not np.isfinite(z["true_force_body_n"]).all()
        ):
          raise ValueError(f"Invalid capture array: {path}")
        contact = np.flatnonzero(z["contact_geometries"] != "")
        onset = int(contact[0] + 1) if len(contact) else None
        first = int(contact[0]) if len(contact) else n
        if np.linalg.norm(z["true_force_body_n"][:first], axis=1).max(initial=0) > 1e-4:
          raise ValueError(f"Nonzero force before first audited collision: {path}")
        loaded = int(
          np.count_nonzero(np.linalg.norm(z["true_force_body_n"], axis=1) >= 10)
        )
      if (
        audit["seed"] != seed
        or audit["target_force_n"] != condition(seed)["target_force_n"]
        or audit["sphere_mass_kg"] != condition(seed)["sphere_mass_kg"]
        or audit["frames"] != n
      ):
        raise ValueError(f"Design mismatch: {path}")
      rows.append(
        {
          "episode": path.name,
          "seed": seed,
          "mode": mode,
          "target_force_n": audit["target_force_n"],
          "sphere_mass_kg": audit["sphere_mass_kg"],
          "frames": n,
          "first_contact_step": onset,
          "loaded_frames": loaded,
          "force_peak_n": audit["force_peak_20ms_n"],
          "gpu_peak_mib": audit["peak_own_gpu_mib"],
        }
      )
      pair.append(onset)
    if pair[0] is not None and pair[1] is not None:
      difference = abs(pair[0] - pair[1])
      paired_onset_differences.append(difference)
      tolerance = 1 if args.split in ("mapping_test", "v2_validation", "v2_test") else 0
      if difference > tolerance:
        raise ValueError(f"Paired contact onset mismatch for seed {seed}: {pair}")
    elif pair[0] != pair[1]:
      raise ValueError(f"Only one mode contacted for seed {seed}: {pair}")
  report = {
    "split": args.split,
    "episodes": len(rows),
    "seeds": len(SPLITS[args.split]),
    "no_contact_episodes": sum(row["first_contact_step"] is None for row in rows),
    "no_loaded_episodes": sum(row["loaded_frames"] == 0 for row in rows),
    "max_own_gpu_mib": max(row["gpu_peak_mib"] for row in rows),
    "max_paired_onset_difference_frames": max(paired_onset_differences, default=0),
    "runs": rows,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({key: report[key] for key in report if key != "runs"}, indent=2))


if __name__ == "__main__":
  main()
