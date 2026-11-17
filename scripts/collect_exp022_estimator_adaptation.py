"""Collect split-by-seed EXP-022 force labels without changing the actor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SPLITS = {
  "train": range(61, 109),
  "validation": range(109, 121),
  "test": range(121, 133),
  "mapping_test": range(145, 157),
  "v2_validation": range(157, 169),
  "v2_test": range(169, 181),
}
FORCES_N = (10.0, 20.0, 30.0, 40.0)
MASSES_KG = (1.0, 5.0)


def condition(seed: int) -> dict[str, float]:
  index = (seed - 61) % 8
  return {
    "target_force_n": FORCES_N[index % 4],
    "sphere_mass_kg": MASSES_KG[index // 4],
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--split", choices=(*SPLITS, "repair"), required=True)
  parser.add_argument("--seed", type=int)
  parser.add_argument("--root", type=Path, required=True)
  args = parser.parse_args()
  repo = Path(__file__).resolve().parents[1]
  root = args.root.resolve()
  root.mkdir(parents=True, exist_ok=True)
  design = {
    "train_seeds": [61, 108],
    "validation_seeds": [109, 120],
    "locked_test_seeds": [121, 132],
    "target_forces_n": FORCES_N,
    "sphere_masses_kg": MASSES_KG,
    "condition_rule": "index=(seed-61)%8; force=FORCES[index%4]; mass=MASSES[index//4]",
    "modes": ["oracle", "masked"],
    "pusher": {
      "sphere_radius_m": 0.1,
      "max_drive_force_weight_ratio": 0.1,
      "target_speed_m_s": 0.06,
      "follow_speed_m_s": 0.4,
      "base_force_gain_m_s_per_n": 0,
      "motor_hold_after_contact": True,
      "freeze_push_heading": True,
      "one_way_speed_limit": True,
      "contact_time_constant_s": 0.05,
    },
  }
  design_file = root / "design.json"
  encoded = json.dumps(design, indent=2) + "\n"
  if design_file.exists() and design_file.read_text() != encoded:
    raise RuntimeError("Dataset design changed after collection began")
  design_file.write_text(encoded)
  if args.split == "repair":
    plan = json.loads((root / "quality_repair_pretest.json").read_text())
    if args.seed not in plan["same_condition_replacement_order"]:
      raise ValueError("Repair seed must be in the frozen quality plan")
    seeds = (args.seed,)
  else:
    if args.seed is not None:
      raise ValueError("--seed is only valid for quality repair")
    seeds = SPLITS[args.split]
  env = os.environ.copy()
  env["PYTHONPATH"] = f"{repo / 'src'}:{repo}"
  for seed in seeds:
    conditions = condition(seed)
    for mode in ("oracle", "masked"):
      name = f"{seed}_{mode}"
      output = root / args.split / name
      audit_file = output / "audit.json"
      capture_file = output / "shadow_inputs.npz"
      if audit_file.exists() and capture_file.exists():
        audit = json.loads(audit_file.read_text())
        if (
          audit["seed"] != seed
          or audit["target_force_n"] != conditions["target_force_n"]
          or audit["sphere_mass_kg"] != conditions["sphere_mass_kg"]
        ):
          raise RuntimeError(f"Existing run mismatches design: {output}")
        print(f"SKIP {args.split} {name}", flush=True)
        continue
      if output.exists():
        raise RuntimeError(f"Incomplete existing output: {output}")
      output.mkdir(parents=True)
      cmd = [
        sys.executable,
        str(repo / "scripts/evaluate_exp022_sustained_contact.py"),
        "--out-dir",
        str(output),
        "--seed",
        str(seed),
        "--target-force-n",
        str(conditions["target_force_n"]),
        "--sphere-radius-m",
        "0.1",
        "--sphere-mass-kg",
        str(conditions["sphere_mass_kg"]),
        "--max-drive-force-weight-ratio",
        "0.1",
        "--target-speed-m-s",
        "0.06",
        "--follow-speed-m-s",
        "0.4",
        "--base-force-gain-m-s-per-n",
        "0",
        "--motor-hold-after-contact",
        "--freeze-push-heading",
        "--one-way-speed-limit",
        "--contact-time-constant-s",
        "0.05",
        "--capture-shadow-inputs",
        "--no-video",
        "--max-gpu-memory-gib",
        "4",
        "--min-free-gpu-gib",
        "2",
      ]
      if mode == "masked":
        cmd.append("--mask-actor-rail-contact")
      with (output / "collection.log").open("w") as log:
        result = subprocess.run(
          cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
        )
      if result.returncode:
        raise RuntimeError(f"Collection failed for {name}: {output / 'collection.log'}")
      audit = json.loads(audit_file.read_text())
      print(
        f"DONE {args.split} {name} frames={audit['frames']} "
        f"peak={audit['force_peak_20ms_n']:.1f}N "
        f"duration={audit['longest_torso_loaded_ge10n_s']:.2f}s "
        f"gpu={audit['peak_own_gpu_mib']}MiB",
        flush=True,
      )


if __name__ == "__main__":
  main()
