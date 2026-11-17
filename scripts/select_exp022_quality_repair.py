"""Select a predeclared same-condition replacement using physics quality only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collect_exp022_estimator_adaptation import condition


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_quality(root: Path, split: str, seed: int) -> tuple[bool, dict]:
  rows = []
  for mode in ("oracle", "masked"):
    path = root / split / f"{seed}_{mode}"
    if not (path / "audit.json").exists():
      return False, {"seed": seed, "missing": f"{seed}_{mode}"}
    audit = json.loads((path / "audit.json").read_text())
    with np.load(path / "shadow_inputs.npz", allow_pickle=False) as z:
      contact = np.flatnonzero(z["contact_geometries"] != "")
      force = np.linalg.norm(z["true_force_body_n"], axis=1)
      loaded = np.flatnonzero(force >= 10)
    qualifies = bool(
      len(contact)
      and len(loaded)
      and audit["target_force_n"] == condition(seed)["target_force_n"]
      and audit["sphere_mass_kg"] == condition(seed)["sphere_mass_kg"]
    )
    rows.append(
      {
        "episode": f"{seed}_{mode}",
        "qualified": qualifies,
        "first_contact_frame": int(contact[0] + 1) if len(contact) else None,
        "first_loaded_frame": int(loaded[0] + 1) if len(loaded) else None,
        "audit_sha256": sha256(path / "audit.json"),
        "capture_sha256": sha256(path / "shadow_inputs.npz"),
      }
    )
  return bool(all(row["qualified"] for row in rows)), {"seed": seed, "runs": rows}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  args = parser.parse_args()
  root = args.root
  plan_file = root / "quality_repair_pretest.json"
  plan = json.loads(plan_file.read_text())
  if sha256(root / "test_audit.json") != plan["original_test_audit_sha256"]:
    raise ValueError("Original test audit changed")
  if sha256(root / "acceptance_pretest.json") != plan["acceptance_pretest_sha256"]:
    raise ValueError("Pretest criteria changed")
  if (
    sha256(root / "candidate_both_v2/candidate.joblib")
    != plan["frozen_candidate_model_sha256"]
  ):
    raise ValueError("Frozen model changed")
  retained = []
  for seed in plan["retained_seeds"]:
    good, detail = pair_quality(root, "test", seed)
    if not good:
      raise ValueError(f"Retained seed failed physics quality: {detail}")
    retained.append(detail)
  failures = []
  chosen = None
  for seed in plan["same_condition_replacement_order"]:
    good, detail = pair_quality(root, "repair", seed)
    if good:
      chosen = detail
      break
    failures.append(detail)
  if chosen is None:
    raise RuntimeError(f"No quality-passing replacement yet: {failures}")
  episodes = [
    f"{seed}_{mode}"
    for seed in (*plan["retained_seeds"], chosen["seed"])
    for mode in ("oracle", "masked")
  ]
  result = {
    "quality_plan_sha256": sha256(plan_file),
    "excluded_raw_episodes": plan["excluded_raw_episodes"],
    "replacement_seed": chosen["seed"],
    "selected_episodes": episodes,
    "retained_quality": retained,
    "replacement_quality": chosen,
    "earlier_replacement_failures": failures,
    "model_predictions_inspected_for_selection": False,
  }
  output = root / "quality_selection.json"
  if output.exists():
    raise FileExistsError(output)
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(
    json.dumps(
      {key: result[key] for key in ("replacement_seed", "selected_episodes")}, indent=2
    )
  )


if __name__ == "__main__":
  main()
