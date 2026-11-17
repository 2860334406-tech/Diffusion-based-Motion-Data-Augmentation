"""Audit a frozen chest actor mapping on fresh paired EXP-022 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from replay_exp022_actor_force_inputs import replay, run_metrics
from smp.force_estimation.exp022_proprio import Exp022ProprioModel


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
  return {
    "episodes": len(rows),
    "precontact_false_active_episodes": sum(
      row["precontact_false_active_frames"] > 0 for row in rows
    ),
    "oracle_active_missed_fraction": sum(
      row["oracle_active_missed_frames"] for row in rows
    )
    / sum(row["oracle_active_frames"] for row in rows),
    **{
      key: float(np.median([row[key] for row in rows if row[key] is not None]))
      for key in (
        "oracle_active_force_obs_mae_n",
        "both_active_direction_mean_deg",
        "both_active_point_mean_m",
        "oracle_active_frame_l2_mean",
        "oracle_active_history_l2_mean",
      )
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--plan-file", default="policy_mapping_acceptance_pretest.json")
  parser.add_argument("--mapping-file", default="policy_mapping_calibration.json")
  parser.add_argument("--candidate-dir", default="candidate_both_v2")
  parser.add_argument("--capture-split", default="mapping_test")
  parser.add_argument("--baseline-dir", default="mapping_test_v392_predictions")
  args = parser.parse_args()
  root = args.root
  plan_path = root / args.plan_file
  plan = json.loads(plan_path.read_text())
  calibration_file = root / args.mapping_file
  model_file = root / args.candidate_dir / "candidate.joblib"
  if sha256(calibration_file) != plan["mapping_calibration_sha256"]:
    raise ValueError("Mapping calibration changed after test lock")
  if sha256(model_file) != plan["force_model_sha256"]:
    raise ValueError("Force model changed after test lock")
  calibration = json.loads(calibration_file.read_text())
  point_body = np.asarray(calibration["point_body_m"], np.float32)
  normal_body = np.asarray(calibration["normal_direction_body"], np.float32)
  model = Exp022ProprioModel.load(model_file)
  first_seed, last_seed = plan["fresh_seeds"]
  episodes = [
    f"{seed}_{mode}"
    for seed in range(first_seed, last_seed + 1)
    for mode in ("oracle", "masked")
  ]
  data = {"mapped": [], "old_point": []}
  for name in episodes:
    source = root / args.capture_split / name
    audit = json.loads((source / "audit.json").read_text())
    if audit["contact_frames"] == 0 or audit["loaded_ge10n_frames"] == 0:
      raise ValueError(f"Fresh episode needs a quality replacement: {name}")
    with np.load(source / "shadow_inputs.npz", allow_pickle=False) as z:
      force, _ = model.predict_episode(z["x"], z["tau_hist"])
      step = z["step"]
      oracle_frame = z["oracle_contact_frame"]
      oracle_history = z["oracle_history"]
      contact = z["contact_geometries"] != ""
      pose = {
        key: z[key]
        for key in (
          "torso_quat_wxyz",
          "torso_pos_w",
          "root_quat_wxyz",
          "root_pos_w",
        )
      }
    with np.load(root / args.baseline_dir / f"{name}.npz", allow_pickle=False) as z:
      if not np.array_equal(step, z["step"]):
        raise ValueError(f"V3.92 prediction step mismatch: {name}")
      old_point = z["estimated_point_body_m"]
    precontact = np.arange(len(step)) < int(np.flatnonzero(contact)[0])
    point_options = {
      "mapped": np.broadcast_to(point_body, force.shape),
      "old_point": old_point,
    }
    for version, point in point_options.items():
      mapped_force = (
        np.linalg.norm(force, axis=1)[:, None] * normal_body[None]
        if version == "mapped"
        else force
      ).astype(np.float32)
      frame, history = replay(mapped_force, point, pose, audit["robot_weight_n"])
      data[version].append(
        {
          "episode": name,
          **run_metrics(
            oracle_frame,
            oracle_history,
            frame,
            history,
            precontact,
            audit["robot_weight_n"],
          ),
        }
      )
  summary = {version: summarize(rows) for version, rows in data.items()}
  mapped = summary["mapped"]
  old = summary["old_point"]
  thresholds = plan["thresholds"]
  checks = {
    "episode_count": mapped["episodes"] == 2 * plan["expected_episode_pairs"],
    "frame_l2": mapped["oracle_active_frame_l2_mean"]
    <= thresholds["median_run_actor_frame_l2_max"],
    "history_l2": mapped["oracle_active_history_l2_mean"]
    <= thresholds["median_run_actor_history_l2_max"],
    "direction": mapped["both_active_direction_mean_deg"]
    <= thresholds["median_run_direction_deg_max"],
    "point": mapped["both_active_point_mean_m"] <= thresholds["median_run_point_m_max"],
    "force_obs": mapped["oracle_active_force_obs_mae_n"]
    <= thresholds["median_run_force_obs_mae_n_max"],
    "precontact": mapped["precontact_false_active_episodes"]
    <= thresholds["precontact_false_active_episodes_max"],
    "active_miss": mapped["oracle_active_missed_fraction"]
    <= thresholds["oracle_active_missed_fraction_max"],
    "frame_l2_reduction": 1
    - mapped["oracle_active_frame_l2_mean"] / old["oracle_active_frame_l2_mean"]
    >= thresholds["frame_l2_reduction_vs_v392_hybrid_min"],
  }
  result = {
    "summary": summary,
    "checks": {key: bool(value) for key, value in checks.items()},
    "pass_for_mainline_step1": all(checks.values()),
    "per_episode": data,
    "acceptance_pretest_sha256": sha256(plan_path),
  }
  if args.out.exists():
    raise FileExistsError(args.out)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(
    json.dumps(
      {key: result[key] for key in ("summary", "checks", "pass_for_mainline_step1")},
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
