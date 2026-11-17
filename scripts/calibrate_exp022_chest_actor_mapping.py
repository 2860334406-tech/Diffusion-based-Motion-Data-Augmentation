"""Calibrate chest-only policy normal/point from train runs and select on validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from replay_exp022_actor_force_inputs import replay, run_metrics
from smp.force_estimation.exp022_proprio import Exp022ProprioModel


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def body_targets(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
  oracle = z["oracle_contact_frame"]
  valid = (np.linalg.norm(z["true_force_body_n"], axis=1) >= 10) & (oracle[:, 0] > 0.5)
  if not valid.any():
    raise ValueError("No active force label for chest mapping")
  root_q = torch.from_numpy(z["root_quat_wxyz"][valid])
  torso_q = torch.from_numpy(z["torso_quat_wxyz"][valid])
  point_world = torch.from_numpy(z["root_pos_w"][valid]) + quat_apply(
    yaw_quat(root_q), torch.from_numpy(oracle[valid, 5:8])
  )
  point_body = quat_apply_inverse(
    torso_q, point_world - torch.from_numpy(z["torso_pos_w"][valid])
  ).numpy()
  direction_body = quat_apply_inverse(
    torso_q,
    quat_apply(yaw_quat(root_q), torch.from_numpy(oracle[valid, 2:5])),
  ).numpy()
  return point_body, direction_body


def normalized(vector: np.ndarray) -> np.ndarray:
  return vector / np.linalg.norm(vector)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  root = args.root
  model_file = root / "candidate_both_v2/candidate.joblib"
  manifest = json.loads((root / "candidate_both_v2/manifest.json").read_text())
  if sha256(model_file) != manifest["model_sha256"]:
    raise ValueError("Frozen force model changed")
  episode_points = []
  episode_directions = []
  all_points = []
  all_directions = []
  train_files = sorted((root / "train").glob("*/shadow_inputs.npz"))
  if len(train_files) != 96:
    raise ValueError("Expected 96 complete train episodes")
  for source in train_files:
    with np.load(source, allow_pickle=False) as z:
      points, directions = body_targets(z)
    all_points.append(points)
    all_directions.append(directions)
    episode_points.append(np.median(points, axis=0))
    episode_directions.append(normalized(directions.mean(axis=0)))
  candidates = {
    "frame_pooled": {
      "point_body_m": np.median(np.concatenate(all_points), axis=0),
      "normal_direction_body": normalized(np.concatenate(all_directions).mean(axis=0)),
    },
    "episode_balanced": {
      "point_body_m": np.median(np.stack(episode_points), axis=0),
      "normal_direction_body": normalized(np.stack(episode_directions).mean(axis=0)),
    },
  }
  model = Exp022ProprioModel.load(model_file)
  validation = sorted((root / "validation").glob("*/shadow_inputs.npz"))
  if len(validation) != 24:
    raise ValueError("Expected 24 complete validation episodes")
  scored = {}
  for key, constants in candidates.items():
    rows = []
    for source in validation:
      with np.load(source, allow_pickle=False) as z:
        force, _ = model.predict_episode(z["x"], z["tau_hist"])
        magnitude = np.linalg.norm(force, axis=1)
        mapped_force = (
          magnitude[:, None] * constants["normal_direction_body"][None]
        ).astype(np.float32)
        point = np.broadcast_to(
          constants["point_body_m"].astype(np.float32), force.shape
        )
        pose = {
          name: z[name]
          for name in (
            "torso_quat_wxyz",
            "torso_pos_w",
            "root_quat_wxyz",
            "root_pos_w",
          )
        }
        oracle = z["oracle_contact_frame"]
        history = z["oracle_history"]
        contact = z["contact_geometries"] != ""
      audit = json.loads((source.parent / "audit.json").read_text())
      frame, mapped_history = replay(mapped_force, point, pose, audit["robot_weight_n"])
      precontact = np.arange(len(force)) < int(np.flatnonzero(contact)[0])
      rows.append(
        {
          "episode": source.parent.name,
          **run_metrics(
            oracle, history, frame, mapped_history, precontact, audit["robot_weight_n"]
          ),
        }
      )
    score = float(np.median([row["oracle_active_frame_l2_mean"] for row in rows]))
    scored[key] = {
      "median_run_frame_l2": score,
      "median_run_history_l2": float(
        np.median([row["oracle_active_history_l2_mean"] for row in rows])
      ),
      "median_run_direction_deg": float(
        np.median([row["both_active_direction_mean_deg"] for row in rows])
      ),
      "median_run_point_m": float(
        np.median([row["both_active_point_mean_m"] for row in rows])
      ),
      "per_episode": rows,
    }
  selected = min(scored, key=lambda key: scored[key]["median_run_frame_l2"])
  result = {
    "name": "EXP022 chest-only policy contact observation mapping",
    "status": "frozen before fresh mapping test",
    "force_model_sha256": manifest["model_sha256"],
    "train_episodes": [path.parent.name for path in train_files],
    "validation_episodes": [path.parent.name for path in validation],
    "selected": selected,
    "point_body_m": candidates[selected]["point_body_m"].tolist(),
    "normal_direction_body": candidates[selected]["normal_direction_body"].tolist(),
    "candidates": {
      key: {
        "point_body_m": value["point_body_m"].tolist(),
        "normal_direction_body": value["normal_direction_body"].tolist(),
        "validation_median_run_frame_l2": scored[key]["median_run_frame_l2"],
        "validation_median_run_history_l2": scored[key]["median_run_history_l2"],
        "validation_median_run_direction_deg": scored[key]["median_run_direction_deg"],
        "validation_median_run_point_m": scored[key]["median_run_point_m"],
      }
      for key, value in candidates.items()
    },
    "inference_contract": "Only estimated force magnitude, frozen body-frame normal/point, and current torso/root poses; no solver labels or future states",
    "limits": "Front chest sphere push under this EXP022 policy; mapping does not estimate arbitrary contact point or full 3D resultant direction",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  if args.out.exists():
    raise FileExistsError(args.out)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  (args.out.parent / "policy_mapping_validation.json").write_text(
    json.dumps(scored, indent=2) + "\n"
  )
  print(json.dumps({"selected": selected, **result["candidates"][selected]}, indent=2))


if __name__ == "__main__":
  main()
