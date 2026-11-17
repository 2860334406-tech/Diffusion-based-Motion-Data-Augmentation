"""Replay frozen force estimates into EXP-022's 8D/32D actor contact term."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from smp.force_estimation.exp022_interface import Exp022ContactAdapter


def replay(
  force: np.ndarray, point: np.ndarray, pose: dict[str, np.ndarray], weight: float
) -> tuple[np.ndarray, np.ndarray]:
  adapter = Exp022ContactAdapter(1, "cpu", robot_weight_n=weight)
  frames = np.empty((len(force), 8), np.float32)
  histories = np.empty((len(force), 32), np.float32)
  for i in range(len(force)):
    frame = adapter.step(
      torch.from_numpy(force[i : i + 1]),
      torch.from_numpy(np.array(point[i : i + 1], copy=True)),
      torso_quat_w=torch.from_numpy(pose["torso_quat_wxyz"][i : i + 1]),
      torso_pos_w=torch.from_numpy(pose["torso_pos_w"][i : i + 1]),
      root_quat_w=torch.from_numpy(pose["root_quat_wxyz"][i : i + 1]),
      root_pos_w=torch.from_numpy(pose["root_pos_w"][i : i + 1]),
    )
    if not torch.allclose(frame[0], adapter.history[0, -8:], atol=1e-6):
      raise ValueError("Contact frame is not at the tail of actor history")
    frames[i] = frame[0].numpy()
    histories[i] = adapter.history.numpy()
  return frames, histories


def run_metrics(
  oracle_frame: np.ndarray,
  oracle_history: np.ndarray,
  estimated_frame: np.ndarray,
  estimated_history: np.ndarray,
  precontact: np.ndarray,
  weight: float,
) -> dict:
  oracle_active = oracle_frame[:, 0] > 0.5
  estimated_active = estimated_frame[:, 0] > 0.5
  both = oracle_active & estimated_active
  force_abs_n = np.abs(oracle_frame[:, 1] - estimated_frame[:, 1]) * weight
  direction_cos = np.sum(oracle_frame[both, 2:5] * estimated_frame[both, 2:5], axis=1)
  direction_deg = np.degrees(np.arccos(np.clip(direction_cos, -1, 1)))
  point_m = np.linalg.norm(oracle_frame[both, 5:8] - estimated_frame[both, 5:8], axis=1)
  return {
    "oracle_active_frames": int(oracle_active.sum()),
    "estimated_active_frames": int(estimated_active.sum()),
    "precontact_false_active_frames": int(
      np.count_nonzero(estimated_active & precontact)
    ),
    "oracle_active_missed_frames": int(
      np.count_nonzero(oracle_active & ~estimated_active)
    ),
    "oracle_active_force_obs_mae_n": float(force_abs_n[oracle_active].mean())
    if oracle_active.any()
    else None,
    "both_active_direction_mean_deg": float(direction_deg.mean())
    if len(direction_deg)
    else None,
    "both_active_point_mean_m": float(point_m.mean()) if len(point_m) else None,
    "oracle_active_frame_l2_mean": float(
      np.linalg.norm(
        oracle_frame[oracle_active] - estimated_frame[oracle_active], axis=1
      ).mean()
    )
    if oracle_active.any()
    else None,
    "oracle_active_history_l2_mean": float(
      np.linalg.norm(
        oracle_history[oracle_active] - estimated_history[oracle_active], axis=1
      ).mean()
    )
    if oracle_active.any()
    else None,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  root = args.root
  quality = json.loads((root / "quality_selection.json").read_text())
  rows = []
  traces = args.out.parent / "actor_input_traces"
  traces.mkdir(parents=True, exist_ok=True)
  for name in quality["selected_episodes"]:
    source = root / "test_selected" / name / "shadow_inputs.npz"
    with np.load(source, allow_pickle=False) as z:
      pose = {
        key: z[key]
        for key in (
          "torso_quat_wxyz",
          "torso_pos_w",
          "root_quat_wxyz",
          "root_pos_w",
        )
      }
      oracle_frame = z["oracle_contact_frame"]
      oracle_history = z["oracle_history"]
      step = z["step"]
      contact = z["contact_geometries"] != ""
    first = int(np.flatnonzero(contact)[0])
    precontact = np.arange(len(step)) < first
    with np.load(
      root / "test_v392_predictions" / f"{name}.npz", allow_pickle=False
    ) as z:
      if not np.array_equal(step, z["step"]):
        raise ValueError(f"V3.92 step mismatch: {name}")
      v392_force = z["estimated_force_body_n"]
      v392_point = z["estimated_point_body_m"]
    with np.load(
      root / "test_frozen/predictions" / f"{name}.npz", allow_pickle=False
    ) as z:
      if not np.array_equal(step, z["step"]):
        raise ValueError(f"Candidate step mismatch: {name}")
      candidate_force = z["estimated_force_body_n"]
    audit_path = (
      root / ("repair" if name.startswith("139_") else "test") / name / "audit.json"
    )
    weight = json.loads(audit_path.read_text())["robot_weight_n"]
    for version, force in (
      ("v392", v392_force),
      ("candidate_force_v392_point", candidate_force),
    ):
      frame, history = replay(force, v392_point, pose, weight)
      metrics = run_metrics(
        oracle_frame, oracle_history, frame, history, precontact, weight
      )
      rows.append({"episode": name, "version": version, **metrics})
      np.savez_compressed(
        traces / f"{name}_{version}.npz",
        step=step,
        oracle_frame=oracle_frame,
        estimated_frame=frame,
        oracle_history=oracle_history,
        estimated_history=history,
      )
  summary = {}
  for version in ("v392", "candidate_force_v392_point"):
    selected = [row for row in rows if row["version"] == version]
    summary[version] = {
      "precontact_false_active_episodes": sum(
        row["precontact_false_active_frames"] > 0 for row in selected
      ),
      "oracle_active_missed_fraction": sum(
        row["oracle_active_missed_frames"] for row in selected
      )
      / sum(row["oracle_active_frames"] for row in selected),
      **{
        key: float(np.median([row[key] for row in selected if row[key] is not None]))
        for key in (
          "oracle_active_force_obs_mae_n",
          "both_active_direction_mean_deg",
          "both_active_point_mean_m",
          "oracle_active_frame_l2_mean",
          "oracle_active_history_l2_mean",
        )
      },
    }
  result = {"summary": summary, "per_episode": rows}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
