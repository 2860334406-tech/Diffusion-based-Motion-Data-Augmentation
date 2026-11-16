"""Headless visual and kinematic review for curated LAFAN G1 clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mediapy as media
import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.viewer import OffscreenRenderer, ViewerConfig

from csv_to_npz import JOINT_NAMES, _setup_sim
from smp.utils import detect_device


@dataclass
class Cfg:
  input_dir: str = "datasets/csv/lafan_low_speed_balance"
  output_dir: str = "logs/reproduction/data/lafan_low_speed_review"
  input_fps: int = 30
  output_fps: int = 50
  device: str = ""
  include_mirrors: bool = False
  width: int = 640
  height: int = 480


@torch.no_grad()
def review(cfg: Cfg) -> list[dict[str, object]]:
  device = cfg.device or detect_device()
  input_dir = Path(cfg.input_dir)
  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  paths = sorted(input_dir.glob("*.csv"))
  if not cfg.include_mirrors:
    paths = [path for path in paths if not path.stem.endswith("_mirror")]
  if not paths:
    raise FileNotFoundError(f"No CSV clips found in {input_dir}")

  sim, scene = _setup_sim(device)
  robot: Entity = scene["robot"]
  joint_ids = torch.tensor(
    robot.find_joints(JOINT_NAMES, preserve_order=True)[0],
    device=sim.device,
    dtype=torch.long,
  )
  limits = robot.data.joint_pos_limits[0, joint_ids].detach().cpu()
  viewer_cfg = ViewerConfig(
    height=cfg.height,
    width=cfg.width,
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    distance=2.3,
    elevation=-8.0,
    azimuth=35.0,
  )
  renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
  renderer.initialize()
  reports: list[dict[str, object]] = []

  try:
    for path in paths:
      motion = MotionLoader(
        motion_file=str(path),
        input_fps=cfg.input_fps,
        output_fps=cfg.output_fps,
        device=sim.device,
      )
      scene.reset()
      frames: list[np.ndarray] = []
      for _ in range(motion.output_frames):
        state, _ = motion.get_next_state()
        base_pos, base_rot, base_lin_vel, base_ang_vel, joint_pos, joint_vel = state
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = base_pos
        root[:, 3:7] = base_rot
        root[:, 7:10] = base_lin_vel
        root[:, 10:13] = base_ang_vel
        robot.write_root_state_to_sim(root)
        all_pos = robot.data.default_joint_pos.clone()
        all_vel = robot.data.default_joint_vel.clone()
        all_pos[:, joint_ids] = joint_pos
        all_vel[:, joint_ids] = joint_vel
        robot.write_joint_state_to_sim(all_pos, all_vel)
        sim.forward()
        scene.update(1.0 / cfg.output_fps)
        renderer.update(sim.data)
        frames.append(renderer.render().copy())

      joint_pos = motion.motion_dof_poss.detach().cpu()
      finite = bool(
        torch.isfinite(motion.motion_base_poss).all()
        and torch.isfinite(motion.motion_base_rots).all()
        and torch.isfinite(joint_pos).all()
      )
      within = (joint_pos >= limits[:, 0]) & (joint_pos <= limits[:, 1])
      planar_speed = torch.linalg.vector_norm(
        motion.motion_base_lin_vels[:, :2], dim=-1
      )
      net_displacement = torch.linalg.vector_norm(
        motion.motion_base_poss[-1, :2] - motion.motion_base_poss[0, :2]
      )
      quat = motion.motion_base_rots
      yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
      )
      yaw_change = torch.rad2deg(
        torch.abs(torch.atan2(torch.sin(yaw[-1] - yaw[0]), torch.cos(yaw[-1] - yaw[0])))
      )
      video_path = output_dir / f"{path.stem}.mp4"
      sheet_path = output_dir / f"{path.stem}_contact_sheet.jpg"
      media.write_video(video_path, frames, fps=cfg.output_fps)
      indices = np.linspace(0, len(frames) - 1, 12, dtype=int)
      rows = [
        np.concatenate([frames[index] for index in indices[start : start + 4]], axis=1)
        for start in range(0, 12, 4)
      ]
      media.write_image(sheet_path, np.concatenate(rows, axis=0))
      reports.append(
        {
          "clip": path.name,
          "frames_50hz": motion.output_frames,
          "finite": finite,
          "joint_limit_violations": int((~within).sum().item()),
          "mean_planar_speed_m_s": float(planar_speed.mean().item()),
          "max_planar_speed_m_s": float(planar_speed.max().item()),
          "net_displacement_m": float(net_displacement.item()),
          "yaw_change_deg": float(yaw_change.item()),
          "video": str(video_path),
          "contact_sheet": str(sheet_path),
        }
      )
  finally:
    renderer.close()

  report_path = output_dir / "review_metrics.json"
  report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
  print(json.dumps(reports, indent=2))
  return reports


if __name__ == "__main__":
  review(tyro.cli(Cfg))
