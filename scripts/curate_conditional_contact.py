"""Curate five motion-response classes with subject-disjoint FK safety screening.

CPU-only curation. No source CSV or historical dataset is modified.
Invoke as: python -m scripts.curate_conditional_contact
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from scipy.spatial.transform import Rotation, Slerp
from scripts.csv_to_npz import EE_BODY_NAMES, JOINT_NAMES, _compute_windows
from scripts.curate_lafan_low_speed import mirror_g1_motion

from smp.pretrain.conditioning import CONDITION_NAMES, CONDITION_SCHEMA, sha256


@dataclass
class Cfg:
  input_dir: str = "datasets/csv/lafan"
  output_dir: str = "logs/reproduction/conditional_contact_v1/data"
  window_size: int = 10
  stride: int = 5
  fps: int = 50
  seed: int = 42


def subject_split(name: str) -> str:
  match = re.search(r"subject([1-5])", name)
  if not match:
    raise ValueError(f"No subject ID in {name}")
  return "validation" if match[1] == "5" else "train"


def interpolate(raw: np.ndarray, fps: int = 50):
  times = np.arange(len(raw), dtype=np.float64) / 30.0
  target = np.arange(0.0, times[-1], 1.0 / fps)
  motion = np.empty((len(target), 36), dtype=np.float64)
  for j in (*range(3), *range(7, 36)):
    motion[:, j] = np.interp(target, times, raw[:, j])
  motion[:, 3:7] = Slerp(times, Rotation.from_quat(raw[:, 3:7]))(target).as_quat()
  return motion


def motion_arrays(motion: np.ndarray, fps: int):
  positions = motion[:, :3]
  rotations = Rotation.from_quat(motion[:, 3:7])
  quaternions = motion[:, [6, 3, 4, 5]]
  lin_vel = np.gradient(positions, 1.0 / fps, axis=0)
  ang_vel = np.empty_like(lin_vel)
  ang_vel[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() * (fps / 2)
  ang_vel[0] = (rotations[1] * rotations[0].inv()).as_rotvec() * fps
  ang_vel[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() * fps
  joints = motion[:, 7:]
  return positions, quaternions, lin_vel, ang_vel, joints


def classify_windows(motion: np.ndarray, cfg: Cfg):
  pos, quat, vel, ang_vel, joints = motion_arrays(motion, cfg.fps)
  starts = np.arange(0, len(motion) - cfg.window_size + 1, cfg.stride)
  idx = starts[:, None] + np.arange(cfg.window_size)
  w, x, y, z = quat.T
  yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z)))
  roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
  pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
  anchor = yaw[idx[:, -1]]
  vx = np.cos(anchor[:, None])*vel[idx, 0] + np.sin(anchor[:, None])*vel[idx, 1]
  vy = -np.sin(anchor[:, None])*vel[idx, 0] + np.cos(anchor[:, None])*vel[idx, 1]
  speed = np.linalg.norm(vel[idx, :2], axis=-1)
  pelvis_change = yaw[idx[:, -1]] - yaw[idx[:, 0]]
  waist = joints[idx, 12]
  waist_delta = waist[:, -1] - waist[:, 0]
  joint_velocity = np.gradient(joints, 1.0 / cfg.fps, axis=0)
  joint_rms = np.sqrt(np.mean(joint_velocity[idx]**2, axis=(1, 2)))
  stable = (
    (pos[idx, 2].min(1) >= .60)
    & (np.abs(roll[idx]).max(1) <= np.deg2rad(20))
    & (np.abs(pitch[idx]).max(1) <= np.deg2rad(20))
    & (np.abs(pelvis_change) <= .05)
  )
  masks = [stable & (speed.max(1) <= .08) & (np.abs(waist).max(1) <= .06)
           & (joint_rms <= .25) & (np.abs(ang_vel[idx, 2]).max(1) <= .15)]
  for sign in (1, -1):
    signed_q, signed_d = sign*waist[:, -1], sign*waist_delta
    entering = (signed_d >= .01) & (signed_q >= 0.0)
    holding = (signed_q >= .08) & (signed_d >= -.02)
    masks.append(stable & (speed.max(1) <= .20) & (signed_q <= .35) & (entering | holding))
  for sign in (-1, 1):
    forward = sign*vx.mean(1)
    masks.append(stable & (forward >= .05) & (forward <= .35)
                 & (np.abs(vy).mean(1) <= .10) & (np.abs(waist).max(1) <= .15))
  masks = np.stack(masks, axis=1)
  labels = np.where(masks.sum(1) == 1, masks.argmax(1), -1)
  metrics = np.stack((speed.max(1), vx.mean(1), vy.mean(1), waist_delta,
                      waist[:, -1], pelvis_change, joint_rms), axis=1)
  return starts, idx, labels, metrics, (pos, quat, vel, ang_vel, joints)


def screening_model():
  spec = get_spec()
  for geom in spec.geoms:
    if geom.name.endswith("_collision"):
      geom.contype = 1
      geom.conaffinity = 1
  spec.worldbody.add_geom(name="conditional_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=(0, 0, .1), contype=1, conaffinity=1)
  model = spec.compile()
  data = mujoco.MjData(model)
  joints = np.array([model.joint(name).id for name in JOINT_NAMES])
  bodies = np.array([model.body(name).id for name in EE_BODY_NAMES])
  feet = np.array([i for i in range(model.ngeom)
                   if re.fullmatch(r"(left|right)_foot[1-7]_collision", model.geom(i).name)])
  if len(feet) != 14:
    raise RuntimeError("Expected fourteen foot support spheres")
  return model, data, joints, bodies, feet


def curate(cfg: Cfg):
  torch.set_num_threads(2)
  output = Path(cfg.output_dir)
  paths = sorted(Path(cfg.input_dir).glob("*.csv"))
  if not paths:
    raise FileNotFoundError(cfg.input_dir)
  sources = {p.name: sha256(p) for p in paths}
  manifest_path = output / "manifest.json"
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    if manifest["sources"] != sources or manifest["config"] != asdict(cfg):
      raise RuntimeError("Existing dataset differs; select a new output directory")
    for record in manifest["files"]:
      if sha256(output / record["path"]) != record["sha256"]:
        raise RuntimeError("Existing curated shard checksum mismatch")
    print(json.dumps({"status": "verified_existing", "counts": manifest["counts"]}), flush=True)
    return
  output.mkdir(parents=True, exist_ok=True)
  model, data, joint_ids, body_ids, foot_ids = screening_model()
  floor_id = model.geom("conditional_floor").id
  free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
  free_adr = int(model.jnt_qposadr[free])
  joint_adr = model.jnt_qposadr[joint_ids]
  limits = model.jnt_range[joint_ids]
  manifest = {
    "schema_version": 1, "condition_schema": CONDITION_SCHEMA,
    "config": asdict(cfg), "sources": sources, "files": [],
    "counts": {split: [0]*5 for split in ("train", "validation")},
    "sampling": {"samples_per_epoch": 3360, "per_condition": [672]*5, "seed": cfg.seed},
    "metric_columns": ["max_speed", "mean_vx", "mean_vy", "waist_delta",
                       "waist_last", "pelvis_yaw_delta", "joint_rms_velocity"],
    "safety": {"joint_margin": 0.0001, "self_penetration_m": 0.0001,
               "support_gap_m": 0.03, "foot_penetration_m": 0.02,
               "nonfoot_ground_penetration_m": 0.002},
  }
  train_chunks = []
  with (output / "windows.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["source", "source_sha256", "split", "mirrored", "frame50_start",
                     "frame50_end_exclusive", "condition_id", "condition", "shard", "index"])
    for source in paths:
      raw = np.loadtxt(source, delimiter=",")
      if raw.ndim != 2 or raw.shape[1] != 36 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid CSV: {source}")
      interpolated = interpolate(raw, cfg.fps)
      split = subject_split(source.name)
      for mirrored in (False, True):
        motion = mirror_g1_motion(interpolated) if mirrored else interpolated
        starts, indices, labels, metrics, arrays = classify_windows(motion, cfg)
        pos, quat, vel, ang, joints = arrays
        valid_frames = np.zeros(len(motion), dtype=bool)
        ee = np.zeros((len(motion), 5, 3), dtype=np.float32)
        selected = np.unique(indices[labels >= 0])
        for frame in selected:
          if not ((joints[frame] >= limits[:, 0]+1e-4)
                  & (joints[frame] <= limits[:, 1]-1e-4)).all():
            continue
          data.qpos[:] = model.qpos0
          data.qpos[free_adr:free_adr+3] = pos[frame]
          data.qpos[free_adr+3:free_adr+7] = quat[frame]
          data.qpos[joint_adr] = joints[frame]
          mujoco.mj_forward(model, data)
          foot_bottom = data.geom_xpos[foot_ids, 2] - model.geom_size[foot_ids, 0]
          if foot_bottom.min() < -.02 or foot_bottom.min() > .03:
            continue
          safe = True
          for contact in data.contact:
            a, b = map(int, contact.geom)
            if floor_id not in (a, b):
              if contact.dist < -1e-4:
                safe = False
                break
            else:
              other = b if a == floor_id else a
              if other not in foot_ids and contact.dist < -.002:
                safe = False
                break
          if safe:
            valid_frames[frame] = True
            ee[frame] = data.xpos[body_ids]
        keep = (labels >= 0) & valid_frames[indices].all(1)
        if not keep.any():
          print(f"{source.name} mirror={mirrored}: no safe windows", flush=True)
          continue
        tensors = [torch.tensor(a, dtype=torch.float32) for a in (pos, quat, vel, ang, ee, joints)]
        windows = _compute_windows(*tensors, cfg.window_size, cfg.stride).numpy()[keep]
        selected_labels = labels[keep].astype(np.int64)
        name = f"{source.stem}{'_mirror' if mirrored else ''}.npz"
        directory = output / split
        directory.mkdir(exist_ok=True)
        shard = directory / name
        np.savez_compressed(shard, windows=windows, condition_id=selected_labels,
                            frame50_start=starts[keep], metrics=metrics[keep])
        counts = np.bincount(selected_labels, minlength=5)
        manifest["counts"][split] = (np.array(manifest["counts"][split])+counts).tolist()
        manifest["files"].append({
          "path": f"{split}/{name}", "source": source.name, "source_sha256": sources[source.name],
          "split": split, "mirrored": mirrored, "sha256": sha256(shard),
          "counts": counts.tolist(), "candidate_windows": int((labels >= 0).sum()),
          "accepted_windows": int(keep.sum()),
        })
        for index, (start, label) in enumerate(zip(starts[keep], selected_labels, strict=True)):
          writer.writerow([source.name, sources[source.name], split, mirrored, int(start),
                           int(start+cfg.window_size), int(label), CONDITION_NAMES[label],
                           f"{split}/{name}", index])
        if split == "train":
          train_chunks.append(windows)
        print(f"{split}/{name}: {counts.tolist()}", flush=True)
  if any(min(counts) == 0 for counts in manifest["counts"].values()):
    raise RuntimeError(f"Missing safe condition examples: {manifest['counts']}")
  training = np.concatenate(train_chunks).reshape(-1, 59)
  q_low, q_high = np.percentile(training, [1, 99], axis=0).astype(np.float32)
  q_high = np.where(q_high-q_low < 1e-6, q_low+1.0, q_high)
  np.savez(output / "norm_stats.npz", q_low=q_low, q_high=q_high)
  manifest["norm_stats_sha256"] = sha256(output / "norm_stats.npz")
  manifest["windows_manifest_sha256"] = sha256(output / "windows.csv")
  manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
  print(json.dumps({"status": "complete", "counts": manifest["counts"]}), flush=True)


if __name__ == "__main__":
  curate(tyro.cli(Cfg))
