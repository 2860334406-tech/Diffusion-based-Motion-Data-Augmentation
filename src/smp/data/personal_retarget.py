"""Audited, foot-constrained retargeting of personal GVHMR/GMR recordings.

The source is a motion reference, not a physics demonstration. Only an accepted
free-root execution may be exported as a training window. Pelvis yaw is free;
direction is measured in the initial heading using torso-attached landmarks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from smp.data.shoulder_ik import (
  FPS,
  IK_SUBSTEPS,
  JOINT_NAMES,
  SIM_DT,
  ClipSpec,
  Execution,
  ShoulderIkGenerator,
  _build_model,
  _dof_ids,
  _qpos_ids,
  _quat_tilt_wxyz,
  _quat_yaw_wxyz,
  atomic_json,
  contract_to_qpos,
  qpos_to_contract,
  qpos_to_csv,
  sha256,
)

CONDITIONS = ("stand", "left_shoulder", "right_shoulder")
SUBJECT_SPLITS = {"P01": "train", "P02": "validation", "P03": "test"}
STANCES = (
  (0.24, 0.70),
  (0.24, 0.55),
  (0.22, 0.70),
  (0.26, 0.70),
  (0.22, 0.55),
  (0.26, 0.55),
)
SCHEMA = "g1_personal_retarget_v1"


@dataclass(frozen=True)
class Config:
  seed: int = 42
  foot_position_m: float = 0.002
  foot_orientation_rad: float = math.radians(1)
  support_inset_m: float = 0.01
  collision_clearance_m: float = 0.003
  self_penetration_m: float = 0.0001
  nonfoot_ground_penetration_m: float = 0.002
  min_height_m: float = 0.60
  max_tilt_rad: float = math.radians(20)
  max_root_displacement_m: float = 0.10
  min_contact_fraction: float = 0.95
  max_slip_m: float = 0.015
  max_foot_yaw_rad: float = math.radians(2)
  max_joint_rmse_rad: float = 0.08
  max_waist_rmse_rad: float = 0.04
  max_effort_p99: float = 0.95
  max_saturation_fraction: float = 0.01
  max_hold_root_speed: float = 0.05
  max_hold_yaw_speed: float = 0.10
  shoulder_retreat_m: float = 0.015
  shoulder_reached_m: float = 0.014
  max_hold_shoulder_deviation_m: float = 0.005
  stand_waist_rad: float = 0.06
  smoothing_frames: int = 11
  smoothing_order: int = 3
  onset_rad: float = 0.02
  onset_fraction: float = 0.05
  onset_sustain_frames: int = 10
  terminal_hold_frames: int = 50
  terminal_blend_frames: int = 50
  terminal_policy: str = "median_pre_hold_reference_then_quintic_settle"
  pelvis_yaw_limit: float | None = None
  # Augmented clips are synthesised from a seed and may not contain a clean
  # neutral-onset-hold structure. When True, fall back to a fixed onset instead
  # of rejecting the clip (existing data production keeps the strict default).
  allow_onset_fallback: bool = False
  minimum_waist_amplitude: None = None


@dataclass
class Source:
  metadata: dict[str, Any]
  identity: dict[str, str]
  root: np.ndarray
  quat: np.ndarray
  joints: np.ndarray
  qpos: np.ndarray
  pelvis_delta: np.ndarray
  torso_delta: np.ndarray
  pelvis_yaw: np.ndarray
  torso_yaw: np.ndarray
  turn_start: int

  @property
  def clip_id(self) -> str:
    return self.metadata["clip_id"]

  @property
  def condition_id(self) -> int:
    return int(self.metadata["condition_id"])

  @property
  def split(self) -> str:
    return SUBJECT_SPLITS[self.metadata["subject_id"]]


def inventory(root: Path) -> list[dict[str, Any]]:
  """Validate every source file before producing any candidate."""
  result = []
  videos: set[str] = set()
  for path in sorted(root.glob("P*_session*_*.json")):
    metadata = json.loads(path.read_text())
    clip = metadata["clip_id"]
    condition = int(metadata["condition_id"])
    if path.stem != clip or metadata["subject_id"] not in SUBJECT_SPLITS:
      raise ValueError(f"Invalid personal source identity: {path}")
    if condition not in range(3) or metadata["condition"] != CONDITIONS[condition]:
      raise ValueError(f"Condition/label mismatch: {clip}")
    if (
      metadata["joint_order"] != "smp_g1_29_v1"
      or metadata["quaternion_order"] != "xyzw"
    ):
      raise ValueError(f"Source contract mismatch: {clip}")
    identity = {"metadata_sha256": sha256(path)}
    for extension in ("npz", "csv"):
      value = sha256(root / f"{clip}.{extension}")
      if value != metadata[f"{extension}_sha256"]:
        raise ValueError(f"Source {extension} identity mismatch: {clip}")
      identity[f"{extension}_sha256"] = value
    video = root / metadata["source_video_delivery_path"]
    video_hash = sha256(video)
    if video_hash != metadata["source_video_sha256"] or video_hash in videos:
      raise ValueError(f"Missing, duplicate or damaged source video: {clip}")
    videos.add(video_hash)
    identity["video_sha256"] = video_hash
    with np.load(root / f"{clip}.npz", allow_pickle=False) as archive:
      required = {"fps", "root_pos", "root_rot", "dof_pos"}
      if set(archive.files) != required:
        raise ValueError(f"Unexpected source keys: {clip}")
      fps = float(archive["fps"])
      position, rotation, joints = (
        archive[k] for k in ("root_pos", "root_rot", "dof_pos")
      )
      count = len(position)
      if (
        fps != float(metadata["fps"])
        or count != metadata["frames"]
        or fps <= 0
        or count < 50
      ):
        raise ValueError(f"Invalid source timing: {clip}")
      if (
        position.shape != (count, 3)
        or rotation.shape != (count, 4)
        or joints.shape != (count, 29)
      ):
        raise ValueError(f"Invalid source shape: {clip}")
      combined = np.concatenate((position, rotation, joints), axis=1)
      csv_values = np.loadtxt(root / f"{clip}.csv", delimiter=",")
      if not np.isfinite(combined).all() or not np.allclose(
        csv_values, combined, atol=1e-7, rtol=1e-7
      ):
        raise ValueError(f"Nonfinite or inconsistent source: {clip}")
      if np.max(np.abs(np.linalg.norm(rotation, axis=1) - 1)) > 1e-3:
        raise ValueError(f"Nonunit source quaternion: {clip}")
      if np.any(np.sum(rotation[1:] * rotation[:-1], axis=1) < 0):
        raise ValueError(f"Discontinuous source quaternion signs: {clip}")
    result.append({"metadata": metadata, "identity": identity})
  if len(result) != 27:
    raise ValueError(f"Expected 27 independent source clips, found {len(result)}")
  for subject in SUBJECT_SPLITS:
    for condition in range(3):
      if (
        sum(
          x["metadata"]["subject_id"] == subject
          and x["metadata"]["condition_id"] == condition
          for x in result
        )
        != 3
      ):
        raise ValueError(f"Source quota mismatch: {subject}/{condition}")
  return result


def resample(root: np.ndarray, quat: np.ndarray, joints: np.ndarray, fps: float):
  time = np.arange(len(root), dtype=np.float64) / fps
  target = np.arange(int(math.floor(time[-1] * FPS + 1e-8)) + 1) / FPS
  position = PchipInterpolator(time, root, axis=0)(target)
  angles = PchipInterpolator(time, joints, axis=0)(target)
  rotation = Slerp(time, Rotation.from_quat(quat))(target).as_quat()
  for index in range(1, len(rotation)):
    if np.dot(rotation[index - 1], rotation[index]) < 0:
      rotation[index] *= -1
  return position, rotation, angles


def detect_onset(signed_torso_yaw: np.ndarray, cfg: Config) -> int:
  values = np.asarray(signed_torso_yaw)
  threshold = max(cfg.onset_rad, cfg.onset_fraction * float(np.quantile(values, 0.95)))
  above = values > threshold
  run = np.convolve(
    above.astype(int), np.ones(cfg.onset_sustain_frames, dtype=int), mode="valid"
  )
  found = np.flatnonzero(run == cfg.onset_sustain_frames)
  if not len(found) or found[0] < FPS or found[0] >= len(above) - FPS:
    raise ValueError(
      "Source lacks a neutral prefix, directional onset or final hold interval"
    )
  return int(found[0])


def stable_terminal_target(values: np.ndarray, cfg: Config) -> np.ndarray:
  """Keep source timing, but give the solver an explicit settled last second.

  This operates only on soft IK targets. Source and measured execution arrays
  are never padded or replaced. Use the preceding observed pose, not an
  extrapolated or amplified angle, when the recording ends with a return.
  """
  result = np.asarray(values).copy()
  hold = cfg.terminal_hold_frames
  blend = cfg.terminal_blend_frames
  start = len(result) - hold - blend
  if start < FPS:
    raise ValueError("Source is too short for neutral motion and terminal hold")
  goal = np.median(result[start : start + blend], axis=0)
  scalar = np.linspace(0, 1, blend)
  scalar = 10 * scalar**3 - 15 * scalar**4 + 6 * scalar**5
  weight = scalar.reshape((blend,) + (1,) * (result.ndim - 1))
  result[start : start + blend] = (1 - weight) * result[
    start : start + blend
  ] + weight * goal
  result[start + blend :] = goal
  return result


def load_source(
  root: Path, item: dict[str, Any], model: mujoco.MjModel, cfg: Config
) -> Source:
  metadata = item["metadata"]
  with np.load(root / f"{metadata['clip_id']}.npz", allow_pickle=False) as z:
    position, rotation, joints = resample(
      z["root_pos"], z["root_rot"], z["dof_pos"], float(z["fps"])
    )
  qpos = contract_to_qpos(position, rotation, joints, model)
  data = mujoco.MjData(model)
  pelvis_rotations, torso_rotations = [], []
  for q in qpos:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    pelvis_rotations.append(data.xmat[model.body("pelvis").id].reshape(3, 3).copy())
    torso_rotations.append(data.xmat[model.body("torso_link").id].reshape(3, 3).copy())
  pelvis_rotations = np.asarray(pelvis_rotations)
  torso_rotations = np.asarray(torso_rotations)
  pelvis_yaw = np.unwrap(
    np.arctan2(pelvis_rotations[:, 1, 0], pelvis_rotations[:, 0, 0])
  )
  torso_yaw = np.unwrap(np.arctan2(torso_rotations[:, 1, 0], torso_rotations[:, 0, 0]))
  pelvis_yaw -= np.median(pelvis_yaw[:FPS])
  torso_yaw -= np.median(torso_yaw[:FPS])
  # Smooth only the soft target, never rewrite the source or measured execution.
  pelvis_delta = Rotation.from_matrix(
    pelvis_rotations[0].T @ pelvis_rotations
  ).as_rotvec()
  torso_delta = Rotation.from_matrix(torso_rotations[0].T @ torso_rotations).as_rotvec()
  pelvis_delta = savgol_filter(
    pelvis_delta, cfg.smoothing_frames, cfg.smoothing_order, axis=0
  )
  torso_delta = savgol_filter(
    torso_delta, cfg.smoothing_frames, cfg.smoothing_order, axis=0
  )
  pelvis_delta = stable_terminal_target(pelvis_delta, cfg)
  # Cap pelvic yaw so the retargeted motion cannot "turn the whole body" instead
  # of twisting at the waist. The torso target is untouched, so any capped pelvis
  # rotation is transferred into the relative waist angle.
  if cfg.pelvis_yaw_limit is not None:
    lim = float(cfg.pelvis_yaw_limit)
    pelvis_delta[:, 2] = np.clip(pelvis_delta[:, 2], -lim, lim)
  torso_delta = stable_terminal_target(torso_delta, cfg)
  try:
    onset = (
      0
      if metadata["condition_id"] == 0
      else detect_onset(torso_yaw * (1 if metadata["condition_id"] == 1 else -1), cfg)
    )
  except ValueError:
    if not cfg.allow_onset_fallback:
      raise
    onset = min(FPS, max(1, len(torso_yaw) // 3))
  return Source(
    metadata,
    item["identity"],
    position,
    rotation,
    joints,
    qpos,
    pelvis_delta,
    torso_delta,
    pelvis_yaw,
    torso_yaw,
    onset,
  )


def support_polygon(
  model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
  points = []
  for index in range(model.ngeom):
    name = model.geom(index).name
    if name.startswith(("left_foot", "right_foot")) and name.endswith("_collision"):
      center = data.geom_xpos[index, :2]
      radius = float(model.geom_size[index, 0])
      points.extend(center + radius * np.array(((1, 0), (-1, 0), (0, 1), (0, -1))))
  hull = ConvexHull(np.asarray(points))
  return hull.equations[:, :2], -hull.equations[:, 2]


class SupportLimit(mink.Limit):
  def __init__(self, normals: np.ndarray, bounds: np.ndarray, inset: float):
    self.normals, self.bounds, self.inset = normals, bounds, inset

  def compute_qp_inequalities(self, configuration, dt):
    del dt
    jac = np.zeros((3, configuration.model.nv))
    mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, jac, 1)
    com = configuration.data.subtree_com[1, :2]
    return mink.Constraint(
      G=self.normals @ jac[:2], h=self.bounds - self.inset - self.normals @ com
    )


def collision_limit(
  model: mujoco.MjModel, clearance: float
) -> mink.CollisionAvoidanceLimit:
  active = [
    i
    for i in range(model.ngeom)
    if model.geom_bodyid[i] and (model.geom_contype[i] or model.geom_conaffinity[i])
  ]
  excluded = set(int(x) for x in model.exclude_signature)
  pairs = []
  for ai, a in enumerate(active):
    for b in active[ai + 1 :]:
      ba, bb = sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b])))
      if (ba << 16) + bb not in excluded:
        pairs.append(([a], [b]))
  floor = model.geom("shoulder_ik_floor").id
  nonfeet = [
    index
    for index in active
    if not model.geom(index).name.startswith(("left_foot", "right_foot"))
  ]
  pairs.append(([floor], nonfeet))
  return mink.CollisionAvoidanceLimit(
    model,
    pairs,
    minimum_distance_from_collisions=clearance,
    collision_detection_distance=0.05,
  )


def heading_retreat(shoulder: np.ndarray, pelvis: np.ndarray, initial_yaw: float):
  axis = np.array((math.cos(initial_yaw), math.sin(initial_yaw), 0))
  world = -(shoulder - shoulder[0]) @ axis
  relative = shoulder - pelvis
  compensated = -(relative - relative[0]) @ axis
  return world, compensated


class SimulationFailure(RuntimeError):
  def __init__(self, message: str, qpos: np.ndarray, qvel: np.ndarray):
    super().__init__(message)
    self.qpos = qpos.copy()
    self.qvel = qvel.copy()


class PersonalRetargeter(ShoulderIkGenerator):
  def __init__(self, cfg: Config | None = None):
    super().__init__()
    self.cfg = cfg or Config()
    self.gates = replace(
      self.gates,
      self_penetration_m=self.cfg.self_penetration_m,
      nonfoot_ground_penetration_m=self.cfg.nonfoot_ground_penetration_m,
    )
    # Real model limits; no historical pelvis cap or fixed shoulder amplitudes.
    self.ik_model = _build_model(soft_limits=False)
    self.ik_joint_qpos = _qpos_ids(self.ik_model)
    self.sim_joint_qpos = _qpos_ids(self.sim_model)
    self.sim_joint_dof = _dof_ids(self.sim_model)

  def _ik_limits(self, model):
    return [
      mink.ConfigurationLimit(model, gain=0.9),
      mink.VelocityLimit(model, {name: 2.0 for name in JOINT_NAMES}),
      mink.FreeJointVelocityLimit(
        model, max_linear_velocity=0.15, max_angular_velocity=0.5
      ),
      collision_limit(model, self.cfg.collision_clearance_m),
    ]

  def reference(self, source: Source, attempt: int, overrides: dict | None = None):
    width, knee = STANCES[attempt]
    ov = overrides or {}
    width = float(ov.get("stance_width_m", width))
    knee = float(ov.get("knee_flexion_rad", knee))
    digest = hashlib.sha256(
      f"{self.cfg.seed}:{source.clip_id}:{attempt}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    spec = ClipSpec(
      source.clip_id,
      source.clip_id,
      "smoke",
      source.split,
      source.condition_id,
      attempt,
      0,
      attempt,
      self.cfg.seed,
      0,
      0,
      0,
      width,
      knee,
      tuple(float(x) for x in rng.uniform(-1, 1, 4)),
      0.0,
      # Optional posture overrides (used by data augmentation to preserve the
      # generated arm/knee intent instead of the fixed personal-retarget stance).
      shoulder_roll_rad=float(ov.get("shoulder_roll_rad", 0.20)),
      shoulder_yaw_rad=float(ov.get("shoulder_yaw_rad", 0.0)),
      shoulder_pitch_rad=float(ov.get("shoulder_pitch_rad", 0.20)),
      elbow_rad=float(ov.get("elbow_rad", 0.60)),
      wrist_roll_rad=float(ov.get("wrist_roll_rad", 0.0)),
      wrist_pitch_rad=float(ov.get("wrist_pitch_rad", 0.0)),
      wrist_yaw_rad=float(ov.get("wrist_yaw_rad", 0.0)),
    )
    configuration, left, right, pelvis, posture, com, limits = (
      self._stand_configuration(spec)
    )
    model = self.ik_model
    normals, bounds = support_polygon(model, configuration.data)
    limits.append(SupportLimit(normals, bounds, self.cfg.support_inset_m))
    stand = configuration.q.copy()
    pelvis_start = configuration.get_transform_frame_to_world("pelvis", "body")
    torso_start = configuration.get_transform_frame_to_world("torso_link", "body")
    torso = mink.FrameTask(
      "torso_link",
      "body",
      position_cost=0,
      orientation_cost=[2.0, 2.0, 10.0],
      lm_damping=1.0,
    )
    pelvis.set_orientation_cost([3.0, 3.0, 1.0])
    cost = np.r_[np.zeros(6), np.full(29, 0.15)]
    cost[self.sim_joint_dof[12]] = 1.0
    cost[self.sim_joint_dof[15:]] = 0.5
    posture = mink.PostureTask(model, cost=cost, lm_damping=1.0)
    source_baseline = np.median(source.joints[:FPS], axis=0)
    deltas = savgol_filter(
      source.joints - source_baseline,
      self.cfg.smoothing_frames,
      self.cfg.smoothing_order,
      axis=0,
    )
    deltas = stable_terminal_target(deltas, self.cfg)
    # The floating pelvis and upper body follow source rotation; the legs solve
    # balance/foot constraints instead of copying the invalid GMR leg pose.
    qpos = np.empty((len(source.qpos), model.nq))
    max_pos = max_rot = 0.0
    min_com_margin = math.inf
    for frame in range(len(qpos)):
      target = stand.copy()
      target[self.ik_joint_qpos[12:]] += deltas[frame, 12:]
      posture.set_target(target)
      pelvis_rot = mink.SO3.from_matrix(
        pelvis_start.rotation().as_matrix()
        @ Rotation.from_rotvec(source.pelvis_delta[frame]).as_matrix()
      )
      torso_rot = mink.SO3.from_matrix(
        torso_start.rotation().as_matrix()
        @ Rotation.from_rotvec(source.torso_delta[frame]).as_matrix()
      )
      pelvis.set_target(
        mink.SE3.from_rotation_and_translation(pelvis_rot, pelvis_start.translation())
      )
      torso.set_target(
        mink.SE3.from_rotation_and_translation(torso_rot, torso_start.translation())
      )
      for _ in range(IK_SUBSTEPS):
        velocity = mink.solve_ik(
          configuration,
          [pelvis, torso, posture, com],
          SIM_DT,
          "daqp",
          damping=1e-4,
          limits=limits,
          constraints=[left, right],
        )
        if not np.isfinite(velocity).all():
          raise FloatingPointError("Nonfinite IK velocity")
        configuration.integrate_inplace(velocity, SIM_DT)
        for foot in (left, right):
          error = foot.compute_error(configuration)
          max_pos = max(max_pos, float(np.linalg.norm(error[:3])))
          max_rot = max(max_rot, float(np.linalg.norm(error[3:])))
        min_com_margin = min(
          min_com_margin,
          float(np.min(bounds - normals @ configuration.data.subtree_com[1, :2])),
        )
      qpos[frame] = configuration.q
    return qpos, {
      "foot_position_error_m": max_pos,
      "foot_orientation_error_rad": max_rot,
      "minimum_ik_com_margin_m": min_com_margin,
      "stance_width_m": width,
      "knee_flexion_rad": knee,
      "collision_pair_count": len(limits[3].geom_id_pairs),
    }

  def execute_checked(self, reference: np.ndarray) -> tuple[Execution, dict]:
    model = self.sim_model
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    mujoco.mj_forward(model, data)
    maps = self._controller_maps()
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    buffers: dict[str, list] = {
      k: []
      for k in (
        "qpos",
        "qvel",
        "effort",
        "left",
        "right",
        "any",
        "lpos",
        "rpos",
        "lyaw",
        "ryaw",
      )
    }
    flags = {"self_collision": False, "nonfoot_ground": False}
    contacts = {
      "pre_self_collision": False,
      "pre_nonfoot_ground": False,
      "post_self_collision": False,
      "post_nonfoot_ground": False,
      "pre_support_lost": False,
      "post_support_lost": False,
      "pre_post_state_invalid": False,
    }
    warning_before = np.array([int(w.number) for w in data.warning])

    def step(target):
      nonlocal warning_before
      ratio = self._track_step(data, target, maps)
      warning_after = np.array([int(w.number) for w in data.warning])
      invalid = not (
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(data.qacc).all()
      )
      if invalid or np.any(warning_after > warning_before):
        raise SimulationFailure(
          f"MuJoCo unstable at {data.time:.4f}s; warning_counts={warning_after.tolist()}",
          data.qpos,
          data.qvel,
        )
      warning_before = warning_after
      return ratio, self._contact_state(data)

    joint_ranges = model.jnt_range[[model.joint(name).id for name in JOINT_NAMES]]

    def audit_extra(state, prefix):
      contacts[prefix + "_self_collision"] |= state[2]
      contacts[prefix + "_nonfoot_ground"] |= state[3]
      contacts[prefix + "_support_lost"] |= not (state[0] or state[1])
      tilt = _quat_tilt_wxyz(data.qpos[3:7])
      joint_values = data.qpos[self.sim_joint_qpos]
      contacts["pre_post_state_invalid"] |= bool(
        data.qpos[2] < self.cfg.min_height_m
        or np.max(np.abs(tilt)) > self.cfg.max_tilt_rad
        or np.any(joint_values < joint_ranges[:, 0] + 1e-4)
        or np.any(joint_values > joint_ranges[:, 1] - 1e-4)
      )

    for _ in range(round(1 / SIM_DT)):
      _, state = step(reference[0])
      audit_extra(state, "pre")
    for target in reference:
      ratios, states = [], []
      for _ in range(IK_SUBSTEPS):
        ratio, state = step(target)
        ratios.append(ratio)
        states.append(state)
        flags["self_collision"] |= state[2]
        flags["nonfoot_ground"] |= state[3]
      buffers["qpos"].append(data.qpos.copy())
      buffers["qvel"].append(data.qvel.copy())
      buffers["effort"].append(np.max(ratios, axis=0))
      buffers["left"].append(np.mean([x[0] for x in states]))
      buffers["right"].append(np.mean([x[1] for x in states]))
      buffers["any"].append(all(x[0] or x[1] for x in states))
      for prefix, body in zip(("l", "r"), feet, strict=True):
        buffers[prefix + "pos"].append(data.xpos[body].copy())
        buffers[prefix + "yaw"].append(_quat_yaw_wxyz(data.xquat[body]))
    for _ in range(FPS * IK_SUBSTEPS):
      _, state = step(reference[-1])
      audit_extra(state, "post")
    execution = Execution(
      np.asarray(buffers["qpos"]),
      np.asarray(buffers["qvel"]),
      np.asarray(buffers["effort"]),
      np.asarray(buffers["left"]),
      np.asarray(buffers["right"]),
      np.asarray(buffers["any"]),
      np.asarray(buffers["lpos"]),
      np.asarray(buffers["rpos"]),
      np.unwrap(buffers["lyaw"]),
      np.unwrap(buffers["ryaw"]),
      flags,
      {},
    )
    return execution, contacts

  def landmarks(self, qpos: np.ndarray):
    model, data = self.sim_model, mujoco.MjData(self.sim_model)
    torso_id, pelvis_id = model.body("torso_link").id, model.body("pelvis").id
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    landmarks = []
    for side in ("left", "right"):
      point = data.geom_xpos[model.geom(f"{side}_shoulder_yaw_collision").id].copy()
      landmarks.append(
        data.xmat[torso_id].reshape(3, 3).T @ (point - data.xpos[torso_id])
      )
    pelvis, shoulder, torso_yaw = [], [], []
    for q in qpos:
      data.qpos[:] = q
      mujoco.mj_forward(model, data)
      matrix = data.xmat[torso_id].reshape(3, 3)
      pelvis.append(data.xpos[pelvis_id].copy())
      shoulder.append([data.xpos[torso_id] + matrix @ point for point in landmarks])
      torso_yaw.append(math.atan2(matrix[1, 0], matrix[0, 0]))
    return np.asarray(pelvis), np.asarray(shoulder), np.unwrap(torso_yaw)

  def validate(
    self, source: Source, reference: np.ndarray, ex: Execution, ik: dict, audits: dict
  ):
    cfg = self.cfg
    actual = ex.qpos
    pelvis, shoulder, torso_yaw = self.landmarks(actual)
    root_yaw = np.unwrap([_quat_yaw_wxyz(x[3:7]) for x in actual])
    root_tilt = np.asarray([_quat_tilt_wxyz(x[3:7]) for x in actual])
    joints = actual[:, self.sim_joint_qpos]
    error = joints - reference[:, self.sim_joint_qpos]
    limits = self.sim_model.jnt_range[[self.sim_model.joint(n).id for n in JOINT_NAMES]]
    hold = slice(-FPS, None)
    waist_delta = float(np.median(joints[hold, 12]) - np.median(joints[:FPS, 12]))
    pelvis_delta = float(np.median(root_yaw[hold]) - np.median(root_yaw[:FPS]))
    metrics = {
      **ik,
      **audits,
      "frames": len(actual),
      "duration_s": len(actual) / FPS,
      "finite": bool(np.isfinite(actual).all() and np.isfinite(ex.qvel).all()),
      "root_height_min_m": float(actual[:, 2].min()),
      "root_tilt_abs_max_rad": float(np.abs(root_tilt).max()),
      "root_displacement_max_m": float(
        np.linalg.norm(actual[:, :2] - actual[0, :2], axis=1).max()
      ),
      "left_contact_fraction": float(ex.left_contact.mean()),
      "right_contact_fraction": float(ex.right_contact.mean()),
      "continuous_support": bool(ex.any_support.all()),
      "foot_slip_max_m": max(
        float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max())
        for p in (ex.left_foot_pos, ex.right_foot_pos)
      ),
      "foot_yaw_drift_max_rad": max(
        float(np.abs(y - y[0]).max()) for y in (ex.left_foot_yaw, ex.right_foot_yaw)
      ),
      "joint_tracking_rmse_rad": float(np.sqrt(np.mean(error**2))),
      "waist_tracking_rmse_rad": float(np.sqrt(np.mean(error[:, 12] ** 2))),
      "effort_p99_ratio": float(np.quantile(ex.effort_ratio, 0.99)),
      "effort_saturation_fraction": float(
        np.any(ex.effort_ratio >= 0.95 - 1e-6, axis=1).mean()
      ),
      "motor_velocity_ratio_max": self._motor_velocity_ratio(ex.qvel),
      "hold_root_speed_p95_m_s": float(
        np.quantile(np.linalg.norm(ex.qvel[hold, :2], axis=1), 0.95)
      ),
      "hold_waist_speed_p95_rad_s": float(
        np.quantile(np.abs(ex.qvel[hold, self.sim_joint_dof[12]]), 0.95)
      ),
      "hold_pelvis_yaw_speed_p95_rad_s": float(
        np.quantile(np.abs(np.gradient(root_yaw, 1 / FPS)[hold]), 0.95)
      ),
      "actual_waist_delta_rad": waist_delta,
      "actual_pelvis_delta_rad": pelvis_delta,
      "actual_torso_delta_rad": float(
        np.median(torso_yaw[hold]) - np.median(torso_yaw[:FPS])
      ),
      "source_waist_delta_rad": float(
        np.median(source.joints[hold, 12]) - np.median(source.joints[:FPS, 12])
      ),
      "source_pelvis_delta_rad": float(np.median(source.pelvis_yaw[hold])),
      "source_torso_delta_rad": float(np.median(source.torso_yaw[hold])),
      "source_torso_tracking_rmse_rad": float(
        np.sqrt(np.mean((torso_yaw - torso_yaw[0] - source.torso_yaw) ** 2))
      ),
      "self_collision": ex.collision_flags["self_collision"],
      "nonfoot_ground": ex.collision_flags["nonfoot_ground"],
      "joint_limit_min_margin_rad": float(
        np.minimum(joints - limits[:, 0], limits[:, 1] - joints).min()
      ),
    }
    checks = {
      "finite": metrics["finite"],
      "joint_limits": metrics["joint_limit_min_margin_rad"] >= 1e-4,
      "ik_feet_position": ik["foot_position_error_m"] <= cfg.foot_position_m,
      "ik_feet_orientation": ik["foot_orientation_error_rad"]
      <= cfg.foot_orientation_rad,
      "ik_com_inside_support": ik["minimum_ik_com_margin_m"]
      >= cfg.support_inset_m - 1e-5,
      "root_height": metrics["root_height_min_m"] >= cfg.min_height_m,
      "root_tilt": metrics["root_tilt_abs_max_rad"] <= cfg.max_tilt_rad,
      "root_displacement": metrics["root_displacement_max_m"]
      <= cfg.max_root_displacement_m,
      "left_contact": metrics["left_contact_fraction"] >= cfg.min_contact_fraction,
      "right_contact": metrics["right_contact_fraction"] >= cfg.min_contact_fraction,
      "continuous_support": metrics["continuous_support"],
      "foot_slip": metrics["foot_slip_max_m"] <= cfg.max_slip_m,
      "foot_yaw": metrics["foot_yaw_drift_max_rad"] <= cfg.max_foot_yaw_rad,
      "joint_tracking": metrics["joint_tracking_rmse_rad"] <= cfg.max_joint_rmse_rad,
      "waist_tracking": metrics["waist_tracking_rmse_rad"] <= cfg.max_waist_rmse_rad,
      "effort_p99": metrics["effort_p99_ratio"] <= cfg.max_effort_p99 + 1e-6,
      "effort_saturation": metrics["effort_saturation_fraction"]
      <= cfg.max_saturation_fraction,
      "motor_velocity": metrics["motor_velocity_ratio_max"] <= 1,
      "hold_root_speed": metrics["hold_root_speed_p95_m_s"] <= cfg.max_hold_root_speed,
      "hold_waist_speed": metrics["hold_waist_speed_p95_rad_s"]
      <= cfg.max_hold_yaw_speed,
      "hold_pelvis_speed": metrics["hold_pelvis_yaw_speed_p95_rad_s"]
      <= cfg.max_hold_yaw_speed,
      "self_collision": not metrics["self_collision"],
      "nonfoot_ground": not metrics["nonfoot_ground"],
      "pre_post_collision": not any(audits.values()),
    }
    if source.condition_id == 0:
      checks["stand_neutral_waist"] = bool(
        np.max(np.abs(joints[:, 12])) <= cfg.stand_waist_rad
      )
    else:
      sign = 1 if source.condition_id == 1 else -1
      world, compensated = heading_retreat(
        shoulder[:, source.condition_id - 1], pelvis, root_yaw[0]
      )
      onset, deadline = source.turn_start, len(actual) - FPS - 1
      reached = (world >= cfg.shoulder_reached_m) & (
        compensated >= cfg.shoulder_reached_m
      )
      hit = np.flatnonzero(reached[onset : deadline + 1])
      shoulder_relative = shoulder[:, source.condition_id - 1] - pelvis
      deviation = np.linalg.norm(
        shoulder_relative[hold] - np.median(shoulder_relative[hold], axis=0), axis=1
      )
      metrics.update(
        {
          "shoulder_retreat_final_world_m": float(world[-1]),
          "shoulder_retreat_final_compensated_m": float(compensated[-1]),
          "shoulder_retreat_hold_min_m": float(
            min(world[hold].min(), compensated[hold].min())
          ),
          "hold_shoulder_deviation_max_m": float(deviation.max()),
          "turn_start_frame": onset,
          "completion_deadline_s": (deadline - onset) / FPS,
          "completion_time_s": float(hit[0] / FPS) if len(hit) else None,
          "signed_waist_delta_rad": sign * waist_delta,
          "signed_pelvis_delta_rad": sign * pelvis_delta,
          "signed_torso_delta_rad": sign * metrics["actual_torso_delta_rad"],
        }
      )
      checks.update(
        {
          "direction": metrics["signed_torso_delta_rad"] > 0,
          "endpoint_world": world[-1] >= cfg.shoulder_retreat_m,
          "endpoint_compensated": compensated[-1] >= cfg.shoulder_retreat_m,
          "completion_time": bool(len(hit)),
          "hold_retreat": metrics["shoulder_retreat_hold_min_m"]
          >= cfg.shoulder_reached_m,
          "hold_shoulder_stability": metrics["hold_shoulder_deviation_max_m"]
          <= cfg.max_hold_shoulder_deviation_m,
        }
      )
    metrics["checks"] = {name: bool(value) for name, value in checks.items()}
    metrics["failed_checks"] = sorted(
      name for name, passed in checks.items() if not passed
    )
    metrics["passed"] = not metrics["failed_checks"]
    return metrics


def write_attempt(
  output: Path,
  source: Source,
  attempt: int,
  reference: np.ndarray,
  execution: Execution | None,
  metrics: dict,
  model,
):
  directory = output / "attempts" / source.clip_id / f"attempt_{attempt}"
  directory.mkdir(parents=True, exist_ok=True)
  paths = {}

  def save(name, q, **extra):
    position, rotation, joints = qpos_to_contract(q, model)
    path = directory / name
    np.savez_compressed(
      path,
      fps=np.float32(FPS),
      root_pos=position,
      root_rot=rotation,
      dof_pos=joints,
      **extra,
    )
    paths[name] = {"path": str(path.relative_to(output)), "sha256": sha256(path)}

  save("source50.npz", source.qpos)
  save("reference.npz", reference)
  if execution is not None:
    save(
      "execution.npz",
      execution.qpos,
      qvel=execution.qvel.astype(np.float32),
      effort_ratio=execution.effort_ratio.astype(np.float32),
    )
    csv_path = directory / "execution.csv"
    np.savetxt(csv_path, qpos_to_csv(execution.qpos, model), delimiter=",", fmt="%.9g")
    paths["execution.csv"] = {
      "path": str(csv_path.relative_to(output)),
      "sha256": sha256(csv_path),
    }
  metadata = {
    "schema": SCHEMA,
    "source_clip_id": source.clip_id,
    "source_identity": source.identity,
    "subject_id": source.metadata["subject_id"],
    "condition_id": source.condition_id,
    "condition": CONDITIONS[source.condition_id],
    "split": source.split,
    "attempt": attempt,
    "fps": FPS,
    "frames": len(reference),
    "turn_start_frame": source.turn_start,
    "quaternion_order": "xyzw",
    "joint_order": "smp_g1_29_v1",
    "source_domain": "personal_gvhmr_gmr",
    "meaning": "motion-response reference with pelvis participation; not measured force/contact labels",
    "training_input": "physics-accepted measured execution only",
    "metrics": metrics,
    "files": paths,
  }
  path = directory / "metadata.json"
  atomic_json(path, metadata)
  return {
    "attempt": attempt,
    "passed": bool(metrics.get("passed")),
    "metadata": str(path.relative_to(output)),
    "metadata_sha256": sha256(path),
    "failed_checks": metrics.get("failed_checks", []),
  }


def verify_attempt(output: Path, record: dict):
  path = output / record["metadata"]
  if sha256(path) != record["metadata_sha256"]:
    raise ValueError(f"Attempt metadata changed: {path}")
  metadata = json.loads(path.read_text())
  for value in metadata["files"].values():
    if sha256(output / value["path"]) != value["sha256"]:
      raise ValueError(f"Attempt artifact changed: {value['path']}")
  return metadata


def coverage(records: dict[str, dict]) -> dict:
  counts = {subject: [0, 0, 0] for subject in SUBJECT_SPLITS}
  for record in records.values():
    if record.get("status") == "accepted":
      counts[record["subject_id"]][record["condition_id"]] += 1
  return {
    "counts_by_subject": counts,
    "passed": all(min(c) >= 1 for c in counts.values()),
  }


def config_dict(cfg: Config):
  return {
    **asdict(cfg),
    "stances": [list(x) for x in STANCES],
    "subject_splits": SUBJECT_SPLITS,
    "schema": SCHEMA,
  }
