"""Deterministic whole-body IK and physics validation for G1 shoulder yielding.

The generator intentionally has no dependency on historical LAFAN/G1 Moves/AMASS
shoulder positives.  A clip is accepted only after its reference motion and the
free-root MuJoCo execution satisfy the frozen kinematic, semantic, and dynamic gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mink
import mujoco
import numpy as np
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from scipy.spatial.transform import Rotation

from smp.pretrain.shoulder80 import CONDITION_NAMES

FPS = 50
SIM_DT = 0.005
CONTROL_DT = 0.02
IK_SUBSTEPS = 4
WINDOW_SIZE = 10
STAND_DURATION_S = 1.0
HOLD_EVALUATION_S = 1.0
WINDOW_STRIDE = 5

JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

STANCE_WIDTHS = (0.22, 0.24, 0.26)
KNEE_FLEXIONS = (0.55, 0.70)
YAW_MIN_DEG = 15.0
YAW_MAX_DEG = 70.0
MEAN_YAW_SPEED_MIN_DEG_S = 12.0
MEAN_YAW_SPEED_MAX_DEG_S = 18.0
SAMPLING_SCHEMA = "v2_lhs_continuous_yaw"
WAIST_YAW_TRACKING_KP = 180.0


@dataclass(frozen=True)
class ClipSpec:
  clip_id: str
  family_id: str
  stage: Literal["pilot", "formal", "smoke"]
  split: Literal["pilot", "train", "validation", "test", "smoke"]
  condition_id: int
  grid_index: int
  slot_index: int
  attempt: int
  seed: int
  yaw_amplitude: float
  entry_duration_s: float
  hold_duration_s: float
  stance_width_m: float
  knee_flexion_rad: float
  arm_variant: tuple[float, float, float, float]
  com_offset_y_m: float
  target_mean_yaw_speed_deg_s: float = 0.0
  yaw_min_deg: float = YAW_MIN_DEG
  yaw_max_deg: float = YAW_MAX_DEG

  @property
  def condition(self) -> str:
    return CONDITION_NAMES[self.condition_id]


@dataclass(frozen=True)
class Gates:
  foot_position_error_m: float = 0.002
  foot_orientation_error_rad: float = math.radians(1.0)
  min_root_height_m: float = 0.60
  max_root_tilt_rad: float = math.radians(20.0)
  max_pelvis_yaw_change_rad: float = 0.05
  max_root_displacement_m: float = 0.05
  min_foot_contact_fraction: float = 0.95
  max_foot_slip_m: float = 0.015
  max_foot_yaw_drift_rad: float = math.radians(2.0)
  max_joint_tracking_rmse_rad: float = 0.08
  max_waist_tracking_rmse_rad: float = 0.04
  max_waist_final_error_rad: float = 0.03
  waist_amplitude_range_tolerance_deg: float = 0.001
  max_effort_p99_ratio: float = 0.95
  max_effort_saturation_fraction: float = 0.01
  max_hold_root_speed_p95_m_s: float = 0.05
  max_hold_waist_speed_p95_rad_s: float = 0.10
  min_shoulder_retreat_m: float = 0.015
  shoulder_position_tolerance_m: float = 0.001
  max_hold_shoulder_position_deviation_m: float = 0.002
  self_penetration_m: float = 0.0001
  nonfoot_ground_penetration_m: float = 0.002


@dataclass
class Motion:
  qpos: np.ndarray
  phase_target: np.ndarray
  reference_progress: np.ndarray


@dataclass
class Execution:
  qpos: np.ndarray
  qvel: np.ndarray
  effort_ratio: np.ndarray
  left_contact: np.ndarray
  right_contact: np.ndarray
  any_support: np.ndarray
  left_foot_pos: np.ndarray
  right_foot_pos: np.ndarray
  left_foot_yaw: np.ndarray
  right_foot_yaw: np.ndarray
  collision_flags: dict[str, bool]
  metrics: dict[str, float | int | bool | list[int]]


def minimum_jerk(value: np.ndarray | float) -> np.ndarray | float:
  """Quintic interpolation with zero endpoint velocity and acceleration."""
  s = np.clip(value, 0.0, 1.0)
  return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _seed(*parts: object) -> int:
  value = "|".join(map(str, parts)).encode()
  return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _variant(seed: int, stage: str, family: str, slot: int, attempt: int):
  rng = np.random.default_rng(_seed(seed, stage, family, slot, attempt))
  arm = tuple(float(x) for x in rng.uniform(-1.0, 1.0, 4))
  hold = float(rng.uniform(1.2, 2.0))
  com_y = float(rng.uniform(-0.008, 0.008))
  return arm, hold, com_y


def _latin_hypercube(
  count: int,
  low: float,
  high: float,
  *seed_parts: object,
  edge_coverage: float | None = None,
) -> np.ndarray:
  """Return a deterministic randomized one-dimensional Latin hypercube."""
  if count <= 0 or not low < high:
    raise ValueError("Latin-hypercube bounds and count must be positive")
  rng = np.random.default_rng(_seed(*seed_parts))
  unit = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
  if edge_coverage is not None:
    if not 0.0 < edge_coverage <= 0.5 * (high - low):
      raise ValueError("Edge coverage must be positive and no more than half-range")
    edge_fraction = edge_coverage / (high - low)
    unit[0] = rng.uniform(0.0, min(1.0 / count, edge_fraction))
    unit[-1] = rng.uniform(max((count - 1.0) / count, 1.0 - edge_fraction), 1.0)
  rng.shuffle(unit)
  return low + (high - low) * unit


def _shoulder_group_counts(stage: str) -> tuple[tuple[str, int], ...]:
  if stage == "pilot":
    return (("pilot", 16),)
  if stage == "formal":
    return (("train", 112), ("validation", 24), ("test", 24))
  raise ValueError(f"Unsupported generation stage: {stage}")


def make_specs(
  stage: Literal["pilot", "formal"],
  seed: int = 42,
  *,
  yaw_min_deg: float = YAW_MIN_DEG,
  yaw_max_deg: float = YAW_MAX_DEG,
  mean_yaw_speed_min_deg_s: float = MEAN_YAW_SPEED_MIN_DEG_S,
  mean_yaw_speed_max_deg_s: float = MEAN_YAW_SPEED_MAX_DEG_S,
) -> list[ClipSpec]:
  """Create exact deterministic quotas with continuous stratified shoulder yaw."""
  if stage not in ("pilot", "formal"):
    raise ValueError(f"Unsupported generation stage: {stage}")
  if not 0.0 < yaw_min_deg < yaw_max_deg:
    raise ValueError("Yaw bounds must satisfy 0 < min < max")
  if not 0.0 < mean_yaw_speed_min_deg_s < mean_yaw_speed_max_deg_s:
    raise ValueError("Mean-yaw-speed bounds must satisfy 0 < min < max")
  specs: list[ClipSpec] = []
  stand_slots = 8 if stage == "pilot" else 80
  for stance_index, width in enumerate(STANCE_WIDTHS):
    for knee_index, knee in enumerate(KNEE_FLEXIONS):
      cell = stance_index * len(KNEE_FLEXIONS) + knee_index
      for slot in range(stand_slots):
        family = f"{stage}-stand-c{cell:02d}-v{slot:03d}"
        arm, hold, com_y = _variant(seed, stage, family, slot, 0)
        split = (
          "pilot"
          if stage == "pilot"
          else ("train" if slot < 56 else "validation" if slot < 68 else "test")
        )
        specs.append(
          ClipSpec(
            clip_id=f"{family}-a0",
            family_id=family,
            stage=stage,
            split=split,
            condition_id=0,
            grid_index=cell,
            slot_index=slot,
            attempt=0,
            seed=seed,
            yaw_amplitude=0.0,
            entry_duration_s=0.0,
            hold_duration_s=3.0,
            stance_width_m=width,
            knee_flexion_rad=knee,
            arm_variant=arm,
            com_offset_y_m=com_y,
            yaw_min_deg=yaw_min_deg,
            yaw_max_deg=yaw_max_deg,
          )
        )
  shoulder_pairs = 0
  for stance_index, width in enumerate(STANCE_WIDTHS):
    for knee_index, knee in enumerate(KNEE_FLEXIONS):
      cell = stance_index * len(KNEE_FLEXIONS) + knee_index
      for split, count in _shoulder_group_counts(stage):
        amplitudes_deg = _latin_hypercube(
          count,
          yaw_min_deg,
          yaw_max_deg,
          seed,
          SAMPLING_SCHEMA,
          stage,
          split,
          cell,
          "yaw",
          edge_coverage=1.0,
        )
        speeds_deg_s = _latin_hypercube(
          count,
          mean_yaw_speed_min_deg_s,
          mean_yaw_speed_max_deg_s,
          seed,
          SAMPLING_SCHEMA,
          stage,
          split,
          cell,
          "speed",
        )
        for slot, (amplitude_deg, speed_deg_s) in enumerate(
          zip(amplitudes_deg, speeds_deg_s, strict=True)
        ):
          family = f"{stage}-shoulder-c{cell:02d}-{split}-v{slot:03d}"
          arm, hold, com_y = _variant(seed, stage, family, slot, 0)
          entry_frames = int(math.ceil(amplitude_deg / speed_deg_s * FPS))
          entry_duration_s = entry_frames / FPS
          for condition in (1, 2):
            side = "left" if condition == 1 else "right"
            specs.append(
              ClipSpec(
                clip_id=f"{family}-{side}-a0",
                family_id=family,
                stage=stage,
                split=split,
                condition_id=condition,
                grid_index=cell,
                slot_index=slot,
                attempt=0,
                seed=seed,
                yaw_amplitude=math.radians(float(amplitude_deg)),
                entry_duration_s=entry_duration_s,
                hold_duration_s=hold,
                stance_width_m=width,
                knee_flexion_rad=knee,
                arm_variant=arm,
                com_offset_y_m=com_y,
                target_mean_yaw_speed_deg_s=float(speed_deg_s),
                yaw_min_deg=yaw_min_deg,
                yaw_max_deg=yaw_max_deg,
              )
            )
          shoulder_pairs += 1
  expected = 240 if stage == "pilot" else 2400
  expected_pairs = 96 if stage == "pilot" else 960
  if len(specs) != expected or shoulder_pairs != expected_pairs:
    raise AssertionError((len(specs), shoulder_pairs))
  return specs


def retry_spec(spec: ClipSpec) -> ClipSpec:
  if spec.attempt != 0:
    raise ValueError("Only one deterministic retry is allowed")
  attempt = 1
  arm, hold, com_y = _variant(
    spec.seed, spec.stage, spec.family_id, spec.slot_index, attempt
  )
  side = (
    "" if spec.condition_id == 0 else "-left" if spec.condition_id == 1 else "-right"
  )
  return ClipSpec(
    **{
      **asdict(spec),
      "clip_id": f"{spec.family_id}{side}-a1",
      "attempt": attempt,
      "arm_variant": arm,
      "hold_duration_s": 3.0 if spec.condition_id == 0 else hold,
      "com_offset_y_m": com_y,
    }
  )


def split_counts(specs: list[ClipSpec]) -> dict[str, list[int]]:
  result: dict[str, list[int]] = {}
  for split in sorted({spec.split for spec in specs}):
    result[split] = [
      sum(s.split == split and s.condition_id == condition for s in specs)
      for condition in range(3)
    ]
  return result


def _build_model(*, soft_limits: bool) -> mujoco.MjModel:
  entity = Entity(get_g1_robot_cfg())
  entity.spec.worldbody.add_geom(
    name="shoulder_ik_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0.0, 0.0, 0.1),
    contype=1,
    conaffinity=1,
    friction=(0.6, 0.005, 0.0001),
  )
  model = entity.spec.compile()
  model.opt.timestep = SIM_DT
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  if soft_limits:
    for name in JOINT_NAMES:
      joint_id = model.joint(name).id
      lo, hi = model.jnt_range[joint_id]
      center = 0.5 * (lo + hi)
      half = 0.45 * (hi - lo)
      model.jnt_range[joint_id] = (center - half, center + half)
  return model


def _qpos_ids(model: mujoco.MjModel) -> np.ndarray:
  return np.array(
    [int(model.jnt_qposadr[model.joint(name).id]) for name in JOINT_NAMES]
  )


def _dof_ids(model: mujoco.MjModel) -> np.ndarray:
  return np.array([int(model.jnt_dofadr[model.joint(name).id]) for name in JOINT_NAMES])


def _quat_yaw_wxyz(quat: np.ndarray) -> float:
  w, x, y, z = quat
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_tilt_wxyz(quat: np.ndarray) -> tuple[float, float]:
  w, x, y, z = quat
  roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
  return roll, pitch


def contract_to_qpos(
  root_pos: np.ndarray,
  root_rot_xyzw: np.ndarray,
  dof_pos: np.ndarray,
  model: mujoco.MjModel,
) -> np.ndarray:
  """Reconstruct MuJoCo qpos from the persisted G1 trajectory contract."""
  frames = len(root_pos)
  expected = ((frames, 4), (frames, len(JOINT_NAMES)))
  if (root_rot_xyzw.shape, dof_pos.shape) != expected:
    raise ValueError("Invalid G1 trajectory contract")
  qpos = np.zeros((frames, model.nq), dtype=np.float64)
  qpos[:, :3] = root_pos
  qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]
  qpos[:, _qpos_ids(model)] = dof_pos
  if not np.isfinite(qpos).all():
    raise ValueError("Nonfinite G1 trajectory contract")
  return qpos


def body_in_pelvis_heading(
  qpos: np.ndarray, model: mujoco.MjModel, body_name: str
) -> np.ndarray:
  """Return a body position relative to pelvis in each frame's heading frame."""
  data = mujoco.MjData(model)
  body_id = model.body(body_name).id
  pelvis_id = model.body("pelvis").id
  result = np.empty((len(qpos), 3), dtype=np.float64)
  for index, pose in enumerate(qpos):
    data.qpos[:] = pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    offset = data.xpos[body_id] - data.xpos[pelvis_id]
    yaw = _quat_yaw_wxyz(data.xquat[pelvis_id])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    result[index] = (
      cosine * offset[0] + sine * offset[1],
      -sine * offset[0] + cosine * offset[1],
      offset[2],
    )
  return result


def shoulder_endpoint_metrics(
  spec: ClipSpec,
  qpos: np.ndarray,
  model: mujoco.MjModel,
  gates: Gates | None = None,
) -> dict[str, float | bool | None]:
  """Evaluate one shoulder clip as an endpoint task, without phase windows."""
  if spec.condition_id not in (1, 2):
    raise ValueError("Endpoint metrics require a left or right shoulder clip")
  active_gates = gates or Gates()
  body_name = (
    "left_shoulder_roll_link" if spec.condition_id == 1 else "right_shoulder_roll_link"
  )
  local_position = body_in_pelvis_heading(qpos, model, body_name)
  retreat = local_position[0, 0] - local_position[:, 0]
  turn_start = int(round(STAND_DURATION_S * FPS))
  entry_frames = int(round(spec.entry_duration_s * FPS))
  deadline = min(len(retreat) - 1, turn_start + entry_frames - 1)
  threshold = (
    active_gates.min_shoulder_retreat_m - active_gates.shoulder_position_tolerance_m
  )
  hit = np.flatnonzero(retreat[turn_start : deadline + 1] >= threshold)
  completion_time = float((int(hit[0]) + 1) / FPS) if len(hit) else None
  hold_frames = min(int(round(HOLD_EVALUATION_S * FPS)), len(retreat))
  hold_position = local_position[-hold_frames:]
  hold_center = np.median(hold_position, axis=0)
  hold_deviation = np.linalg.norm(hold_position - hold_center, axis=1)
  hold_retreat = retreat[-hold_frames:]
  return {
    "shoulder_retreat_final_m": float(retreat[-1]),
    "shoulder_retreat_at_deadline_m": float(retreat[deadline]),
    "shoulder_retreat_hold_min_m": float(hold_retreat.min()),
    "shoulder_completion_time_s": completion_time,
    "shoulder_completion_deadline_s": float(spec.entry_duration_s),
    "shoulder_completion_margin_s": (
      None
      if completion_time is None
      else float(spec.entry_duration_s - completion_time)
    ),
    "hold_shoulder_position_max_deviation_m": float(hold_deviation.max()),
    "endpoint_position_reached": bool(
      retreat[-1] >= active_gates.min_shoulder_retreat_m
    ),
    "completion_time_passed": bool(completion_time is not None),
    "hold_position_passed": bool(
      hold_retreat.min() >= threshold
      and hold_deviation.max() <= active_gates.max_hold_shoulder_position_deviation_m
    ),
  }


def _rotation_error(a: np.ndarray, b: np.ndarray) -> float:
  # Inputs are MuJoCo wxyz quaternions.
  qa = np.asarray([a[1], a[2], a[3], a[0]])
  qb = np.asarray([b[1], b[2], b[3], b[0]])
  return float((Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)).magnitude())


def _set_joint(q: np.ndarray, model: mujoco.MjModel, name: str, value: float):
  q[int(model.jnt_qposadr[model.joint(name).id])] = value


def _target_pose(transform: mink.SE3, translation: np.ndarray | None = None, yaw=0.0):
  target_translation = (
    transform.translation().copy() if translation is None else translation.copy()
  )
  target_rotation = mink.SO3.from_z_radians(yaw) @ transform.rotation()
  return mink.SE3.from_rotation_and_translation(target_rotation, target_translation)


class ShoulderIkGenerator:
  """One-process generator; construct one instance per worker."""

  def __init__(
    self,
    gates: Gates | None = None,
    waist_yaw_tracking_kp: float = WAIST_YAW_TRACKING_KP,
  ):
    if waist_yaw_tracking_kp <= 0.0:
      raise ValueError("waist_yaw_tracking_kp must be positive")
    self.gates = gates or Gates()
    self.waist_yaw_tracking_kp = float(waist_yaw_tracking_kp)
    self.ik_model = _build_model(soft_limits=True)
    self.sim_model = _build_model(soft_limits=False)
    self.ik_joint_qpos = _qpos_ids(self.ik_model)
    self.sim_joint_qpos = _qpos_ids(self.sim_model)
    self.sim_joint_dof = _dof_ids(self.sim_model)

  def _ik_limits(self, model: mujoco.MjModel):
    collision_geoms = {
      model.geom(i).name
      for i in range(model.ngeom)
      if model.geom(i).name.endswith("_collision")
    }
    left_arm = sorted(
      name
      for name in collision_geoms
      if name.startswith("left_")
      and any(token in name for token in ("shoulder", "elbow", "wrist", "hand"))
    )
    right_arm = sorted(
      name
      for name in collision_geoms
      if name.startswith("right_")
      and any(token in name for token in ("shoulder", "elbow", "wrist", "hand"))
    )
    trunk_legs = sorted(
      name
      for name in collision_geoms
      if any(token in name for token in ("pelvis", "torso", "thigh"))
    )
    velocity = {name: 2.0 for name in JOINT_NAMES}
    return [
      mink.ConfigurationLimit(model, gain=0.9),
      mink.VelocityLimit(model, velocity),
      mink.FreeJointVelocityLimit(
        model, max_linear_velocity=0.15, max_angular_velocity=0.5
      ),
      mink.CollisionAvoidanceLimit(
        model,
        geom_pairs=[
          (left_arm, trunk_legs),
          (right_arm, trunk_legs),
          (left_arm, right_arm),
        ],
        minimum_distance_from_collisions=0.003,
        collision_detection_distance=0.05,
      ),
    ]

  def _stand_configuration(self, spec: ClipSpec):
    model = self.ik_model
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("init_state")
    target_q = configuration.q.copy()
    knee = spec.knee_flexion_rad
    for side in ("left", "right"):
      _set_joint(target_q, model, f"{side}_knee_joint", knee)
      _set_joint(target_q, model, f"{side}_hip_pitch_joint", -0.46 * knee)
      _set_joint(target_q, model, f"{side}_ankle_pitch_joint", -0.54 * knee)
    a0, a1, a2, a3 = spec.arm_variant
    _set_joint(target_q, model, "left_shoulder_pitch_joint", 0.20 + 0.04 * a0)
    _set_joint(target_q, model, "right_shoulder_pitch_joint", 0.20 + 0.04 * a1)
    _set_joint(target_q, model, "left_shoulder_roll_joint", 0.20 + 0.03 * a2)
    _set_joint(target_q, model, "right_shoulder_roll_joint", -0.20 - 0.03 * a2)
    _set_joint(target_q, model, "left_elbow_joint", 0.60 + 0.08 * a3)
    _set_joint(target_q, model, "right_elbow_joint", 0.60 - 0.08 * a3)

    left = mink.FrameTask("left_ankle_roll_link", "body", 20.0, 10.0, lm_damping=1.0)
    right = mink.FrameTask("right_ankle_roll_link", "body", 20.0, 10.0, lm_damping=1.0)
    left_start = configuration.get_transform_frame_to_world(
      "left_ankle_roll_link", "body"
    )
    right_start = configuration.get_transform_frame_to_world(
      "right_ankle_roll_link", "body"
    )
    left_pos, right_pos = (
      left_start.translation().copy(),
      right_start.translation().copy(),
    )
    foot_mid_x = 0.5 * (left_pos[0] + right_pos[0])
    left_pos[1], right_pos[1] = spec.stance_width_m / 2, -spec.stance_width_m / 2
    left_pos[2] -= 0.004
    right_pos[2] -= 0.004
    left.set_target(_target_pose(left_start, left_pos))
    right.set_target(_target_pose(right_start, right_pos))

    pelvis = mink.FrameTask(
      "pelvis",
      "body",
      position_cost=[0.5, 0.5, 1.0],
      orientation_cost=5.0,
      lm_damping=1.0,
    )
    pelvis_start = configuration.get_transform_frame_to_world("pelvis", "body")
    pelvis_pos = pelvis_start.translation().copy()
    pelvis_pos[0] = foot_mid_x + 0.028
    pelvis_pos[1] = 0.0
    pelvis_pos[2] = 0.76 + 0.06 * (0.669 - knee)
    pelvis.set_target(_target_pose(pelvis_start, pelvis_pos))

    posture = mink.PostureTask(model, cost=0.15, lm_damping=1.0)
    posture.set_target(target_q)
    com = mink.ComTask(cost=[20.0, 20.0, 1.0], lm_damping=1.0)
    com_target = configuration.data.subtree_com[1].copy()
    com_target[0] = foot_mid_x + 0.056
    com_target[1] = spec.com_offset_y_m
    com.set_target(com_target)
    limits = self._ik_limits(model)
    for _ in range(800):
      velocity = mink.solve_ik(
        configuration,
        [left, right, pelvis, posture, com],
        SIM_DT,
        "daqp",
        damping=1e-4,
        limits=limits,
      )
      configuration.integrate_inplace(velocity, SIM_DT)
    return configuration, left, right, pelvis, posture, com, limits

  def generate_reference(self, spec: ClipSpec) -> tuple[Motion, dict[str, float]]:
    (
      configuration,
      left_foot,
      right_foot,
      pelvis,
      posture,
      com,
      limits,
    ) = self._stand_configuration(spec)
    stand_q = configuration.q.copy()
    foot_position_error = max(
      np.linalg.norm(left_foot.compute_error(configuration)[:3]),
      np.linalg.norm(right_foot.compute_error(configuration)[:3]),
    )
    foot_orientation_error = max(
      np.linalg.norm(left_foot.compute_error(configuration)[3:]),
      np.linalg.norm(right_foot.compute_error(configuration)[3:]),
    )
    if spec.condition_id == 0:
      frames = int(round(3.0 * FPS))
      return (
        Motion(
          qpos=np.repeat(stand_q[None], frames, axis=0),
          phase_target=np.zeros(frames, dtype=np.int8),
          reference_progress=np.zeros(frames, dtype=np.float32),
        ),
        {
          "foot_position_error_m": float(foot_position_error),
          "foot_orientation_error_rad": float(foot_orientation_error),
        },
      )

    torso = mink.RelativeFrameTask(
      "torso_link",
      "body",
      "pelvis",
      "body",
      position_cost=[0.2, 0.2, 0.2],
      orientation_cost=[0.5, 0.5, 10.0],
      lm_damping=1.0,
    )
    torso_start = configuration.get_transform("torso_link", "body", "pelvis", "body")
    sign = 1.0 if spec.condition_id == 1 else -1.0
    # The floating-base equilibrium has a sub-milliradian asymmetric waist
    # offset. Pre-settle a visually neutral, direction-consistent pose so the
    # frozen signed-first-frame gate is not decided by numerical equilibrium
    # bias. The ramp still ends at the exact requested amplitude.
    neutral_yaw = 0.002
    waist_id = int(self.ik_model.jnt_qposadr[self.ik_model.joint("waist_yaw_joint").id])
    neutral_q = stand_q.copy()
    neutral_q[waist_id] = sign * neutral_yaw
    posture.set_target(neutral_q)
    torso.set_target(_target_pose(torso_start, yaw=sign * neutral_yaw))
    for _ in range(100):
      velocity = mink.solve_ik(
        configuration,
        [pelvis, torso, posture, com],
        SIM_DT,
        "daqp",
        damping=1e-4,
        limits=limits,
        constraints=[left_foot, right_foot],
      )
      configuration.integrate_inplace(velocity, SIM_DT)
    stand_q = configuration.q.copy()
    stand_frames = int(round(STAND_DURATION_S * FPS))
    entry_frames = int(round(spec.entry_duration_s * FPS))
    hold_frames = int(round(spec.hold_duration_s * FPS))
    total = stand_frames + entry_frames + hold_frames
    progress = np.zeros(total, dtype=np.float64)
    progress[stand_frames : stand_frames + entry_frames] = minimum_jerk(
      np.arange(1, entry_frames + 1) / entry_frames
    )
    progress[stand_frames + entry_frames :] = 1.0
    qpos = np.empty((total, self.ik_model.nq), dtype=np.float64)
    phase = np.zeros(total, dtype=np.int8)
    for frame, scalar in enumerate(progress):
      # ``yaw_amplitude`` is the requested change from the settled neutral pose,
      # not an absolute joint endpoint. This keeps the persisted 15--70 degree
      # distribution independent of the tiny direction-consistent pre-settle.
      yaw = sign * (neutral_yaw + spec.yaw_amplitude * scalar)
      target_q = stand_q.copy()
      target_q[waist_id] = yaw
      posture.set_target(target_q)
      torso.set_target(_target_pose(torso_start, yaw=yaw))
      for _ in range(IK_SUBSTEPS):
        velocity = mink.solve_ik(
          configuration,
          [pelvis, torso, posture, com],
          SIM_DT,
          "daqp",
          damping=1e-4,
          limits=limits,
          constraints=[left_foot, right_foot],
        )
        configuration.integrate_inplace(velocity, SIM_DT)
      qpos[frame] = configuration.q
      signed = sign * configuration.q[waist_id]
      if frame < stand_frames:
        phase[frame] = 0
      elif signed < 0.08:
        phase[frame] = 1
      elif scalar < 1.0:
        phase[frame] = 2
      else:
        phase[frame] = 3
    return (
      Motion(
        qpos=qpos, phase_target=phase, reference_progress=progress.astype(np.float32)
      ),
      {
        "foot_position_error_m": float(foot_position_error),
        "foot_orientation_error_rad": float(foot_orientation_error),
      },
    )

  def _controller_maps(self):
    model = self.sim_model
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    qpos_ids = np.array(
      [int(model.jnt_qposadr[model.joint(name).id]) for name in actuator_names]
    )
    dof_ids = np.array(
      [int(model.jnt_dofadr[model.joint(name).id]) for name in actuator_names]
    )
    effort_limits = np.array(
      [abs(float(model.actuator_forcerange[i, 1])) for i in range(model.nu)]
    )
    return qpos_ids, dof_ids, effort_limits

  def _track_step(
    self,
    data: mujoco.MjData,
    reference: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray],
  ):
    model = self.sim_model
    qpos_ids, dof_ids, effort_limits = maps
    mujoco.mj_forward(model, data)
    error = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, error, 1.0, data.qpos, reference)
    kp_dof = np.r_[np.full(3, 80.0), np.full(3, 100.0), np.full(29, 80.0)]
    waist_yaw_dof = int(model.jnt_dofadr[model.joint("waist_yaw_joint").id])
    kp_dof[waist_yaw_dof] = self.waist_yaw_tracking_kp
    kd_dof = 2.0 * np.sqrt(kp_dof)
    qdd_des = kp_dof * error - kd_dof * data.qvel
    blocks = []
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
      jac_pos = np.zeros((3, model.nv))
      jac_rot = np.zeros((3, model.nv))
      mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(body_name).id)
      blocks.extend((jac_pos, jac_rot))
    jacobian = np.vstack(blocks)
    gram = jacobian @ jacobian.T + 1e-6 * np.eye(12)
    qdd = qdd_des - jacobian.T @ np.linalg.solve(gram, jacobian @ qdd_des)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    required = mass @ qdd + data.qfrc_bias - data.qfrc_passive
    contact_wrench = np.linalg.lstsq(jacobian[:, :6].T, required[:6], rcond=1e-6)[0]
    torque = required[dof_ids] - jacobian[:, dof_ids].T @ contact_wrench
    torque = np.clip(torque, -0.95 * effort_limits, 0.95 * effort_limits)
    for index in range(model.nu):
      stiffness = float(model.actuator_gainprm[index, 0])
      damping = -float(model.actuator_biasprm[index, 2])
      data.ctrl[index] = (
        data.qpos[qpos_ids[index]]
        + (torque[index] + damping * data.qvel[dof_ids[index]]) / stiffness
      )
    mujoco.mj_step(model, data)
    return np.abs(data.actuator_force) / effort_limits

  def _contact_state(self, data: mujoco.MjData):
    model = self.sim_model
    floor = model.geom("shoulder_ik_floor").id
    left = right = False
    self_collision = nonfoot_ground = False
    for contact in data.contact:
      first, second = int(contact.geom[0]), int(contact.geom[1])
      if floor in (first, second):
        other = second if first == floor else first
        name = model.geom(other).name
        if name.startswith("left_foot"):
          left = True
        elif name.startswith("right_foot"):
          right = True
        elif contact.dist < -self.gates.nonfoot_ground_penetration_m:
          nonfoot_ground = True
      elif contact.dist < -self.gates.self_penetration_m:
        self_collision = True
    return left, right, self_collision, nonfoot_ground

  def execute(self, motion: Motion) -> Execution:
    model = self.sim_model
    data = mujoco.MjData(model)
    data.qpos[:] = motion.qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    maps = self._controller_maps()
    for _ in range(int(round(1.0 / SIM_DT))):
      self._track_step(data, motion.qpos[0], maps)

    qpos, qvel, efforts = [], [], []
    left_contact, right_contact, any_support = [], [], []
    left_pos, right_pos, left_yaw, right_yaw = [], [], [], []
    collision_flags = {"self_collision": False, "nonfoot_ground": False}
    left_body = model.body("left_ankle_roll_link").id
    right_body = model.body("right_ankle_roll_link").id
    all_references = list(motion.qpos) + [motion.qpos[-1]] * FPS
    for reference_index, reference in enumerate(all_references):
      ratios = []
      contacts = []
      for _ in range(IK_SUBSTEPS):
        ratios.append(self._track_step(data, reference, maps))
        state = self._contact_state(data)
        contacts.append(state[:2])
        collision_flags["self_collision"] |= state[2]
        collision_flags["nonfoot_ground"] |= state[3]
      if reference_index >= len(motion.qpos):
        continue
      qpos.append(data.qpos.copy())
      qvel.append(data.qvel.copy())
      efforts.append(np.max(ratios, axis=0))
      left_seen = any(item[0] for item in contacts)
      right_seen = any(item[1] for item in contacts)
      left_contact.append(left_seen)
      right_contact.append(right_seen)
      any_support.append(left_seen or right_seen)
      left_pos.append(data.xpos[left_body].copy())
      right_pos.append(data.xpos[right_body].copy())
      left_yaw.append(_quat_yaw_wxyz(data.xquat[left_body]))
      right_yaw.append(_quat_yaw_wxyz(data.xquat[right_body]))
    execution = Execution(
      qpos=np.asarray(qpos),
      qvel=np.asarray(qvel),
      effort_ratio=np.asarray(efforts),
      left_contact=np.asarray(left_contact),
      right_contact=np.asarray(right_contact),
      any_support=np.asarray(any_support),
      left_foot_pos=np.asarray(left_pos),
      right_foot_pos=np.asarray(right_pos),
      left_foot_yaw=np.unwrap(np.asarray(left_yaw)),
      right_foot_yaw=np.unwrap(np.asarray(right_yaw)),
      collision_flags=collision_flags,
      metrics={},
    )
    return execution

  def _body_trajectory(self, qpos: np.ndarray, body_name: str) -> np.ndarray:
    model, data = self.sim_model, mujoco.MjData(self.sim_model)
    body_id = model.body(body_name).id
    result = np.empty((len(qpos), 3), dtype=np.float64)
    for index, pose in enumerate(qpos):
      data.qpos[:] = pose
      data.qvel[:] = 0.0
      mujoco.mj_forward(model, data)
      result[index] = data.xpos[body_id]
    return result

  def validate(
    self,
    spec: ClipSpec,
    motion: Motion,
    execution: Execution,
    ik_metrics: dict[str, float],
  ) -> tuple[bool, dict[str, Any]]:
    gates = self.gates
    model = self.sim_model
    qpos_ids = self.sim_joint_qpos
    reference_joints = motion.qpos[:, qpos_ids]
    actual_joints = execution.qpos[:, qpos_ids]
    errors = actual_joints - reference_joints
    waist_index = JOINT_NAMES.index("waist_yaw_joint")
    sign = 0.0 if spec.condition_id == 0 else 1.0 if spec.condition_id == 1 else -1.0
    roll_pitch = np.asarray([_quat_tilt_wxyz(q[3:7]) for q in execution.qpos])
    root_yaw = np.unwrap(np.asarray([_quat_yaw_wxyz(q[3:7]) for q in execution.qpos]))
    root_displacement = np.linalg.norm(
      execution.qpos[:, :2] - execution.qpos[0, :2], axis=1
    )
    root_speed = np.linalg.norm(execution.qvel[:, :2], axis=1)
    # Stability is evaluated only on the final settled second, not on a
    # hand-authored sequence of early/strong/hold phase windows.
    hold_mask = np.zeros(len(motion.phase_target), dtype=bool)
    hold_frames = min(int(round(HOLD_EVALUATION_S * FPS)), len(hold_mask))
    hold_mask[-hold_frames:] = True
    waist_velocity = execution.qvel[:, self.sim_joint_dof[waist_index]]
    stand_frames = min(int(round(STAND_DURATION_S * FPS)), len(actual_joints))
    stand_waist_rad = float(np.median(actual_joints[:stand_frames, waist_index]))
    actual_waist_amplitude_rad = (
      0.0
      if spec.condition_id == 0
      else float(sign * (actual_joints[-1, waist_index] - stand_waist_rad))
    )
    amplitude_error_rad = (
      0.0
      if spec.condition_id == 0
      else float(abs(actual_waist_amplitude_rad - spec.yaw_amplitude))
    )
    endpoint_metrics = (
      shoulder_endpoint_metrics(spec, execution.qpos, model, gates)
      if spec.condition_id
      else {}
    )

    joint_ranges = model.jnt_range[[model.joint(name).id for name in JOINT_NAMES]]
    centers = joint_ranges.mean(axis=1)
    half = 0.45 * (joint_ranges[:, 1] - joint_ranges[:, 0])
    soft_limit = bool(
      ((actual_joints >= centers - half) & (actual_joints <= centers + half)).all()
    )
    effort_p99 = float(np.quantile(execution.effort_ratio, 0.99))
    saturation_fraction = float((execution.effort_ratio >= 0.95 - 1e-6).mean())
    metrics: dict[str, Any] = {
      **ik_metrics,
      "finite": bool(
        np.isfinite(motion.qpos).all()
        and np.isfinite(execution.qpos).all()
        and np.isfinite(execution.qvel).all()
      ),
      "soft_joint_limits": soft_limit,
      "root_height_min_m": float(execution.qpos[:, 2].min()),
      "root_roll_pitch_abs_max_rad": float(np.abs(roll_pitch).max()),
      "pelvis_yaw_change_abs_max_rad": float(np.abs(root_yaw - root_yaw[0]).max()),
      "root_displacement_max_m": float(root_displacement.max()),
      "left_contact_fraction": float(execution.left_contact.mean()),
      "right_contact_fraction": float(execution.right_contact.mean()),
      "any_support_all_frames": bool(execution.any_support.all()),
      "left_foot_slip_m": float(
        np.linalg.norm(
          execution.left_foot_pos[:, :2] - execution.left_foot_pos[0, :2], axis=1
        ).max()
      ),
      "right_foot_slip_m": float(
        np.linalg.norm(
          execution.right_foot_pos[:, :2] - execution.right_foot_pos[0, :2], axis=1
        ).max()
      ),
      "left_foot_yaw_drift_rad": float(
        np.abs(execution.left_foot_yaw - execution.left_foot_yaw[0]).max()
      ),
      "right_foot_yaw_drift_rad": float(
        np.abs(execution.right_foot_yaw - execution.right_foot_yaw[0]).max()
      ),
      "joint_tracking_rmse_rad": float(np.sqrt(np.mean(errors**2))),
      "waist_tracking_rmse_rad": float(np.sqrt(np.mean(errors[:, waist_index] ** 2))),
      "waist_final_error_rad": float(abs(errors[-1, waist_index])),
      "effort_p99_ratio": effort_p99,
      "effort_saturation_fraction": saturation_fraction,
      "max_motor_velocity_ratio": float(self._motor_velocity_ratio(execution.qvel)),
      "hold_root_speed_p95_m_s": float(np.quantile(root_speed[hold_mask], 0.95)),
      "hold_waist_speed_p95_rad_s": float(
        np.quantile(np.abs(waist_velocity[hold_mask]), 0.95)
      ),
      "self_collision": execution.collision_flags["self_collision"],
      "nonfoot_ground": execution.collision_flags["nonfoot_ground"],
      "actual_final_waist_rad": float(actual_joints[-1, waist_index]),
      "signed_final_waist_rad": float(sign * actual_joints[-1, waist_index]),
      "stand_waist_baseline_rad": stand_waist_rad,
      "target_waist_amplitude_rad": float(spec.yaw_amplitude),
      "target_waist_amplitude_deg": float(math.degrees(spec.yaw_amplitude)),
      "actual_waist_amplitude_rad": actual_waist_amplitude_rad,
      "actual_waist_amplitude_deg": float(math.degrees(actual_waist_amplitude_rad)),
      "waist_amplitude_target_error_rad": amplitude_error_rad,
      "target_mean_yaw_speed_deg_s": float(spec.target_mean_yaw_speed_deg_s),
      "waist_yaw_tracking_kp": float(self.waist_yaw_tracking_kp),
      "realized_target_mean_yaw_speed_deg_s": (
        0.0
        if spec.condition_id == 0
        else float(math.degrees(spec.yaw_amplitude) / spec.entry_duration_s)
      ),
      **endpoint_metrics,
    }
    checks = {
      "finite": metrics["finite"],
      "soft_joint_limits": soft_limit,
      "foot_position_error": ik_metrics["foot_position_error_m"]
      <= gates.foot_position_error_m,
      "foot_orientation_error": ik_metrics["foot_orientation_error_rad"]
      <= gates.foot_orientation_error_rad,
      "root_height": metrics["root_height_min_m"] >= gates.min_root_height_m,
      "root_tilt": metrics["root_roll_pitch_abs_max_rad"] <= gates.max_root_tilt_rad,
      "pelvis_yaw": metrics["pelvis_yaw_change_abs_max_rad"]
      <= gates.max_pelvis_yaw_change_rad,
      "root_displacement": metrics["root_displacement_max_m"]
      <= gates.max_root_displacement_m,
      "left_contact": metrics["left_contact_fraction"]
      >= gates.min_foot_contact_fraction,
      "right_contact": metrics["right_contact_fraction"]
      >= gates.min_foot_contact_fraction,
      "continuous_support": metrics["any_support_all_frames"],
      "foot_slip": max(metrics["left_foot_slip_m"], metrics["right_foot_slip_m"])
      <= gates.max_foot_slip_m,
      "foot_yaw": max(
        metrics["left_foot_yaw_drift_rad"], metrics["right_foot_yaw_drift_rad"]
      )
      <= gates.max_foot_yaw_drift_rad,
      "joint_tracking": metrics["joint_tracking_rmse_rad"]
      <= gates.max_joint_tracking_rmse_rad,
      "waist_tracking": metrics["waist_tracking_rmse_rad"]
      <= gates.max_waist_tracking_rmse_rad,
      "waist_final": metrics["waist_final_error_rad"]
      <= gates.max_waist_final_error_rad,
      "effort_p99": effort_p99 <= gates.max_effort_p99_ratio + 1e-6,
      "effort_saturation": saturation_fraction <= gates.max_effort_saturation_fraction,
      "motor_velocity": metrics["max_motor_velocity_ratio"] <= 1.0,
      "hold_root_speed": metrics["hold_root_speed_p95_m_s"]
      <= gates.max_hold_root_speed_p95_m_s,
      "hold_waist_speed": metrics["hold_waist_speed_p95_rad_s"]
      <= gates.max_hold_waist_speed_p95_rad_s,
      "collision": not metrics["self_collision"],
      "ground": not metrics["nonfoot_ground"],
    }
    if spec.condition_id == 0:
      checks["stand_waist"] = abs(metrics["actual_final_waist_rad"]) <= 0.06
    else:
      checks["waist_amplitude_direction"] = actual_waist_amplitude_rad > 0.0
      range_tolerance = math.radians(gates.waist_amplitude_range_tolerance_deg)
      checks["waist_amplitude_range"] = (
        math.radians(spec.yaw_min_deg) - range_tolerance
        <= actual_waist_amplitude_rad
        <= math.radians(spec.yaw_max_deg) + range_tolerance
      )
      checks["waist_amplitude_target"] = (
        amplitude_error_rad <= gates.max_waist_final_error_rad
      )
      checks["endpoint_shoulder_position"] = metrics["endpoint_position_reached"]
      checks["completion_time"] = metrics["completion_time_passed"]
      checks["hold_shoulder_position"] = metrics["hold_position_passed"]
    metrics["checks"] = checks
    metrics["failed_checks"] = sorted(
      name for name, passed in checks.items() if not passed
    )
    metrics["passed"] = bool(all(checks.values()))
    execution.metrics = metrics
    return metrics["passed"], metrics

  def _motor_velocity_ratio(self, qvel: np.ndarray) -> float:
    limits = []
    for name in JOINT_NAMES:
      if "wrist_pitch" in name or "wrist_yaw" in name:
        limits.append(22.0)
      elif "hip_roll" in name or "knee" in name:
        limits.append(20.0)
      elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
        limits.append(32.0)
      else:
        limits.append(37.0)
    velocity = np.abs(qvel[:, self.sim_joint_dof])
    return float((velocity / np.asarray(limits)[None]).max())

  def produce(self, spec: ClipSpec) -> tuple[bool, Motion, Execution, dict[str, Any]]:
    motion, ik_metrics = self.generate_reference(spec)
    execution = self.execute(motion)
    passed, metrics = self.validate(spec, motion, execution, ik_metrics)
    return passed, motion, execution, metrics


def qpos_to_contract(qpos: np.ndarray, model: mujoco.MjModel):
  root_pos = qpos[:, :3].astype(np.float32)
  root_wxyz = qpos[:, 3:7]
  root_xyzw = root_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
  joints = qpos[:, _qpos_ids(model)].astype(np.float32)
  return root_pos, root_xyzw, joints


def qpos_to_csv(qpos: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
  root, quat, joints = qpos_to_contract(qpos, model)
  return np.concatenate((root, quat, joints), axis=1)


def classify_condition_windows(
  motion: np.ndarray,
  condition_id: int,
  turn_start_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Assign only stand/left/right labels; no motion phase is inferred.

  A shoulder clip's initial neutral windows are stand. Windows fully after the
  configured turn onset carry the clip shoulder condition. The single window
  crossing the condition boundary is excluded rather than given a mixed label.
  """
  if motion.ndim != 2 or motion.shape[1] != 36 or len(motion) < WINDOW_SIZE:
    raise ValueError("Expected a finite (T,36) motion with at least ten frames")
  if not np.isfinite(motion).all() or condition_id not in (0, 1, 2):
    raise ValueError("Invalid motion or condition")
  starts = np.arange(0, len(motion) - WINDOW_SIZE + 1, WINDOW_STRIDE)
  labels = np.full(len(starts), -1, dtype=np.int8)
  if condition_id == 0:
    labels[:] = 0
    return starts, labels
  onset = (
    int(round(STAND_DURATION_S * FPS))
    if turn_start_frame is None
    else int(turn_start_frame)
  )
  if onset < 0 or onset >= len(motion):
    raise ValueError("Turn onset lies outside the trajectory")
  labels[starts + WINDOW_SIZE <= onset] = 0
  labels[starts >= onset] = condition_id
  return starts, labels


def classify_phase_windows(
  motion: np.ndarray, condition_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Label safe 10-frame windows from an executed clip.

  Returns frame starts, condition IDs, and phase IDs.  Rejected or ambiguous
  windows have condition/phase ``-1``.  Shoulder clips may therefore yield
  condition-0 neutral windows before the requested turn starts.
  """
  if motion.ndim != 2 or motion.shape[1] != 36 or len(motion) < WINDOW_SIZE:
    raise ValueError("Expected a finite (T,36) motion with at least ten frames")
  if not np.isfinite(motion).all() or condition_id not in (0, 1, 2):
    raise ValueError("Invalid motion or condition")
  starts = np.arange(0, len(motion) - WINDOW_SIZE + 1, WINDOW_STRIDE)
  index = starts[:, None] + np.arange(WINDOW_SIZE)
  position = motion[:, :3]
  velocity = np.gradient(position, 1.0 / FPS, axis=0)
  rotations = Rotation.from_quat(motion[:, 3:7])
  yaw = np.unwrap(rotations.as_euler("xyz")[:, 2])
  waist = motion[:, 7 + JOINT_NAMES.index("waist_yaw_joint")]
  pelvis_delta = yaw[index[:, -1]] - yaw[index[:, 0]]
  speed = np.linalg.norm(velocity[index, :2], axis=-1).max(axis=1)
  joint_velocity = np.gradient(motion[:, 7:], 1.0 / FPS, axis=0)
  joint_rms = np.sqrt(np.mean(joint_velocity[index] ** 2, axis=(1, 2)))
  stand = (
    (speed <= 0.08)
    & (np.abs(waist[index]).max(axis=1) <= 0.06)
    & (np.abs(waist[index[:, -1]] - waist[index[:, 0]]) <= 0.005)
    & (np.abs(pelvis_delta) <= 0.05)
    & (joint_rms <= 0.25)
  )
  labels = np.full(len(starts), -1, dtype=np.int8)
  phases = np.full(len(starts), -1, dtype=np.int8)
  labels[stand] = 0
  phases[stand] = 0
  if condition_id == 0:
    return starts, labels, phases
  sign = 1.0 if condition_id == 1 else -1.0
  signed = sign * waist[index]
  delta = signed[:, -1] - signed[:, 0]
  stable = (speed <= 0.20) & (np.abs(pelvis_delta) <= 0.05)
  shoulder_masks = (
    stable & (signed[:, 0] >= 0) & (signed[:, -1] < 0.08) & (delta >= 0.01),
    stable & (signed[:, 0] >= 0) & (signed[:, -1] >= 0.08) & (delta >= 0.02),
    stable
    & (signed[:, -1] >= 0.12)
    & (signed.min(axis=1) >= 0.10)
    & (np.abs(delta) <= 0.01),
  )
  for phase, mask in enumerate(shoulder_masks, start=1):
    labels[mask] = condition_id
    phases[mask] = phase
  return starts, labels, phases


def count_phase_windows(motion: np.ndarray, condition_id: int) -> list[int]:
  _, labels, phases = classify_phase_windows(motion, condition_id)
  counts = np.zeros(4, dtype=int)
  for phase in range(4):
    expected = 0 if phase == 0 else condition_id
    counts[phase] = int(((labels == expected) & (phases == phase)).sum())
  return counts.tolist()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json_default(value: Any):
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, Path):
    return str(value)
  raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def atomic_json(path: Path, payload: Any):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(
      payload,
      indent=2,
      allow_nan=False,
      default=_json_default,
    )
    + "\n"
  )
  os.replace(temporary, path)


def write_result(
  output: Path,
  spec: ClipSpec,
  passed: bool,
  motion: Motion,
  execution: Execution,
  metrics: dict[str, Any],
  model: mujoco.MjModel,
) -> dict[str, Any]:
  category = "accepted" if passed else "rejected"
  directory = output / "clips" / category / spec.split / spec.condition
  directory.mkdir(parents=True, exist_ok=True)
  stem = directory / spec.clip_id
  ref_root, ref_quat, ref_joints = qpos_to_contract(motion.qpos, model)
  exe_root, exe_quat, exe_joints = qpos_to_contract(execution.qpos, model)
  reference_path = stem.with_suffix(".reference.npz")
  execution_path = stem.with_suffix(".npz")
  csv_path = stem.with_suffix(".csv")
  np.savez_compressed(
    reference_path,
    fps=np.float32(FPS),
    root_pos=ref_root,
    root_rot=ref_quat,
    dof_pos=ref_joints,
    phase_target=motion.phase_target,
    reference_progress=motion.reference_progress,
  )
  np.savez_compressed(
    execution_path,
    fps=np.float32(FPS),
    root_pos=exe_root,
    root_rot=exe_quat,
    dof_pos=exe_joints,
    qvel=execution.qvel.astype(np.float32),
    phase_target=motion.phase_target,
    effort_ratio=execution.effort_ratio.astype(np.float32),
  )
  np.savetxt(csv_path, qpos_to_csv(execution.qpos, model), delimiter=",", fmt="%.9g")
  metadata = {
    "schema_version": 1,
    "spec": asdict(spec),
    "condition": spec.condition,
    "condition_id": spec.condition_id,
    "fps": FPS,
    "frames": len(execution.qpos),
    "coordinate_system": "+x forward, +y left, +z up",
    "quaternion_order": "xyzw",
    "joint_order": "smp_g1_29_v1",
    "training_input": "executed free-root MuJoCo trajectory",
    "metrics": metrics,
    "passed": passed,
    "reference_npz_sha256": sha256(reference_path),
    "execution_npz_sha256": sha256(execution_path),
    "csv_sha256": sha256(csv_path),
  }
  metadata_path = stem.with_suffix(".json")
  atomic_json(metadata_path, metadata)
  return {
    "clip_id": spec.clip_id,
    "family_id": spec.family_id,
    "split": spec.split,
    "condition_id": spec.condition_id,
    "condition": spec.condition,
    "attempt": spec.attempt,
    "passed": passed,
    "target_yaw_amplitude_deg": float(math.degrees(spec.yaw_amplitude)),
    "actual_waist_amplitude_deg": float(metrics["actual_waist_amplitude_deg"]),
    "target_mean_yaw_speed_deg_s": float(spec.target_mean_yaw_speed_deg_s),
    "entry_duration_s": float(spec.entry_duration_s),
    "metadata": str(metadata_path.relative_to(output)),
    "failed_checks": metrics["failed_checks"],
  }


def write_window_index(path: Path, rows: list[dict[str, Any]]):
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
