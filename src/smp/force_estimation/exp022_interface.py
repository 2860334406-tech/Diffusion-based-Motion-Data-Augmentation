"""Observable input and contact-observation adapters for EXP-022.

The extra observation groups do not change the existing actor/critic tensors.
They reproduce the SuperMimic 10-frame, term-major proprioceptive input and
frame-end joint-space actuator-force history. Solver contact is not read here.

``Exp022ContactAdapter`` converts a frozen estimator's torso-frame resultant
and point to the chest policy's 8D heading-frame contact observation. Its
force scalar is the *magnitude of the estimated resultant*. EXP-022's oracle
scalar is a sum of contact normal magnitudes, so exact equality is not implied
when contacts have different normals or tangential components.
"""

from __future__ import annotations

import torch
from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

G1_JOINT_NAMES = (
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


def joint_space_actuator_force(env) -> torch.Tensor:
  """The SuperMimic label's 29 joint DoF forces, not motor-space ``actuator_force``."""
  return env.scene["robot"].data.qfrc_actuator


def add_force_estimator_observations(env_cfg) -> None:
  """Add two read-only groups before constructing the mjlab environment.

  ``force_estimator_x`` is [B,930]. ``force_estimator_tau`` is [B,10,29].
  mjlab's history buffer backfills reset frames and orders history oldest-first.
  The original estimator was trained with SuperMimic observation corruption;
  inference here deliberately uses deterministic observable values.
  """
  for name in ("force_estimator_x", "force_estimator_tau"):
    if name in env_cfg.observations:
      raise ValueError(f"Observation group already exists: {name}")
  env_cfg.observations["force_estimator_x"] = ObservationGroupCfg(
    terms={
      "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}),
      "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
      "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
      "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
      "actions": ObservationTermCfg(func=mdp.last_action),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=10,
    flatten_history_dim=True,
  )
  env_cfg.observations["force_estimator_tau"] = ObservationGroupCfg(
    terms={"joint_force": ObservationTermCfg(func=joint_space_actuator_force)},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=10,
    flatten_history_dim=False,
  )


def validate_force_estimator_input(env, observations: dict) -> None:
  """Fail loudly if a G1/runtime layout differs from the frozen model contract."""
  robot = env.scene["robot"]
  if tuple(robot.joint_names) != G1_JOINT_NAMES:
    raise ValueError("G1 joint order differs from SuperMimic force-estimator training")
  action_names = tuple(env.action_manager.get_term("joint_pos").target_names)
  if action_names != G1_JOINT_NAMES:
    raise ValueError("Action order differs from SuperMimic force-estimator training")
  x = observations["force_estimator_x"]
  tau = observations["force_estimator_tau"]
  if x.shape != (env.num_envs, 930) or tau.shape != (env.num_envs, 10, 29):
    raise ValueError(
      f"Unexpected estimator input shapes: {tuple(x.shape)}, {tuple(tau.shape)}"
    )
  if not torch.isfinite(x).all() or not torch.isfinite(tau).all():
    raise ValueError("Non-finite estimator input")


class Exp022ContactAdapter:
  """Turn estimated torso-frame force/point into the policy's 8D contact term.

  Call ``reset`` for each episode boundary and ``step`` once per 20 ms control
  step. ``history`` is the 32D actor/critic term, oldest frame first. The
  estimation model must be stepped separately with the two observation groups.
  """

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    *,
    robot_weight_n: float,
    ema_alpha: float = 0.8,
    force_on_ratio: float = 0.0075,
    force_obs_clip: float = 0.4,
  ) -> None:
    if (
      num_envs < 1
      or robot_weight_n <= 0
      or not 0 <= ema_alpha < 1
      or force_on_ratio < 0
      or force_obs_clip <= 0
    ):
      raise ValueError("Invalid contact adapter dimensions or filter parameters")
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.weight = float(robot_weight_n)
    self.ema_alpha = ema_alpha
    self.force_on_ratio = force_on_ratio
    self.force_obs_clip = force_obs_clip
    self.filtered_force = torch.zeros(num_envs, device=self.device)
    self.last_direction_local = torch.zeros(num_envs, 3, device=self.device)
    self.last_position_local = torch.zeros(num_envs, 3, device=self.device)
    self._history = CircularBuffer(
      max_len=4, batch_size=num_envs, device=str(self.device)
    )
    self._history.append(torch.zeros(num_envs, 8, device=self.device))

  @classmethod
  def from_controller(cls, controller) -> Exp022ContactAdapter:
    """Copy the live EXP-022 force-observation constants and robot weight."""
    return cls(
      controller.num_envs,
      controller.device,
      robot_weight_n=controller.weight,
      ema_alpha=controller.cfg.ema_alpha,
      force_on_ratio=controller.cfg.force_on_ratio,
      force_obs_clip=controller.cfg.force_obs_clip,
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    else:
      env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    self.filtered_force[env_ids] = 0
    self.last_direction_local[env_ids] = 0
    self.last_position_local[env_ids] = 0
    self._history.reset(env_ids)
    self._history.backfill(torch.zeros(self.num_envs, 8, device=self.device), env_ids)

  @property
  def history(self) -> torch.Tensor:
    return self._history.buffer.reshape(self.num_envs, 32)

  def step(
    self,
    force_body_n: torch.Tensor,
    point_body_m: torch.Tensor,
    *,
    torso_quat_w: torch.Tensor,
    torso_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
  ) -> torch.Tensor:
    vectors = (force_body_n, point_body_m, torso_pos_w, root_pos_w)
    quats = (torso_quat_w, root_quat_w)
    if any(v.shape != (self.num_envs, 3) for v in vectors) or any(
      q.shape != (self.num_envs, 4) for q in quats
    ):
      raise ValueError("Expected batched torso/root 3D vectors and wxyz quaternions")
    if any(v.device != self.device for v in (*vectors, *quats)):
      raise ValueError("Estimator output and poses must be on the adapter device")
    if not all(torch.isfinite(v).all() for v in (*vectors, *quats)):
      raise ValueError("Non-finite estimator output or pose")

    magnitude = torch.linalg.vector_norm(force_body_n, dim=-1)
    heading_quat = yaw_quat(root_quat_w)
    force_local = quat_apply_inverse(
      heading_quat, quat_apply(torso_quat_w, force_body_n)
    )
    direction_local = force_local / magnitude.clamp_min(1e-8)[:, None]
    position_w = torso_pos_w + quat_apply(torso_quat_w, point_body_m)
    position_local = quat_apply_inverse(heading_quat, position_w - root_pos_w)
    valid = magnitude > 1e-8
    self.last_direction_local = torch.where(
      valid[:, None], direction_local, self.last_direction_local
    )
    self.last_position_local = torch.where(
      valid[:, None], position_local, self.last_position_local
    )
    self.filtered_force = (
      self.ema_alpha * self.filtered_force + (1 - self.ema_alpha) * magnitude
    )
    active = self.filtered_force > self.force_on_ratio * self.weight
    frame = torch.zeros(self.num_envs, 8, device=self.device)
    frame[:, 0] = active.float()
    frame[:, 1] = (
      torch.clamp(self.filtered_force / self.weight, 0, self.force_obs_clip) * active
    )
    frame[:, 2:5] = self.last_direction_local * active[:, None]
    frame[:, 5:8] = self.last_position_local * active[:, None]
    self._history.append(frame)
    return frame
