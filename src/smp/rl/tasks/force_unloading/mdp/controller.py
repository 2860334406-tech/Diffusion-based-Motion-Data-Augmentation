"""Rail-mounted spherical pusher and force-unloading event controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Literal

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from smp.rl.tasks.force_unloading.mdp.conditions import (
  capsule_start_distance,
  initialize_conditions,
  reset_conditions,
  update_conditions,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class RailPhase(IntEnum):
  WAIT = 0
  DRIVE = 1
  RETRACT = 2
  DONE = 3


class PushTarget(IntEnum):
  CHEST = 0
  LEFT_SHOULDER = 1
  RIGHT_SHOULDER = 2
  BACK = 3


def canonical_drive_direction(
  targets: torch.Tensor,
  mode: Literal["target_default", "front_to_back"],
) -> torch.Tensor:
  """Return heading-local rail directions for the selected contact targets."""
  if mode == "front_to_back":
    direction = torch.zeros((len(targets), 2), device=targets.device)
    direction[:, 0] = torch.where(targets == int(PushTarget.BACK), 1.0, -1.0)
    return direction
  canonical = torch.tensor(
    ((-1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 0.0)),
    device=targets.device,
  )
  return canonical[targets]


@dataclass(frozen=True)
class RailPusherCfg:
  """Physical, event, and safety parameters for the first-stage baseline."""

  sphere_radius: float = 0.175
  sphere_mass: float = 5.0
  friction: float = 0.2
  reference_speed: float = 0.05
  stiffness: float = 700.0
  damping: float = 2.0 * math.sqrt(700.0 * 5.0)
  max_force_weight_ratio: float = 0.15
  max_travel: float = 0.375
  ramp_time: float = 0.25
  retract_speed: float = 0.15
  start_distance: float = 0.35
  start_surface_gap: float | None = None
  conditional: bool = False
  condition_transition_mode: Literal["null_until_stable", "masked_current"] = (
    "null_until_stable"
  )
  wait_range_s: tuple[float, float] = (1.0, 3.0)
  target_probabilities: tuple[float, ...] = (1.0, 0.0, 0.0)
  push_direction_mode: Literal["target_default", "front_to_back"] = "target_default"
  stationary_reward_weight: float = 0.1
  stationary_speed_scale: float = 1.0
  ema_alpha: float = 0.8
  force_obs_clip: float = 0.4
  force_on_ratio: float = 0.0075
  force_done_ratio: float = 0.03
  force_rate_done_ratio_s: float = 0.05
  force_emergency_ratio: float = 0.30
  contact_latch_steps: int = 3
  emergency_substeps: int = 3
  min_drive_time_s: float = 0.5
  min_reference_travel: float = 0.03
  stable_time_s: float = 0.4

  def __post_init__(self) -> None:
    if len(self.target_probabilities) not in (3, 4):
      raise ValueError("Expected three legacy or four contact target probabilities")
    if self.start_surface_gap is not None and self.start_surface_gap <= 0:
      raise ValueError("start_surface_gap must be positive")
    if not 0.0 <= self.stationary_reward_weight <= 1.0:
      raise ValueError("stationary_reward_weight must be in [0, 1]")

    if self.stationary_speed_scale <= 0.0:
      raise ValueError("stationary_speed_scale must be positive")
    if self.condition_transition_mode not in ("null_until_stable", "masked_current"):
      raise ValueError("Unsupported condition_transition_mode")
    if any(probability < 0.0 for probability in self.target_probabilities):
      raise ValueError("target_probabilities must be non-negative")
    if sum(self.target_probabilities) <= 0.0:
      raise ValueError("target_probabilities must contain a positive value")


def get_rail_pusher_spec(
  radius: float = 0.175,
  mass: float = 5.0,
  friction: float = 0.2,
  max_travel: float = 0.375,
) -> mujoco.MjSpec:
  """Create a one-DoF pusher. mjlab wraps its fixed base in a mocap body."""
  spec = mujoco.MjSpec()
  slider = spec.worldbody.add_body(name="slider")
  slider.add_joint(
    name="rail_joint",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=(1.0, 0.0, 0.0),
    limited=True,
    range=(0.0, max_travel),
    damping=0.0,
  )
  slider.add_geom(
    name="sphere",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    mass=mass,
    friction=(friction, 0.005, 0.0001),
    rgba=(0.9, 0.22, 0.08, 1.0),
    solref=(0.01, 1.0),
  )
  return spec


def rail_pusher_entity_cfg(cfg: RailPusherCfg) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: get_rail_pusher_spec(
      radius=cfg.sphere_radius,
      mass=cfg.sphere_mass,
      friction=cfg.friction,
      max_travel=cfg.max_travel,
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, -2.0),
      joint_pos={"rail_joint": 0.0},
      joint_vel={"rail_joint": 0.0},
    ),
  )


def _heading_local_xy(vec_w: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
  """Rotate world xy vectors into a heading-aligned frame."""
  c = torch.cos(heading)
  s = torch.sin(heading)
  x = c * vec_w[:, 0] + s * vec_w[:, 1]
  y = -s * vec_w[:, 0] + c * vec_w[:, 1]
  return torch.stack((x, y), dim=-1)


def mapped_force_drop(
  previous_force_ratio: torch.Tensor,
  current_force_ratio: torch.Tensor,
  drive_mask: torch.Tensor,
  scale: float = 0.005,
) -> torch.Tensor:
  """Map signed force decrease to [0, 1] before SMP multiplication."""
  signed = torch.tanh((previous_force_ratio - current_force_ratio) / scale)
  return drive_mask * 0.5 * (1.0 + signed)


class RailPusherController:
  """Class-based step event implementing WAIT→DRIVE→RETRACT→DONE.

  The term also owns the task's contact filter and cached quantities. Other
  manager terms call :meth:`refresh_contact`; it is idempotent per policy step.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.cfg: RailPusherCfg = cfg.params["rail_cfg"]
    self.device = torch.device(env.device)
    self.num_envs = env.num_envs
    self.robot = env.scene["robot"]
    self.pusher = env.scene["rail_pusher"]
    self.contact_sensor = env.scene["rail_pusher_contact"]

    target_names = (
      "torso_collision",
      "left_shoulder_yaw_collision",
      "right_shoulder_yaw_collision",
    )
    self.target_geom_ids = torch.tensor(
      self.robot.find_geoms(target_names, preserve_order=True)[0],
      device=self.device,
      dtype=torch.long,
    )
    if len(self.cfg.target_probabilities) == 4:
      self.target_geom_ids = torch.cat((self.target_geom_ids, self.target_geom_ids[:1]))
    global_geom_ids = self.robot.indexing.geom_ids[self.target_geom_ids].cpu().numpy()
    self.target_sizes = torch.as_tensor(
      env.sim.mj_model.geom_size[global_geom_ids].copy(),
      device=self.device,
      dtype=torch.float32,
    )
    if self.cfg.start_surface_gap is not None:
      types = env.sim.mj_model.geom_type[global_geom_ids]
      if not all(
        int(t)
        in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE))
        for t in types
      ):
        raise ValueError("Surface gap placement supports capsule/sphere targets only")
      for i, geom_type in enumerate(types):
        if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
          self.target_sizes[i, 1] = 0.0
    self.foot_geom_ids = torch.tensor(
      self.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")[0],
      device=self.device,
      dtype=torch.long,
    )

    body_ids = self.robot.indexing.body_ids.cpu().numpy()
    self.robot_mass = float(env.sim.mj_model.body_mass[body_ids].sum())
    self.weight = self.robot_mass * 9.81
    self.max_drive_force = self.cfg.max_force_weight_ratio * self.weight
    self.force_on = self.cfg.force_on_ratio * self.weight
    self.force_done = self.cfg.force_done_ratio * self.weight
    self.force_rate_done = self.cfg.force_rate_done_ratio_s * self.weight
    self.force_emergency = self.cfg.force_emergency_ratio * self.weight

    n = self.num_envs
    self.phase = torch.full(
      (n,), int(RailPhase.WAIT), device=self.device, dtype=torch.long
    )
    self.target = torch.zeros(n, device=self.device, dtype=torch.long)
    self.wait_left = torch.zeros(n, device=self.device)
    self.drive_elapsed = torch.zeros(n, device=self.device)
    self.reference_pos = torch.zeros(n, device=self.device)
    self.reference_vel = torch.zeros(n, device=self.device)
    self.filtered_force = torch.zeros(n, device=self.device)
    self.previous_force = torch.zeros(n, device=self.device)
    self.force_rate = torch.zeros(n, device=self.device)
    self.raw_force_peak = torch.zeros(n, device=self.device)
    self.event_peak_force = torch.zeros(n, device=self.device)
    self.force_impulse = torch.zeros(n, device=self.device)
    self.contact_count = torch.zeros(n, device=self.device, dtype=torch.long)
    self.contact_latched = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.stable_count = torch.zeros(n, device=self.device, dtype=torch.long)
    self.emergency_count = torch.zeros(n, device=self.device, dtype=torch.long)
    self.emergency = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.event_success = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.success_count = torch.zeros(n, device=self.device)
    self.unload_time = torch.zeros(n, device=self.device)
    self.drive_mask = torch.zeros(n, device=self.device)
    self.force_drop = torch.zeros(n, device=self.device)
    self.contact_obs = torch.zeros((n, 8), device=self.device)
    self.last_direction_local = torch.zeros((n, 3), device=self.device)
    self.last_position_local = torch.zeros((n, 3), device=self.device)
    self.initial_root_pos = torch.zeros((n, 3), device=self.device)
    self.event_root_pos = torch.zeros((n, 3), device=self.device)
    self.initial_heading = torch.zeros(n, device=self.device)
    self.max_abs_displacement = torch.zeros((n, 2), device=self.device)
    self.rail_cross_axis_error = torch.zeros(n, device=self.device)
    self.failure_nonfoot_ground = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.failure_height = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.failure_tilt = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.failure_workspace = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.travel_limit_reached = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.failure_joint = torch.zeros(n, device=self.device, dtype=torch.bool)
    self.failure_nan = torch.zeros(n, device=self.device, dtype=torch.bool)
    self._last_contact_step = -1

    if self.cfg.conditional:
      initialize_conditions(self)
    env._rail_controller = self  # type: ignore[attr-defined]

  def _ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
      return torch.arange(self.num_envs, device=self.device)
    return env_ids

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = self._ids(env_ids)
    if ids.numel() == 0:
      return
    if self.cfg.conditional:
      reset_conditions(self, ids)
    lo, hi = self.cfg.wait_range_s
    self.phase[ids] = int(RailPhase.WAIT)
    self.wait_left[ids] = lo + torch.rand(len(ids), device=self.device) * (hi - lo)
    for tensor in (
      self.drive_elapsed,
      self.reference_pos,
      self.reference_vel,
      self.filtered_force,
      self.previous_force,
      self.force_rate,
      self.raw_force_peak,
      self.event_peak_force,
      self.force_impulse,
      self.success_count,
      self.unload_time,
      self.drive_mask,
      self.force_drop,
      self.contact_count,
      self.stable_count,
      self.emergency_count,
    ):
      tensor[ids] = 0
    for tensor in (
      self.contact_latched,
      self.emergency,
      self.event_success,
      self.failure_nonfoot_ground,
      self.failure_height,
      self.failure_tilt,
      self.failure_workspace,
      self.travel_limit_reached,
      self.failure_joint,
      self.failure_nan,
    ):
      tensor[ids] = False
    self.contact_obs[ids] = 0.0
    self.last_direction_local[ids] = 0.0
    self.last_position_local[ids] = 0.0
    self.max_abs_displacement[ids] = 0.0
    self.rail_cross_axis_error[ids] = 0.0

    q_adr = self.robot.indexing.free_joint_q_adr
    self.initial_root_pos[ids] = self.robot.data.data.qpos[ids[:, None], q_adr[:3]]
    self.event_root_pos[ids] = self.initial_root_pos[ids]
    initial_quat = self.robot.data.data.qpos[ids[:, None], q_adr[3:7]]
    initial_yaw_quat = yaw_quat(initial_quat)
    self.initial_heading[ids] = 2.0 * torch.atan2(
      initial_yaw_quat[:, 3], initial_yaw_quat[:, 0]
    )
    self._park(ids)

  def _park(self, ids: torch.Tensor) -> None:
    pose = torch.zeros((len(ids), 7), device=self.device)
    pose[:, :3] = self.env.scene.env_origins[ids]
    pose[:, 2] = -2.0
    pose[:, 3] = 1.0
    self.pusher.write_mocap_pose_to_sim(pose, env_ids=ids)
    zero = torch.zeros((len(ids), 1), device=self.device)
    self.pusher.write_joint_state_to_sim(zero, zero, env_ids=ids)
    dof = int(self.pusher.indexing.joint_v_adr[0])
    self.env.sim.data.qfrc_applied[ids, dof] = 0.0

  def _start_event(self, ids: torch.Tensor) -> None:
    if ids.numel() == 0:
      return
    probs = torch.tensor(self.cfg.target_probabilities, device=self.device)
    probs = probs / probs.sum()
    targets = torch.multinomial(probs, len(ids), replacement=True)
    self.target[ids] = targets

    root_heading = self.robot.data.heading_w[ids]
    canonical = canonical_drive_direction(targets, self.cfg.push_direction_mode)
    c = torch.cos(root_heading)
    s = torch.sin(root_heading)
    drive_dir = torch.stack(
      (
        c * canonical[:, 0] - s * canonical[:, 1],
        s * canonical[:, 0] + c * canonical[:, 1],
      ),
      dim=-1,
    )
    target_pos = self.robot.data.geom_pose_w[ids, self.target_geom_ids[targets], :3]
    base_pos = target_pos.clone()
    if self.cfg.start_surface_gap is None:
      distance = self.cfg.start_distance
    else:
      target_quat = self.robot.data.geom_pose_w[ids, self.target_geom_ids[targets], 3:]
      local_axis = torch.zeros((len(ids), 3), device=self.device)
      local_axis[:, 2] = 1.0
      axis = quat_apply(target_quat, local_axis)
      direction3 = torch.cat(
        (drive_dir, torch.zeros((len(ids), 1), device=self.device)), -1
      )
      sizes = self.target_sizes[targets]
      # Aim at the upper shoulder capsule endpoint, not its lower upper-arm
      # center, which is occluded by forearms in the default bent-elbow pose.
      shoulder = ((targets == 1) | (targets == 2)) & self.cfg.conditional
      offset = axis * (sizes[:, 1] * torch.sign(axis[:, 2]) * shoulder)[:, None]
      base_pos += offset
      distance = capsule_start_distance(
        direction3,
        axis,
        sizes[:, 1],
        sizes[:, 0],
        self.cfg.sphere_radius,
        self.cfg.start_surface_gap,
        offset=offset,
      )[:, None]
    base_pos[:, :2] -= distance * drive_dir

    rail_yaw = torch.atan2(drive_dir[:, 1], drive_dir[:, 0])
    pose = torch.zeros((len(ids), 7), device=self.device)
    pose[:, :3] = base_pos
    pose[:, 3] = torch.cos(0.5 * rail_yaw)
    pose[:, 6] = torch.sin(0.5 * rail_yaw)
    self.pusher.write_mocap_pose_to_sim(pose, env_ids=ids)
    zero = torch.zeros((len(ids), 1), device=self.device)
    self.pusher.write_joint_state_to_sim(zero, zero, env_ids=ids)

    self.phase[ids] = int(RailPhase.DRIVE)
    self.drive_elapsed[ids] = 0.0
    self.reference_pos[ids] = 0.0
    self.reference_vel[ids] = 0.0
    self.filtered_force[ids] = 0.0
    self.previous_force[ids] = 0.0
    self.force_rate[ids] = 0.0
    self.event_peak_force[ids] = 0.0
    self.force_impulse[ids] = 0.0
    self.contact_count[ids] = 0
    self.contact_latched[ids] = False
    self.stable_count[ids] = 0
    self.emergency_count[ids] = 0
    self.emergency[ids] = False
    self.event_success[ids] = False
    self.contact_obs[ids] = 0.0
    self.event_root_pos[ids] = self.robot.data.root_link_pos_w[ids]
    self.initial_heading[ids] = self.robot.data.heading_w[ids]
    self.max_abs_displacement[ids] = 0.0

  def refresh_contact(self) -> None:
    step = int(self.env.common_step_counter)
    if self._last_contact_step == step:
      return
    self._last_contact_step = step
    data = self.contact_sensor.data
    assert data.found is not None
    assert data.force is not None
    assert data.pos is not None
    assert data.normal is not None

    found = data.found > 0
    normal_force = data.force[..., 0].abs() * found
    force_sum = normal_force.sum(dim=1)
    weighted_vec = (normal_force[..., None] * data.normal).sum(dim=1)
    vec_norm = torch.linalg.vector_norm(weighted_vec, dim=-1)
    valid_current = vec_norm > 1e-8
    direction_w = weighted_vec / vec_norm.clamp_min(1e-8)[:, None]
    position_w = (normal_force[..., None] * data.pos).sum(dim=1) / force_sum.clamp_min(
      1e-8
    )[:, None]

    if data.force_history is not None:
      substep_force = data.force_history[..., 0].abs().sum(dim=1)
      control_force = substep_force.mean(dim=-1)
      self.raw_force_peak = substep_force.max(dim=-1).values
      count = torch.zeros_like(self.emergency_count)
      peak_count = torch.zeros_like(count)
      for h in reversed(range(substep_force.shape[1])):
        over = substep_force[:, h] > self.force_emergency
        count = torch.where(over, count + 1, torch.zeros_like(count))
        peak_count = torch.maximum(peak_count, count)
      self.emergency_count = count
      self.emergency = peak_count >= self.cfg.emergency_substeps
    else:
      control_force = force_sum
      self.raw_force_peak = force_sum
      self.emergency = force_sum > self.force_emergency

    old_force = self.filtered_force.clone()
    self.filtered_force = (
      self.cfg.ema_alpha * self.filtered_force
      + (1.0 - self.cfg.ema_alpha) * control_force
    )
    self.force_rate = (self.filtered_force - old_force) / self.env.step_dt
    self.previous_force = old_force
    self.event_peak_force = torch.maximum(self.event_peak_force, self.raw_force_peak)
    self.force_impulse += self.filtered_force * self.env.step_dt

    heading_quat = yaw_quat(self.robot.data.root_link_quat_w)
    direction_local = quat_apply_inverse(heading_quat, direction_w)
    relative_pos_w = position_w - self.robot.data.root_link_pos_w
    position_local = quat_apply_inverse(heading_quat, relative_pos_w)
    self.last_direction_local = torch.where(
      valid_current[:, None], direction_local, self.last_direction_local
    )
    self.last_position_local = torch.where(
      valid_current[:, None], position_local, self.last_position_local
    )

    active_contact = self.filtered_force > self.force_on
    self.contact_obs.zero_()
    self.contact_obs[:, 0] = active_contact.float()
    self.contact_obs[:, 1] = (
      torch.clamp(self.filtered_force / self.weight, 0.0, self.cfg.force_obs_clip)
      * active_contact
    )
    self.contact_obs[:, 2:5] = self.last_direction_local * active_contact[:, None]
    self.contact_obs[:, 5:8] = self.last_position_local * active_contact[:, None]

    if self.cfg.conditional:
      update_conditions(self)

    driving = self.phase == int(RailPhase.DRIVE)
    self.drive_mask = driving.float()
    force_drop = mapped_force_drop(
      old_force / self.weight,
      self.filtered_force / self.weight,
      self.drive_mask,
    )
    # Before the event, unchanged zero force is the neutral force-difference
    # case. Keeping it at 0.5 preserves a dense SMP-gated learning signal
    # during the randomized wait; RETRACT and DONE remain masked to zero.
    waiting = self.phase == int(RailPhase.WAIT)
    self.force_drop = torch.where(waiting, torch.full_like(force_drop, 0.5), force_drop)

    self.contact_count = torch.where(
      driving & active_contact,
      self.contact_count + 1,
      torch.where(driving, torch.zeros_like(self.contact_count), self.contact_count),
    )
    self.contact_latched |= self.contact_count >= self.cfg.contact_latch_steps

    eligible = (
      self.contact_latched
      & (self.event_peak_force >= self.force_done)
      & (
        (self.drive_elapsed >= self.cfg.min_drive_time_s)
        | (self.reference_pos >= self.cfg.min_reference_travel)
      )
    )
    stable = (
      driving
      & eligible
      & (self.filtered_force < self.force_done)
      & (self.force_rate.abs() < self.force_rate_done)
    )
    self.stable_count = torch.where(
      stable, self.stable_count + 1, torch.zeros_like(self.stable_count)
    )
    stable_steps = math.ceil(self.cfg.stable_time_s / self.env.step_dt)
    success = stable & (self.stable_count >= stable_steps)
    self.event_success = success
    self.success_count += success.float()
    self.unload_time = torch.where(success, self.drive_elapsed, self.unload_time)
    self.phase = torch.where(
      success,
      torch.full_like(self.phase, int(RailPhase.RETRACT)),
      self.phase,
    )

    displacement_w = self.robot.data.root_link_pos_w[:, :2] - self.event_root_pos[:, :2]
    displacement_h = _heading_local_xy(displacement_w, self.initial_heading)
    self.max_abs_displacement = torch.maximum(
      self.max_abs_displacement, displacement_h.abs()
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    rail_cfg: RailPusherCfg | None = None,
  ) -> None:
    del env_ids, rail_cfg
    self.refresh_contact()
    dt = env.step_dt
    was_active = (self.phase == int(RailPhase.DRIVE)) | (
      self.phase == int(RailPhase.RETRACT)
    )

    waiting = self.phase == int(RailPhase.WAIT)
    self.wait_left = torch.where(waiting, self.wait_left - dt, self.wait_left)
    start_ids = (waiting & (self.wait_left <= 0.0)).nonzero().flatten()
    self._start_event(start_ids)

    drive = self.phase == int(RailPhase.DRIVE)
    retract = self.phase == int(RailPhase.RETRACT)
    self.drive_elapsed += drive.float() * dt
    ramp = torch.clamp(self.drive_elapsed / self.cfg.ramp_time, 0.0, 1.0)
    drive_vel = self.cfg.reference_speed * ramp
    self.reference_vel = torch.where(
      drive,
      drive_vel,
      torch.where(retract, torch.full_like(drive_vel, -self.cfg.retract_speed), 0.0),
    )
    self.reference_pos = torch.where(
      drive,
      self.reference_pos + self.reference_vel * dt,
      torch.where(
        retract,
        torch.clamp(self.reference_pos + self.reference_vel * dt, min=0.0),
        self.reference_pos,
      ),
    )

    u = self.pusher.data.joint_pos[:, 0]
    u_dot = self.pusher.data.joint_vel[:, 0]
    effort = self.cfg.stiffness * (self.reference_pos - u) + self.cfg.damping * (
      self.reference_vel - u_dot
    )
    effort = torch.clamp(effort, -self.max_drive_force, self.max_drive_force)
    effort = torch.where(drive | retract, effort, torch.zeros_like(effort))
    dof = int(self.pusher.indexing.joint_v_adr[0])
    env.sim.data.qfrc_applied[:, dof] = effort

    mocap_id = self.pusher.indexing.mocap_id
    assert mocap_id is not None
    base_pos = self.pusher.data.data.mocap_pos[:, mocap_id]
    base_quat = self.pusher.data.data.mocap_quat[:, mocap_id]
    sphere_pos = self.pusher.data.geom_pose_w[:, 0, :3]
    rail_local = quat_apply_inverse(base_quat, sphere_pos - base_pos)
    cross_axis = torch.linalg.vector_norm(rail_local[:, 1:], dim=-1)
    self.rail_cross_axis_error = torch.where(
      was_active, cross_axis, torch.zeros_like(cross_axis)
    )

    finished = retract & (u < 0.005) & (u_dot.abs() < 0.02)
    finished_ids = finished.nonzero().flatten()
    if finished_ids.numel() > 0:
      self.phase[finished_ids] = int(RailPhase.DONE)
      self._park(finished_ids)

    # Reaching the rail's physical limit is telemetry, not an episode failure.
    # The slide-joint range still prevents the pusher from leaving its workspace.
    self.travel_limit_reached = u >= self.cfg.max_travel - 1e-4


def get_controller(env: ManagerBasedRlEnv) -> RailPusherController:
  controller = getattr(env, "_rail_controller", None)
  if controller is None:
    raise RuntimeError("RailPusherController has not been initialized")
  return controller
