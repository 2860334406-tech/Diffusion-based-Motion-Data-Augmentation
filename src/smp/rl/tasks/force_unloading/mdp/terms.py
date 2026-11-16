"""Observations, rewards, terminations, and metrics for force unloading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs import mdp as base_mdp
from mjlab.utils.noise import NoiseCfg

from smp.rl.tasks.force_unloading.mdp.controller import get_controller

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RailContactNoiseCfg(NoiseCfg):
  """Contact-conditional noise that preserves the exact no-contact zero."""

  force_relative: float = 0.02
  direction_std_rad: float = math.radians(0.5)
  position_std_m: float = 0.003

  def apply(self, data: torch.Tensor) -> torch.Tensor:
    out = data.clone()
    active = out[:, 0:1] > 0.5
    force_scale = 1.0 + (2.0 * torch.rand_like(out[:, 1:2]) - 1.0) * self.force_relative
    out[:, 1:2] *= force_scale
    direction = out[:, 2:5] + torch.randn_like(out[:, 2:5]) * self.direction_std_rad
    direction = direction / torch.linalg.vector_norm(
      direction, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    out[:, 2:5] = direction
    out[:, 5:8] += torch.randn_like(out[:, 5:8]) * self.position_std_m
    out[:, 1:8] *= active
    return out


def rail_contact_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  return controller.contact_obs


def force_unloading_task_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  root_vel_xy = controller.robot.data.root_link_lin_vel_w[:, :2]
  stationary = torch.exp(
    -controller.cfg.stationary_speed_scale * torch.sum(root_vel_xy.square(), dim=-1)
  )
  stationary_weight = controller.cfg.stationary_reward_weight
  force_drop_weight = 1.0 - stationary_weight
  task_reward = (
    force_drop_weight * controller.force_drop + stationary_weight * stationary
  )
  return torch.where(controller.emergency, torch.zeros_like(task_reward), task_reward)


def emergency_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  return controller.emergency


def travel_limit_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Report rail saturation without terminating or labelling the episode failed."""
  controller = get_controller(env)
  controller.travel_limit_reached = (
    controller.pusher.data.joint_pos[:, 0] >= controller.cfg.max_travel - 1e-4
  )
  return controller.travel_limit_reached.float()


def nonfoot_ground_contact(
  env: ManagerBasedRlEnv, sensor_name: str = "nonfoot_ground_contact"
) -> torch.Tensor:
  controller = get_controller(env)
  data = env.scene[sensor_name].data
  assert data.found is not None
  current = data.found.bool().any(dim=-1)
  if data.force_history is not None:
    force = torch.linalg.vector_norm(data.force_history, dim=-1)
    current |= (force > 1.0).any(dim=-1).any(dim=-1)
  controller.failure_nonfoot_ground = current
  return controller.failure_nonfoot_ground


def root_height_failure(env: ManagerBasedRlEnv, fraction: float = 0.60) -> torch.Tensor:
  controller = get_controller(env)
  controller.failure_height = (
    controller.robot.data.root_link_pos_w[:, 2]
    < fraction * controller.initial_root_pos[:, 2]
  )
  return controller.failure_height


def torso_tilt_failure(
  env: ManagerBasedRlEnv, limit_angle: float = math.radians(60.0)
) -> torch.Tensor:
  controller = get_controller(env)
  gravity = controller.robot.data.projected_gravity_b
  angle = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)).abs()
  controller.failure_tilt = angle > limit_angle
  return controller.failure_tilt


def workspace_failure(
  env: ManagerBasedRlEnv, x_limit: float = 0.60, y_limit: float = 0.40
) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  displacement_w = (
    controller.robot.data.root_link_pos_w[:, :2] - controller.event_root_pos[:, :2]
  )
  heading = controller.initial_heading
  c = torch.cos(heading)
  s = torch.sin(heading)
  x = c * displacement_w[:, 0] + s * displacement_w[:, 1]
  y = -s * displacement_w[:, 0] + c * displacement_w[:, 1]
  event_started = controller.phase != 0
  controller.failure_workspace = event_started & (
    (x.abs() > x_limit) | (y.abs() > y_limit)
  )
  return controller.failure_workspace


def joint_hard_limit_failure(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  pos = controller.robot.data.joint_pos
  limits = controller.robot.data.joint_pos_limits
  controller.failure_joint = ((pos < limits[..., 0]) | (pos > limits[..., 1])).any(
    dim=-1
  )
  return controller.failure_joint


def nan_failure(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.failure_nan = base_mdp.nan_detection(env)
  return controller.failure_nan


def peak_force_ratio(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  return controller.raw_force_peak / controller.weight


def normalized_force_impulse(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  return controller.filtered_force / controller.weight * env.step_dt


def unload_time(env: ManagerBasedRlEnv) -> torch.Tensor:
  return get_controller(env).unload_time


def event_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  return get_controller(env).event_success.float()


def max_x_displacement(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  return controller.max_abs_displacement[:, 0]


def max_y_displacement(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  return controller.max_abs_displacement[:, 1]


def precontact_root_planar_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Root planar speed while the controller is in its randomized wait phase."""
  controller = get_controller(env)
  controller.refresh_contact()
  speed = torch.linalg.vector_norm(
    controller.robot.data.root_link_lin_vel_w[:, :2], dim=-1
  )
  return torch.where(controller.phase == 0, speed, torch.zeros_like(speed))


def precontact_root_displacement(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Planar displacement from reset while waiting for the pusher event."""
  controller = get_controller(env)
  controller.refresh_contact()
  delta = (
    controller.robot.data.root_link_pos_w[:, :2] - controller.initial_root_pos[:, :2]
  )
  distance = torch.linalg.vector_norm(delta, dim=-1)
  return torch.where(controller.phase == 0, distance, torch.zeros_like(distance))


def capture_point_overflow(
  env: ManagerBasedRlEnv, sensor_name: str = "foot_ground_contact"
) -> torch.Tensor:
  """Capture-point distance outside an axis-aligned support-polygon proxy."""
  controller = get_controller(env)
  data = env.scene[sensor_name].data
  assert data.found is not None
  contact = data.found.bool()
  feet_xy = controller.robot.data.geom_pose_w[:, controller.foot_geom_ids, :2]
  inf = torch.full_like(feet_xy, float("inf"))
  neg_inf = torch.full_like(feet_xy, float("-inf"))
  low = torch.where(contact[..., None], feet_xy, inf).amin(dim=1) - 0.03
  high = torch.where(contact[..., None], feet_xy, neg_inf).amax(dim=1) + 0.03
  root = controller.robot.data.root_link_pos_w
  omega = torch.sqrt(9.81 / root[:, 2].clamp_min(0.2))
  cp = root[:, :2] + controller.robot.data.root_link_lin_vel_w[:, :2] / omega[:, None]
  overflow = torch.maximum(low - cp, cp - high).clamp_min(0.0)
  valid = contact.any(dim=-1)
  distance = torch.linalg.vector_norm(overflow, dim=-1)
  return torch.where(valid, distance, torch.zeros_like(distance))


def max_contact_foot_slip(
  env: ManagerBasedRlEnv, sensor_name: str = "foot_ground_contact"
) -> torch.Tensor:
  controller = get_controller(env)
  data = env.scene[sensor_name].data
  assert data.found is not None
  contact = data.found.bool()
  speed = torch.linalg.vector_norm(
    controller.robot.data.geom_vel_w[:, controller.foot_geom_ids, :2], dim=-1
  )
  return torch.where(contact, speed, torch.zeros_like(speed)).amax(dim=-1)


def failure_metric(env: ManagerBasedRlEnv, reason: str) -> torch.Tensor:
  controller = get_controller(env)
  return getattr(controller, f"failure_{reason}").float()


def emergency_failure_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  controller = get_controller(env)
  controller.refresh_contact()
  return controller.emergency.float()


def smp_reward_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
  return getattr(env, "_smp_reward", torch.zeros(env.num_envs, device=env.device))


def smp_reward_variance(env: ManagerBasedRlEnv) -> torch.Tensor:
  reward = smp_reward_mean(env)
  return (reward - reward.mean()).square()


def smp_component(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  """Expose a named V3 score component without changing legacy tasks."""
  return getattr(env, f"_smp_{name}", torch.zeros(env.num_envs, device=env.device))
