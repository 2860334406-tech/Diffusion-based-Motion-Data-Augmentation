"""Measured contact conditioning, hysteresis, transition windows, and geometry."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from smp.pretrain.conditioning import NULL_CONDITION


class ContactConditionState:
  def __init__(
    self,
    count,
    device,
    window_size=10,
    on=0.0075,
    off=0.005,
    dwell=3,
    transition_mode="null_until_stable",
  ):
    self.on, self.off, self.dwell = on, off, dwell
    self.transition_mode = transition_mode
    self.ids = torch.zeros(count, device=device, dtype=torch.long)
    self.pending = self.ids.clone()
    self.count = self.ids.clone()
    self.release = self.ids.clone()
    self.history = torch.zeros(count, window_size, device=device, dtype=torch.long)
    self.reward_ids = self.ids.clone()
    self.reward_mask = torch.ones(count, window_size, device=device, dtype=torch.bool)

  def reset(self, env_ids):
    for value in (
      self.ids,
      self.pending,
      self.count,
      self.release,
      self.history,
      self.reward_ids,
    ):
      value[env_ids] = 0
    self.reward_mask[env_ids] = True

  def update(self, ratios):
    # Columns: left, right, chest, back. IDs are independent of PushTarget.
    force, column = ratios.max(-1)
    candidate = column + 1
    strong = force > self.on
    same_pending = candidate == self.pending
    self.count = torch.where(strong, torch.where(same_pending, self.count + 1, 1), 0)
    self.pending = torch.where(strong, candidate, self.pending)
    self.release = torch.where(force < self.off, self.release + 1, 0)
    self.ids = torch.where(self.count >= self.dwell, self.pending, self.ids)
    self.ids = torch.where(self.release >= self.dwell, 0, self.ids)
    self.history = torch.roll(self.history, shifts=-1, dims=1)
    self.history[:, -1] = self.ids
    stable = (self.history == self.ids[:, None]).all(-1)
    if self.transition_mode == "masked_current":
      self.reward_ids.copy_(self.ids)
      self.reward_mask.copy_(self.history == self.ids[:, None])
    else:
      self.reward_ids = torch.where(stable, self.ids, NULL_CONDITION)
      self.reward_mask.fill_(True)
    return self.ids


def capsule_start_distance(
  direction, axis, half_length, radius, ball_radius, gap, offset=None
):
  """Exact center-line placement at gap from a capsule (sphere is half_length=0)."""
  lower = torch.zeros_like(radius)
  upper = radius + ball_radius + half_length + gap
  distance = radius + ball_radius + gap
  for _ in range(24):
    middle = (lower + upper) * 0.5
    point = -middle[:, None] * direction
    if offset is not None:
      point = point + offset
    projection = (point * axis).sum(-1).clamp(-half_length, half_length)
    separation = torch.linalg.vector_norm(point - projection[:, None] * axis, dim=-1)
    lower = torch.where(separation < distance, middle, lower)
    upper = torch.where(separation >= distance, middle, upper)
  return (lower + upper) * 0.5


def initialize_conditions(controller):
  controller.condition_state = ContactConditionState(
    controller.num_envs,
    controller.device,
    transition_mode=controller.cfg.condition_transition_mode,
  )
  controller.region_sensor = controller.env.scene["rail_contact_regions"]
  names = controller.region_sensor.primary_names
  controller.region_names = [name.split("/")[-1] for name in names]
  controller.region_ratios = torch.zeros(
    (controller.num_envs, 4), device=controller.device
  )
  controller.conditional_metrics = {
    name: torch.zeros(controller.num_envs, device=controller.device)
    for name in (
      "contact_coverage",
      "response_direction_correct",
      "waist_rotation_rad",
      "pelvis_rotation_rad",
      "signed_waist_response_peak_rad",
      "pelvis_yaw_abs_peak_rad",
      "response_x_displacement_m",
      "recovered_static",
      "unclassified_contact",
    )
  }
  controller.response_origin = torch.zeros(
    (controller.num_envs, 5), device=controller.device
  )
  controller.response_condition = torch.zeros(
    controller.num_envs, device=controller.device, dtype=torch.long
  )
  controller.response_steps = torch.zeros(
    controller.num_envs, device=controller.device, dtype=torch.long
  )
  controller.waist_joint_id = controller.robot.find_joints("waist_yaw_joint")[0][0]
  controller.torso_body_id = controller.robot.find_bodies("torso_link")[0][0]


def reset_conditions(controller, ids):
  controller.condition_state.reset(ids)
  controller.region_ratios[ids] = 0
  controller.response_origin[ids] = 0
  controller.response_condition[ids] = 0
  controller.response_steps[ids] = 0
  for value in controller.conditional_metrics.values():
    value[ids] = 0


def update_conditions(controller):
  sensor = controller.region_sensor
  data = sensor.data
  n, slots = controller.num_envs, sensor.cfg.num_slots
  force = (data.force[..., 0].abs() * (data.found > 0)).reshape(n, -1, slots)
  if data.force_history is not None:
    force = data.force_history[..., 0].abs().mean(-1).reshape(n, -1, slots)
  positions = data.pos.reshape(n, -1, slots, 3)
  found = (data.found > 0).reshape(n, -1, slots)
  ratios = torch.zeros_like(controller.region_ratios)
  for group, name in enumerate(controller.region_names):
    if name == "left_shoulder_yaw_collision":
      ratios[:, 0] = force[:, group].sum(-1)
    elif name == "right_shoulder_yaw_collision":
      ratios[:, 1] = force[:, group].sum(-1)
    elif name == "torso_collision":
      geom = controller.robot.data.geom_pose_w[:, controller.target_geom_ids[0]]
      # A contact can resolve before the last physics substep. Its force is
      # retained in history; use the measured ball position for surface side
      # only in that case, never the scheduled target.
      measured_pos = torch.where(
        found[:, group, :, None],
        positions[:, group],
        controller.pusher.data.geom_pose_w[:, :1, :3],
      )
      relative = measured_pos - geom[:, None, :3]
      local = quat_apply_inverse(
        controller.robot.data.body_link_quat_w[:, controller.torso_body_id, None, :]
        .expand(-1, slots, -1)
        .reshape(-1, 4),
        relative.reshape(-1, 3),
      ).reshape(n, slots, 3)
      front = local[..., 0] >= 0
      ratios[:, 2] = (force[:, group] * front).sum(-1)
      ratios[:, 3] = (force[:, group] * ~front).sum(-1)
  controller.region_ratios.mul_(0.8).add_(ratios / controller.weight, alpha=0.2)
  ids = controller.condition_state.update(controller.region_ratios)
  robot = controller.robot
  pos = robot.data.root_link_pos_w
  heading = robot.data.heading_w
  waist = robot.data.joint_pos[:, controller.waist_joint_id]
  first = (ids > 0) & (controller.response_condition == 0)
  controller.response_origin[first, :3] = pos[first]
  controller.response_origin[first, 3] = heading[first]
  controller.response_origin[first, 4] = waist[first]
  controller.response_condition[first] = ids[first]
  active = controller.response_condition > 0
  controller.response_steps += active.long()
  elapsed = controller.response_steps * controller.env.step_dt
  initial = controller.response_origin
  delta = pos - initial[:, :3]
  dx = torch.cos(initial[:, 3]) * delta[:, 0] + torch.sin(initial[:, 3]) * delta[:, 1]
  dyaw = torch.atan2(
    torch.sin(heading - initial[:, 3]), torch.cos(heading - initial[:, 3])
  )
  d_waist = waist - initial[:, 4]
  condition = controller.response_condition
  correct = (
    (
      ((condition == 1) & (d_waist >= 0.05) & (dyaw.abs() <= 0.175))
      | ((condition == 2) & (d_waist <= -0.05) & (dyaw.abs() <= 0.175))
      | ((condition == 3) & (dx <= -0.03))
      | ((condition == 4) & (dx >= 0.03))
    )
    & active
    & (elapsed <= 2.0)
  )
  metrics = controller.conditional_metrics
  metrics["contact_coverage"] = active.float()
  metrics["response_direction_correct"] = torch.maximum(
    metrics["response_direction_correct"], correct.float()
  )
  metrics["waist_rotation_rad"] = torch.where(active, d_waist, 0.0)
  metrics["pelvis_rotation_rad"] = torch.where(active, dyaw, 0.0)
  response_window = active & (elapsed <= 2.0)
  signed_waist = torch.where(
    condition == 1,
    d_waist,
    torch.where(condition == 2, -d_waist, torch.zeros_like(d_waist)),
  )
  metrics["signed_waist_response_peak_rad"] = torch.maximum(
    metrics["signed_waist_response_peak_rad"],
    torch.where(response_window, signed_waist, torch.zeros_like(signed_waist)),
  )
  metrics["pelvis_yaw_abs_peak_rad"] = torch.maximum(
    metrics["pelvis_yaw_abs_peak_rad"],
    torch.where(response_window, dyaw.abs(), torch.zeros_like(dyaw)),
  )
  metrics["response_x_displacement_m"] = torch.where(active, dx, 0.0)
  speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
  metrics["recovered_static"] = (
    active & (ids == 0) & (elapsed > 1.0) & (speed <= 0.15)
  ).float()
  metrics["unclassified_contact"] = (
    (controller.filtered_force > controller.force_on)
    & (controller.region_ratios.max(-1).values < 0.005)
  ).float()


def conditional_metric(env, name):
  controller = env._rail_controller
  controller.refresh_contact()
  return controller.conditional_metrics[name]
