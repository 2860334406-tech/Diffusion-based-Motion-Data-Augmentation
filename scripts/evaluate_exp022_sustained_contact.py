"""EXP-022 closed-loop oracle/estimate test with a physical rail sphere.

The frozen actor receives either its original contact history or the streamed
estimator history. The pusher can use the original EXP-022 push/retract rule or
the sustained-pressure rule. All four physics-substep collision geometries and
control-step forces are recorded for auditing.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

import evaluate_force_unloading as evaluation
from smp.force_estimation import load_exp022_chest_estimator
from smp.force_estimation.exp022_interface import (
  Exp022ContactAdapter,
  add_force_estimator_observations,
  validate_force_estimator_input,
)
from smp.force_estimation.exp022_shoulder_rescue import Exp022ShoulderRescueEstimator
from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)
from smp.rl.tasks.force_unloading.mdp.controller import (
  RailPhase,
  RailPusherCfg,
  RailPusherController,
)


def own_gpu_mib() -> int | None:
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-compute-apps=pid,used_gpu_memory",
      "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
    check=False,
  )
  if result.returncode:
    return None
  for line in result.stdout.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) >= 2 and parts[0] == str(os.getpid()):
      return int(parts[1])
  return None


class ForceRegulatedRail(RailPusherController):
  """Low-speed rail approach, then force feedback on a tracked fixture base."""

  target_force_n = 30.0
  force_gain = 0.5
  speed_gain = 500.0
  target_speed_m_s = 0.12
  recovery_speed_m_s = 0.30
  follow_speed_m_s = 0.50
  follow_yaw_rad_s = 2.0
  base_force_gain_m_s_per_n = 0.01
  motor_hold_after_contact = False
  freeze_push_heading = False
  one_way_speed_limit = False

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.commanded_effort_n = torch.zeros(self.num_envs, device=self.device)
    self.measured_normal_n = torch.zeros(self.num_envs, device=self.device)
    self.seen_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self.seen_contact[self._ids(env_ids)] = False

  def __call__(self, env, env_ids=None, rail_cfg=None):
    super().__call__(env, env_ids=env_ids, rail_cfg=rail_cfg)
    history = self.contact_sensor.data.force_history
    if history is None or history.shape[-2:] != (env.cfg.decimation, 3):
      raise RuntimeError("The force regulator requires four 5-ms contact samples")
    normal = history[..., 0].abs().sum(dim=1).mean(dim=-1)
    self.measured_normal_n = normal
    self.seen_contact |= normal >= 1.0
    driving = self.phase == int(RailPhase.DRIVE)
    if bool(driving.any()):
      ids = driving.nonzero().flatten()
      heading = (
        self.initial_heading[ids]
        if self.freeze_push_heading
        else self.robot.data.heading_w[ids]
      )
      target_pos = self.robot.data.geom_pose_w[
        ids, self.target_geom_ids[self.target[ids]], :3
      ]
      desired_pos = target_pos.clone()
      desired_pos[:, 0] += self.cfg.start_distance * torch.cos(heading)
      desired_pos[:, 1] += self.cfg.start_distance * torch.sin(heading)
      mocap_id = self.pusher.indexing.mocap_id
      assert mocap_id is not None
      current_pos = self.pusher.data.data.mocap_pos[ids, mocap_id]
      current_quat = self.pusher.data.data.mocap_quat[ids, mocap_id]
      delta = desired_pos - current_pos
      # The fixture tracks lateral motion. Normal-axis motion is force feedback
      # after first contact and is explicitly speed-limited.
      axis = torch.stack(
        (torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=-1
      )
      delta = delta - (delta * axis).sum(dim=-1)[:, None] * axis
      max_step = self.follow_speed_m_s * env.step_dt
      delta = (
        delta
        * (max_step / torch.linalg.vector_norm(delta, dim=-1).clamp_min(max_step))[
          :, None
        ]
      )
      normal_speed = (
        self.base_force_gain_m_s_per_n * (self.target_force_n - normal[ids])
      ).clamp(min=-0.10, max=0.30)
      normal_speed = torch.where(
        self.seen_contact[ids], normal_speed, torch.zeros_like(normal_speed)
      )
      new_pos = current_pos + delta - axis * (normal_speed * env.step_dt)[:, None]
      current_yaw = 2.0 * torch.atan2(current_quat[:, 3], current_quat[:, 0])
      desired_yaw = heading + torch.pi
      yaw_error = torch.atan2(
        torch.sin(desired_yaw - current_yaw), torch.cos(desired_yaw - current_yaw)
      )
      new_yaw = current_yaw + yaw_error.clamp(
        min=-self.follow_yaw_rad_s * env.step_dt,
        max=self.follow_yaw_rad_s * env.step_dt,
      )
      pose = torch.zeros((len(ids), 7), device=self.device)
      pose[:, :3] = new_pos
      pose[:, 3] = torch.cos(0.5 * new_yaw)
      pose[:, 6] = torch.sin(0.5 * new_yaw)
      self.pusher.write_mocap_pose_to_sim(pose, env_ids=ids)
    velocity = self.pusher.data.joint_vel[:, 0]
    force_request = self.target_force_n + self.force_gain * (
      self.target_force_n - normal
    )
    tracking_speed = torch.clamp(
      self.target_speed_m_s + 0.006 * (self.target_force_n - normal),
      min=0.0,
      max=self.recovery_speed_m_s,
    )
    tracking_speed = torch.where(
      self.seen_contact, tracking_speed, self.target_speed_m_s
    )
    speed_limit = self.speed_gain * (tracking_speed - velocity)
    if self.one_way_speed_limit:
      speed_limit = speed_limit.clamp_min(0.0)
    effort = torch.minimum(force_request, speed_limit)
    if not self.motor_hold_after_contact:
      effort = torch.where(self.seen_contact, -self.speed_gain * velocity, effort)
    effort = effort.clamp(min=-self.max_drive_force, max=self.max_drive_force)
    effort = torch.where(self.phase == int(RailPhase.DRIVE), effort, 0.0)
    self.commanded_effort_n = effort
    dof = int(self.pusher.indexing.joint_v_adr[0])
    env.sim.data.qfrc_applied[:, dof] = effort


class KinematicForceProbe(RailPusherController):
  """Track the chest with a spherical probe and regulate measured contact load."""

  target_force_n = 20.0
  force_to_speed_m_s_per_n = 0.001
  max_probe_speed_m_s = 0.50
  max_probe_yaw_rad_s = 2.0

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.approach_m = torch.zeros(self.num_envs, device=self.device)
    self.measured_normal_n = torch.zeros(self.num_envs, device=self.device)
    self.commanded_effort_n = torch.zeros(self.num_envs, device=self.device)
    self.seen_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self.approach_m[self._ids(env_ids)] = 0.0
    self.seen_contact[self._ids(env_ids)] = False

  def __call__(self, env, env_ids=None, rail_cfg=None):
    super().__call__(env, env_ids=env_ids, rail_cfg=rail_cfg)
    history = self.contact_sensor.data.force_history
    if history is None:
      raise RuntimeError("Probe requires substep contact force history")
    normal = history[..., 0].abs().sum(dim=1).mean(dim=-1)
    self.measured_normal_n = normal
    self.seen_contact |= normal >= 1.0
    dof = int(self.pusher.indexing.joint_v_adr[0])
    env.sim.data.qfrc_applied[:, dof] = 0.0
    drive_ids = (self.phase == int(RailPhase.DRIVE)).nonzero().flatten()
    if drive_ids.numel() == 0:
      return
    zero = torch.zeros((len(drive_ids), 1), device=self.device)
    self.pusher.write_joint_state_to_sim(zero, zero, env_ids=drive_ids)
    regulated_speed = (
      self.force_to_speed_m_s_per_n * (self.target_force_n - normal[drive_ids])
    ).clamp(min=-0.03, max=0.03)
    approach_speed = min(0.08, 0.005 * self.target_force_n)
    speed = torch.where(
      self.seen_contact[drive_ids],
      regulated_speed,
      torch.full_like(regulated_speed, approach_speed),
    )
    self.approach_m[drive_ids] = (
      self.approach_m[drive_ids] + speed * env.step_dt
    ).clamp(min=0.0, max=0.20)
    heading = self.robot.data.heading_w[drive_ids]
    target_pos = self.robot.data.geom_pose_w[
      drive_ids, self.target_geom_ids[self.target[drive_ids]], :3
    ]
    axis = torch.stack(
      (torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=-1
    )
    desired_pos = (
      target_pos
      + (self.cfg.start_distance - self.approach_m[drive_ids])[:, None] * axis
    )
    mocap_id = self.pusher.indexing.mocap_id
    assert mocap_id is not None
    current_pos = self.pusher.data.data.mocap_pos[drive_ids, mocap_id]
    current_quat = self.pusher.data.data.mocap_quat[drive_ids, mocap_id]
    delta = desired_pos - current_pos
    max_step = self.max_probe_speed_m_s * env.step_dt
    delta = (
      delta
      * (max_step / torch.linalg.vector_norm(delta, dim=-1).clamp_min(max_step))[
        :, None
      ]
    )
    current_yaw = 2.0 * torch.atan2(current_quat[:, 3], current_quat[:, 0])
    desired_yaw = heading + torch.pi
    yaw_error = torch.atan2(
      torch.sin(desired_yaw - current_yaw), torch.cos(desired_yaw - current_yaw)
    )
    new_yaw = current_yaw + yaw_error.clamp(
      min=-self.max_probe_yaw_rad_s * env.step_dt,
      max=self.max_probe_yaw_rad_s * env.step_dt,
    )
    pose = torch.zeros((len(drive_ids), 7), device=self.device)
    pose[:, :3] = current_pos + delta
    pose[:, 3] = torch.cos(0.5 * new_yaw)
    pose[:, 6] = torch.sin(0.5 * new_yaw)
    self.pusher.write_mocap_pose_to_sim(pose, env_ids=drive_ids)


def longest_true_run(mask: np.ndarray) -> int:
  best = current = 0
  for value in mask:
    current = current + 1 if value else 0
    best = max(best, current)
  return best


def main(
  out_dir: str = "logs/reproduction/exp022_sustained_contact",
  seed: int = 42,
  video: bool = True,
  target_force_n: float = 30.0,
  sphere_radius_m: float = 0.175,
  sphere_mass_kg: float = 5.0,
  rail_joint_damping_n_s_per_m: float = 0.0,
  max_drive_force_weight_ratio: float = 0.15,
  max_travel_m: float = 0.75,
  target_speed_m_s: float = 0.12,
  force_gain: float = 0.5,
  speed_gain: float = 500.0,
  follow_speed_m_s: float = 0.50,
  follow_yaw_rad_s: float = 2.0,
  base_force_gain_m_s_per_n: float = 0.01,
  motor_hold_after_contact: bool = False,
  freeze_push_heading: bool = False,
  one_way_speed_limit: bool = False,
  mask_actor_rail_contact: bool = False,
  actor_contact_source: Literal["oracle", "estimated"] = "oracle",
  shadow_estimator: bool = False,
  estimator_candidate_bundle: str = "",
  estimator_candidate_sha256: str = "",
  estimator_candidate_mapping: Literal[
    "dynamic", "chest_fixed", "point_only", "normal_only"
  ] = "dynamic",
  actor_contact_gain: float = 1.0,
  capture_shadow_inputs: bool = False,
  contact_time_constant_s: float = 0.01,
  kinematic_probe: bool = False,
  pusher_mode: Literal["sustained", "original"] = "sustained",
  max_gpu_memory_gib: float = 30.0,
  min_free_gpu_gib: float = 35.0,
) -> None:
  if (
    target_force_n <= 0
    or sphere_radius_m <= 0
    or sphere_mass_kg <= 0
    or rail_joint_damping_n_s_per_m < 0
    or not 0.0 <= actor_contact_gain <= 1.0
    or max_travel_m < 0.375
    or target_speed_m_s <= 0
  ):
    raise ValueError("Invalid force-regulator geometry or target")
  if pusher_mode == "original" and kinematic_probe:
    raise ValueError("Original pusher cannot be combined with the kinematic probe")
  if actor_contact_source == "estimated" and (
    mask_actor_rail_contact or actor_contact_gain != 1.0
  ):
    raise ValueError("Estimated contact input requires unmasked unit actor gain")
  if shadow_estimator or actor_contact_source == "estimated":
    capture_shadow_inputs = True
  estimator_needed = shadow_estimator or actor_contact_source == "estimated"
  if estimator_candidate_bundle and (not estimator_needed or not estimator_candidate_sha256):
    raise ValueError("Candidate bundle requires estimator inference and its SHA256")
  if estimator_candidate_mapping != "dynamic" and not estimator_candidate_bundle:
    raise ValueError("Candidate mapping ablation requires a candidate bundle")
  estimator = (
    Exp022ShoulderRescueEstimator(
      estimator_candidate_bundle, expected_sha256=estimator_candidate_sha256
    )
    if estimator_candidate_bundle
    else load_exp022_chest_estimator() if estimator_needed else None
  )
  chest_mapping = (
    load_exp022_chest_estimator()
    if estimator_candidate_mapping != "dynamic"
    else None
  )
  free, total = torch.cuda.mem_get_info(0)
  if free < min_free_gpu_gib * 1024**3:
    raise RuntimeError(f"GPU has only {free / 1024**3:.1f} GiB free")
  torch.cuda.set_per_process_memory_fraction(
    min(max_gpu_memory_gib / (total / 1024**3), 0.95), 0
  )
  output = Path(out_dir)
  output.mkdir(parents=True, exist_ok=True)
  saved = json.loads(
    Path("policy/chest_speed_gradient_v1/eval_config.json").read_text()
  )
  saved.update(
    num_envs=1,
    num_episodes=1,
    seed=seed,
    video=video,
    device="cuda:0",
    output=str(output / "evaluation.json"),
    video_folder=str(output / "video"),
    video_length=600,
    sphere_radius=sphere_radius_m,
    max_force_weight_ratio=max_drive_force_weight_ratio,
  )
  cfg = evaluation.EvalCfg(**saved)
  ForceRegulatedRail.target_force_n = target_force_n
  ForceRegulatedRail.force_gain = force_gain
  ForceRegulatedRail.speed_gain = speed_gain
  ForceRegulatedRail.target_speed_m_s = target_speed_m_s
  ForceRegulatedRail.follow_speed_m_s = follow_speed_m_s
  ForceRegulatedRail.follow_yaw_rad_s = follow_yaw_rad_s
  ForceRegulatedRail.base_force_gain_m_s_per_n = base_force_gain_m_s_per_n
  ForceRegulatedRail.motor_hold_after_contact = motor_hold_after_contact
  ForceRegulatedRail.freeze_push_heading = freeze_push_heading
  ForceRegulatedRail.one_way_speed_limit = one_way_speed_limit
  KinematicForceProbe.target_force_n = target_force_n
  original_load_cfg = evaluation.load_env_cfg
  original_env = evaluation.ManagerBasedRlEnv
  original_wrapper = evaluation.RslRlVecEnvWrapper
  effective_actor_contact_gain = 0.0 if mask_actor_rail_contact else actor_contact_gain
  state: dict = {
    "env": None,
    "rows": [],
    "peak_gpu_mib": 0,
    "substeps": [],
    "capture": False,
    "shadow_steps": [],
    "estimator": estimator,
    "adapter": None,
    "initial_observation_seen": False,
  }

  def load_cfg(*args, **kwargs):
    del args, kwargs
    rail_cfg = RailPusherCfg(
      sphere_radius=sphere_radius_m,
      sphere_mass=sphere_mass_kg,
      max_travel=max_travel_m,
      max_force_weight_ratio=max_drive_force_weight_ratio,
      force_done_ratio=0.03 if pusher_mode == "original" else 0.0,
    )
    env_cfg = g1_force_unloading_smp_env_cfg(play=True, rail_cfg=rail_cfg)
    if contact_time_constant_s != 0.01 or rail_joint_damping_n_s_per_m > 0:
      entity = env_cfg.scene.entities["rail_pusher"]
      original_spec = entity.spec_fn

      def compliant_spec():
        spec = original_spec()
        for body in spec.worldbody.bodies:
          for geom in body.geoms:
            if geom.name == "sphere":
              geom.solref = (contact_time_constant_s, 1.0)
          for joint in body.joints:
            if joint.name == "rail_joint":
              joint.damping = np.array([rail_joint_damping_n_s_per_m, 0.0, 0.0])
        return spec

      entity.spec_fn = compliant_spec
    if pusher_mode == "sustained":
      env_cfg.events["rail_pusher_controller"].func = (
        KinematicForceProbe if kinematic_probe else ForceRegulatedRail
      )
    if capture_shadow_inputs or estimator is not None:
      add_force_estimator_observations(env_cfg)
    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (
      ContactSensorCfg(
        name="rail_force_world_audit",
        primary=ContactMatch(mode="geom", pattern="sphere", entity="rail_pusher"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force", "normal", "tangent", "pos"),
        reduce="none",
        num_slots=8,
        history_length=env_cfg.decimation,
        global_frame=True,
      ),
    )
    return env_cfg

  def make_env(*args, **kwargs):
    env = original_env(*args, **kwargs)
    state["env"] = env
    sphere_id = env.sim.mj_model.geom("rail_pusher/sphere").id
    original_step = env.sim.step

    def audit_step(*step_args, **step_kwargs):
      result = original_step(*step_args, **step_kwargs)
      if state["capture"]:
        contact = env.sim.data.contact
        world_ids = contact.worldid.cpu().numpy()
        pairs = contact.geom.cpu().numpy()
        names: list[str] = []
        for pair in pairs[world_ids == 0]:
          if sphere_id in pair:
            other = int(pair[1] if pair[0] == sphere_id else pair[0])
            name = env.sim.mj_model.geom(other).name
            if name.startswith("robot/"):
              names.append(name)
        state["substeps"].append(names)
      return result

    env.sim.step = audit_step
    return env

  class RecordingWrapper(original_wrapper):
    def _apply_actor_contact(self, result, prediction=None):
      actor = result["actor"]
      if actor.shape[-1] != 128:
        raise RuntimeError(f"Expected 128D EXP-022 actor input, got {actor.shape}")
      if actor_contact_source == "oracle":
        actor[:, -32:] *= effective_actor_contact_gain
      elif prediction is None:
        actor[:, -32:] = 0
      else:
        adapter = state["adapter"]
        env = state["env"]
        robot = env.scene["robot"]
        torso_id = robot.body_names.index("torso_link")
        device = actor.device
        normal = torch.as_tensor(
          prediction["policy_normal_force_body_n"], device=device
        ).reshape(1, 3)
        point = torch.as_tensor(
          prediction["policy_point_body_m"], device=device
        ).reshape(1, 3)
        adapter.step(
          normal,
          point,
          torso_quat_w=robot.data.body_link_quat_w[:, torso_id],
          torso_pos_w=robot.data.body_link_pos_w[:, torso_id],
          root_quat_w=robot.data.root_link_quat_w,
          root_pos_w=robot.data.root_link_pos_w,
        )
        actor[:, -32:] = adapter.history
        if not torch.allclose(actor[:, -8:], adapter.history[:, -8:], atol=1e-6):
          raise RuntimeError("Estimated actor history was not installed")
      return result

    def get_observations(self):
      result = super().get_observations()
      if estimator is not None and not state["initial_observation_seen"]:
        estimator.reset()
        state["adapter"] = Exp022ContactAdapter.from_controller(
          state["env"]._rail_controller
        )
        state["adapter"].reset()
        state["initial_observation_seen"] = True
      return self._apply_actor_contact(result)

    def step(self, actions):
      state["substeps"].clear()
      state["capture"] = True
      try:
        result = super().step(actions)
      finally:
        state["capture"] = False
      env = state["env"]
      if len(state["substeps"]) != env.cfg.decimation:
        raise RuntimeError(
          f"Expected {env.cfg.decimation} physics substeps, got {len(state['substeps'])}"
        )
      controller = env._rail_controller
      robot = env.scene["robot"]
      torso_id = robot.body_names.index("torso_link")
      world = (
        env.scene["rail_force_world_audit"].data.force_history.sum(dim=1).mean(dim=1)
      )
      body = quat_apply_inverse(robot.data.body_link_quat_w[:, torso_id], world)[0]
      geometries = sorted(set(name for names in state["substeps"] for name in names))
      if capture_shadow_inputs:
        observation = result[0]
        validate_force_estimator_input(env, observation)
        controller = env._rail_controller
        if not torch.allclose(
          observation["actor"][0, -8:], controller.contact_obs[0], atol=1e-5
        ):
          raise RuntimeError(
            "Actor contact-history tail differs from current oracle term"
          )
        state["shadow_steps"].append(
          {
            "x": observation["force_estimator_x"][0].detach().cpu().numpy().copy(),
            "tau_hist": observation["force_estimator_tau"][0]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "true_force_body_n": body.detach().cpu().numpy().copy(),
            "oracle_history": observation["actor"][0, -32:]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "oracle_contact_frame": controller.contact_obs[0]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "actor_base_obs": observation["actor"][0, :-32]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "action": actions[0].detach().cpu().numpy().copy(),
            "torso_quat_wxyz": robot.data.body_link_quat_w[0, torso_id]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "torso_pos_w": robot.data.body_link_pos_w[0, torso_id]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "root_quat_wxyz": robot.data.root_link_quat_w[0]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "root_pos_w": robot.data.root_link_pos_w[0].detach().cpu().numpy().copy(),
            "target_site_id": int(controller.target[0]),
            "contact_geometries": ";".join(geometries),
          }
        )
      prediction = None
      if estimator is not None:
        observation = result[0]
        validate_force_estimator_input(env, observation)
        prediction = estimator.step(
          observation["force_estimator_x"][0].detach().cpu().numpy(),
          observation["force_estimator_tau"][0].detach().cpu().numpy(),
        )
        if chest_mapping is not None:
          magnitude = float(np.linalg.norm(prediction["force_body_n"]))
          if estimator_candidate_mapping in ("chest_fixed", "point_only"):
            prediction["policy_normal_force_body_n"] = (
              magnitude * chest_mapping.normal_body
            ).astype(np.float32)
          if estimator_candidate_mapping in ("chest_fixed", "normal_only"):
            prediction["policy_point_body_m"] = chest_mapping.point_body.copy()
      contact_history = controller.contact_sensor.data.force_history
      measured_normal = contact_history[..., 0].abs().sum(dim=1).mean(dim=-1)
      rail_dof = int(controller.pusher.indexing.joint_v_adr[0])
      state["rows"].append(
        {
          "step": len(state["rows"]) + 1,
          "time_s": (len(state["rows"]) + 1) * env.step_dt,
          "force_x_n": float(body[0]),
          "force_y_n": float(body[1]),
          "force_z_n": float(body[2]),
          "force_resultant_n": float(torch.linalg.vector_norm(body)),
          "normal_mean_n": float(measured_normal[0]),
          "normal_peak_5ms_n": float(controller.raw_force_peak[0]),
          "motor_effort_n": float(env.sim.data.qfrc_applied[0, rail_dof]),
          "pusher_joint_pos_m": float(controller.pusher.data.joint_pos[0, 0]),
          "pusher_joint_vel_m_s": float(controller.pusher.data.joint_vel[0, 0]),
          "phase": int(controller.phase[0]),
          "contact_geometries": ";".join(geometries),
          "contact_substeps": sum(bool(names) for names in state["substeps"]),
          "root_x_m": float(robot.data.root_link_pos_w[0, 0]),
          "root_y_m": float(robot.data.root_link_pos_w[0, 1]),
          "done": int(bool(result[2][0])),
        }
      )
      if len(state["rows"]) % 25 == 0:
        own = own_gpu_mib()
        if own is not None:
          state["peak_gpu_mib"] = max(state["peak_gpu_mib"], own)
          if own > max_gpu_memory_gib * 1024:
            raise RuntimeError(f"Own GPU memory {own / 1024:.2f} GiB exceeds cap")
      self._apply_actor_contact(result[0], prediction)
      if prediction is not None:
        state["rows"][-1].update(
          estimated_force_x_n=float(prediction["force_body_n"][0]),
          estimated_force_y_n=float(prediction["force_body_n"][1]),
          estimated_force_z_n=float(prediction["force_body_n"][2]),
          estimated_force_resultant_n=float(np.linalg.norm(prediction["force_body_n"])),
          estimated_contact_probability=float(prediction["contact_probability"]),
          actor_contact_active=float(result[0]["actor"][0, -8]),
          actor_contact_force_ratio=float(result[0]["actor"][0, -7]),
          oracle_contact_active=float(controller.contact_obs[0, 0]),
          oracle_contact_force_ratio=float(controller.contact_obs[0, 1]),
          actor_contact_frame_l2=float(
            torch.linalg.vector_norm(
              result[0]["actor"][0, -8:] - controller.contact_obs[0]
            )
          ),
        )
        if estimator_candidate_bundle:
          state["rows"][-1].update(
            shoulder_probability=float(prediction["shoulder_probability"]),
            shoulder_force_rescued=int(prediction["shoulder_force_rescued"]),
            shoulder_mapping_active=int(prediction["shoulder_mapping_active"]),
          )
        if capture_shadow_inputs:
          state["shadow_steps"][-1].update(
            estimated_force_body_n=prediction["force_body_n"].copy(),
            estimated_contact_probability=np.float32(prediction["contact_probability"]),
            actor_contact_history=result[0]["actor"][0, -32:]
            .detach()
            .cpu()
            .numpy()
            .copy(),
          )
          if estimator_candidate_bundle:
            state["shadow_steps"][-1].update(
              shoulder_probability=np.float32(prediction["shoulder_probability"]),
              shoulder_force_rescued=np.bool_(prediction["shoulder_force_rescued"]),
              shoulder_mapping_active=np.bool_(prediction["shoulder_mapping_active"]),
            )
      return result

  evaluation.load_env_cfg = load_cfg
  evaluation.ManagerBasedRlEnv = make_env
  evaluation.RslRlVecEnvWrapper = RecordingWrapper
  try:
    evaluation.main(cfg)
  finally:
    evaluation.load_env_cfg = original_load_cfg
    evaluation.ManagerBasedRlEnv = original_env
    evaluation.RslRlVecEnvWrapper = original_wrapper

  rows = state["rows"]
  if not rows:
    raise RuntimeError("No simulation frames recorded")
  with (output / "contact_trace.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  force = np.array([row["force_resultant_n"] for row in rows])
  contact = np.array([row["contact_substeps"] > 0 for row in rows])
  loaded = force >= 10.0
  frame_geoms = [row["contact_geometries"] for row in rows]
  torso_loaded = loaded & np.array(
    ["robot/torso_collision" in frame.split(";") for frame in frame_geoms]
  )
  evaluation_report = json.loads((output / "evaluation.json").read_text())
  terminations = {
    key: value["mean"]
    for key, value in evaluation_report["metrics"].items()
    if key.startswith("termination/")
  }
  no_failure = all(
    value == 0.0 for key, value in terminations.items() if key != "termination/time_out"
  )
  audit = {
    "protocol": f"EXP-022 frozen policy (actor contact gain {effective_actor_contact_gain}); physical sphere {sphere_radius_m} m / {sphere_mass_kg} kg; "
    + (
      "original EXP-022 rail push/retract"
      if pusher_mode == "original"
      else (
        "kinematic force-controlled tracking probe"
        if kinematic_probe
        else "low-speed rail approach; "
        + (
          "rail-motor force hold" if motor_hold_after_contact else "base force feedback"
        )
        + ("; fixed push heading" if freeze_push_heading else "; moving heading")
      )
    ),
    "target_force_n": target_force_n,
    "robot_weight_n": float(state["env"]._rail_controller.weight),
    "seed": seed,
    "sphere_radius_m": sphere_radius_m,
    "sphere_mass_kg": sphere_mass_kg,
    "rail_joint_damping_n_s_per_m": rail_joint_damping_n_s_per_m,
    "motor_force_cap_ratio": max_drive_force_weight_ratio,
    "max_travel_m": max_travel_m,
    "target_speed_m_s": target_speed_m_s,
    "force_gain": force_gain,
    "speed_gain": speed_gain,
    "follow_speed_m_s": follow_speed_m_s,
    "follow_yaw_rad_s": follow_yaw_rad_s,
    "base_force_gain_m_s_per_n": base_force_gain_m_s_per_n,
    "motor_hold_after_contact": motor_hold_after_contact,
    "freeze_push_heading": freeze_push_heading,
    "one_way_speed_limit": one_way_speed_limit,
    "mask_actor_rail_contact": mask_actor_rail_contact,
    "actor_contact_source": actor_contact_source,
    "shadow_estimator": estimator is not None,
    "estimator_candidate_bundle": estimator_candidate_bundle or None,
    "estimator_candidate_sha256": estimator_candidate_sha256 or None,
    "estimator_candidate_mapping": estimator_candidate_mapping,
    "actor_contact_gain": effective_actor_contact_gain,
    "capture_shadow_inputs": capture_shadow_inputs,
    "contact_time_constant_s": contact_time_constant_s,
    "kinematic_probe": kinematic_probe,
    "pusher_mode": pusher_mode,
    "frames": len(rows),
    "contact_frames": int(contact.sum()),
    "loaded_ge10n_frames": int(loaded.sum()),
    "longest_loaded_ge10n_frames": longest_true_run(loaded),
    "longest_loaded_ge10n_s": 0.02 * longest_true_run(loaded),
    "longest_torso_loaded_ge10n_s": 0.02 * longest_true_run(torso_loaded),
    "force_peak_20ms_n": float(force.max()),
    "force_peak_5ms_normal_n": max(row["normal_peak_5ms_n"] for row in rows),
    "contact_geometry_counts": {
      name: sum(name in frame.split(";") for frame in frame_geoms)
      for name in sorted(
        set(name for frame in frame_geoms for name in frame.split(";") if name)
      )
    },
    "terminations": terminations,
    "no_failure_termination": no_failure,
    "passed_1s_sustained_contact": longest_true_run(torso_loaded) >= 50 and no_failure,
    "peak_own_gpu_mib": state["peak_gpu_mib"],
    "start_free_gpu_gib": free / 1024**3,
  }
  (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
  if capture_shadow_inputs:
    samples = state["shadow_steps"]
    if len(samples) != len(rows):
      raise RuntimeError("Shadow observation and physics row counts differ")
    np.savez_compressed(
      output / "shadow_inputs.npz",
      **{key: np.stack([sample[key] for sample in samples]) for key in samples[0]},
      step=np.arange(1, len(samples) + 1, dtype=np.int64),
      time_s=np.arange(1, len(samples) + 1, dtype=np.float64) * 0.02,
    )
  print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
  tyro.cli(main)
