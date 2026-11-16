"""Five-contact-condition SMP task. Historical task defaults stay unchanged."""
from __future__ import annotations

from pathlib import Path

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)
from smp.rl.tasks.force_unloading.mdp.conditions import conditional_metric
from smp.rl.tasks.force_unloading.mdp.controller import RailPusherCfg

TASK_ID = "Smp-Force-Unloading-Conditional-G1"
PRIOR_PATH = "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v1.pt"


def conditional_env_cfg(play=False, rail_cfg=None):
  rail_cfg = rail_cfg or RailPusherCfg(
    conditional=True, sphere_radius=.10, sphere_mass=5.,
    start_surface_gap=.02, target_probabilities=(.25, .25, .25, .25),
    push_direction_mode="front_to_back", stationary_speed_scale=10.,
  )
  cfg = g1_force_unloading_smp_env_cfg(play, rail_cfg)
  cfg.scene.sensors += (ContactSensorCfg(
    name="rail_contact_regions",
    primary=ContactMatch(mode="geom", entity="robot", pattern=(
      "torso_collision", "left_shoulder_yaw_collision", "right_shoulder_yaw_collision",
    )),
    secondary=ContactMatch(mode="geom", entity="rail_pusher", pattern="sphere"),
    fields=("found", "force", "pos"), reduce="none", num_slots=4,
    history_length=cfg.decimation,
  ),)
  cfg.events["init_smp_state"].params.update(ckpt_path=PRIOR_PATH, gsi_max_root_planar_speed=.15)
  for name in ("contact_coverage", "response_direction_correct", "recovered_static",
               "unclassified_contact"):
    cfg.metrics[name] = MetricsTermCfg(func=conditional_metric, params={"name": name}, reduce="max")
  for name in ("waist_rotation_rad", "pelvis_rotation_rad", "response_x_displacement_m"):
    cfg.metrics[name] = MetricsTermCfg(func=conditional_metric, params={"name": name}, reduce="last")
  return cfg


class ConditionalSmpRunner(MjlabOnPolicyRunner):
  def save(self, path, infos=None):
    env = self.env.unwrapped
    infos = {
      **(infos or {}), "env_state": {"common_step_counter": env.common_step_counter},
      "smp_prior": env._smp_prior_identity,
      "smp_normalizer": env._smp_normalizer.state_dict(),
    }
    state = self.alg.save()
    state.update(iter=self.current_learning_iteration, infos=infos)
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    # Reject mismatched prior before mutating policy or optimizer state.
    state = torch.load(path, map_location="cpu", weights_only=True)
    infos = state.get("infos", {})
    env = self.env.unwrapped
    if infos.get("smp_prior") != env._smp_prior_identity:
      raise ValueError("PPO checkpoint conditional prior/schema mismatch")
    env._smp_normalizer.load_state_dict(infos["smp_normalizer"])
    return super().load(path, load_cfg, strict, map_location)


def register_conditional_task():
  rl = unitree_g1_smp_ppo_runner_cfg()
  rl.experiment_name = "smp_force_unloading_conditional_g1"
  rl.run_name = "conditional_contact_v1"
  rl.actor.distribution_cfg["init_std"] = .10
  rl.algorithm.learning_rate = 1e-4
  rl.save_interval = 1000
  register_mjlab_task(TASK_ID, conditional_env_cfg(), conditional_env_cfg(play=True),
                     rl, runner_cls=ConditionalSmpRunner)
