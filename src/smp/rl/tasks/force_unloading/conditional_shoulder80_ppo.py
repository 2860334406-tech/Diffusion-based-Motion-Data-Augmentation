"""Exploratory shoulder-only PPO using the unaccepted Shoulder80 Prior.

This task is deliberately isolated from the historical Conditional and
ShoulderScoreV3 tasks.  It exists only because the user explicitly overrode the
Prior score gate; the task identity never implies that the Prior was accepted.
"""

from __future__ import annotations

from pathlib import Path

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rewards import task_conditional_contrastive_smp_product
from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.force_unloading.conditional import conditional_env_cfg
from smp.rl.tasks.force_unloading.mdp import force_unloading_task_reward, smp_component
from smp.rl.tasks.force_unloading.mdp.conditions import conditional_metric
from smp.rl.tasks.force_unloading.mdp.controller import RailPusherCfg

TASK_ID = "Smp-Force-Unloading-Conditional-Shoulder80V1-G1"
PRIOR_PATH = (
  "logs/reproduction/conditional_shoulder80_prior_v1_expanded/prior/checkpoint_02000.pt"
)
GSI_PRIOR_PATH = "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
CALIBRATION_PATH = (
  "logs/reproduction/conditional_shoulder80_ppo_v1_override/"
  "calibration/runtime_calibration.pt"
)


def shoulder80_ppo_env_cfg(play: bool = False):
  rail_cfg = RailPusherCfg(
    conditional=True,
    condition_transition_mode="masked_current",
    sphere_radius=0.10,
    sphere_mass=5.0,
    start_surface_gap=0.02,
    target_probabilities=(0.0, 0.5, 0.5, 0.0),
    push_direction_mode="front_to_back",
    stationary_speed_scale=10.0,
  )
  cfg = conditional_env_cfg(play=play, rail_cfg=rail_cfg)
  cfg.events["init_smp_state"].params.update(
    ckpt_path=PRIOR_PATH,
    gsi_ckpt_path=GSI_PRIOR_PATH,
    conditional_score_calibration_path=CALIBRATION_PATH,
    gsi_max_root_planar_speed=0.15,
  )
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_conditional_contrastive_smp_product,
    weight=1.0,
    params={"task_terms": ((force_unloading_task_reward, 1.0, {}),)},
  )
  for name in (
    "correct_energy",
    "opposite_energy",
    "normalized_correct_energy",
    "contrast_margin",
    "base_reward",
    "contrast_reward",
    "valid_history_frames",
    "condition_id",
    "unexpected_condition",
  ):
    cfg.metrics[f"smp_{name}"] = MetricsTermCfg(
      func=smp_component,
      params={"name": name},
      reduce="mean",
    )
  for name in ("signed_waist_response_peak_rad", "pelvis_yaw_abs_peak_rad"):
    cfg.metrics[name] = MetricsTermCfg(
      func=conditional_metric,
      params={"name": name},
      reduce="max",
    )
  return cfg


class Shoulder80PpoRunner(MjlabOnPolicyRunner):
  """Runner with strict three-condition Prior and score identity checks."""

  def save(self, path, infos=None):
    env = self.env.unwrapped
    infos = {
      **(infos or {}),
      "env_state": {"common_step_counter": env.common_step_counter},
      "smp_prior": env._smp_prior_identity,
      "smp_gsi_prior": env._smp_gsi_prior_identity,
      "smp_score": env._smp_score_identity,
      "prior_acceptance_override": True,
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
    state = torch.load(path, map_location="cpu", weights_only=True)
    infos = state.get("infos", {})
    env = self.env.unwrapped
    if infos.get("smp_prior") != env._smp_prior_identity:
      raise ValueError("PPO checkpoint Shoulder80 Prior/schema mismatch")
    if infos.get("smp_gsi_prior") != env._smp_gsi_prior_identity:
      raise ValueError("PPO checkpoint Conditional-v2 GSI Prior/schema mismatch")
    if infos.get("smp_score") != env._smp_score_identity:
      raise ValueError("PPO checkpoint Shoulder80 score calibration mismatch")
    if infos.get("prior_acceptance_override") is not True:
      raise ValueError("PPO checkpoint lacks explicit failed-Prior override identity")
    return super().load(path, load_cfg, strict, map_location)


def register_shoulder80_ppo_task() -> None:
  rl = unitree_g1_smp_ppo_runner_cfg()
  rl.experiment_name = "smp_force_unloading_conditional_shoulder80_v1"
  rl.run_name = "conditional_shoulder80_v1_override"
  rl.actor.distribution_cfg["init_std"] = 0.10
  rl.algorithm.learning_rate = 1.0e-4
  rl.save_interval = 1000
  rl.upload_model = False
  register_mjlab_task(
    TASK_ID,
    shoulder80_ppo_env_cfg(),
    shoulder80_ppo_env_cfg(play=True),
    rl,
    runner_cls=Shoulder80PpoRunner,
  )
