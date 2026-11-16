"""Shoulder-only force-unloading task backed by the accepted G1 IK-v2 Prior."""

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

TASK_ID = "Smp-Force-Unloading-Conditional-ShoulderIKV2-G1"
PRIOR_PATH = "datasets/pretrain_ckpt/pretrained_g1_shoulder_ik_v2_conditional_v1.pt"
CALIBRATION_PATH = (
  "logs/reproduction/g1_shoulder_ik_v2_prior_ppo_v1/calibration/final.pt"
)


def shoulder_ik_v2_env_cfg(
  play: bool = False,
  *,
  prior_path: str = PRIOR_PATH,
  calibration_path: str = CALIBRATION_PATH,
):
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
    ckpt_path=prior_path,
    conditional_score_calibration_path=calibration_path,
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


class ShoulderIkV2Runner(MjlabOnPolicyRunner):
  """PPO runner with immutable Prior/calibration identity and full resume state."""

  def save(self, path, infos=None):
    env = self.env.unwrapped
    infos = {
      **(infos or {}),
      "env_state": {"common_step_counter": env.common_step_counter},
      "smp_prior": env._smp_prior_identity,
      "smp_gsi_prior": env._smp_gsi_prior_identity,
      "smp_score": env._smp_score_identity,
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
      raise ValueError("PPO checkpoint IK-v2 Prior/schema mismatch")
    if infos.get("smp_gsi_prior") != env._smp_gsi_prior_identity:
      raise ValueError("PPO checkpoint IK-v2 GSI Prior mismatch")
    if infos.get("smp_score") != env._smp_score_identity:
      raise ValueError("PPO checkpoint IK-v2 score calibration mismatch")
    env_state = infos.get("env_state")
    if not isinstance(env_state, dict) or "common_step_counter" not in env_state:
      raise ValueError("PPO checkpoint is missing the environment step counter")
    env.common_step_counter = int(env_state["common_step_counter"])
    return super().load(path, load_cfg, strict, map_location)


def register_shoulder_ik_v2_task() -> None:
  rl = unitree_g1_smp_ppo_runner_cfg()
  rl.experiment_name = "smp_force_unloading_conditional_shoulder_ik_v2"
  rl.run_name = "g1_shoulder_ik_v2"
  rl.actor.distribution_cfg["init_std"] = 0.10
  rl.algorithm.learning_rate = 1.0e-4
  rl.save_interval = 1000
  rl.upload_model = False
  register_mjlab_task(
    TASK_ID,
    shoulder_ik_v2_env_cfg(),
    shoulder_ik_v2_env_cfg(play=True),
    rl,
    runner_cls=ShoulderIkV2Runner,
  )
