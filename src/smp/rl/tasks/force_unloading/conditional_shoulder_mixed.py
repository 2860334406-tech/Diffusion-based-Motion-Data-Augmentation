"""Shoulder-only force-unloading task backed by the accepted G1 mixed-source Prior."""

from __future__ import annotations

from pathlib import Path

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply

from smp.rl.rewards import task_conditional_contrastive_smp_product
from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.force_unloading.conditional import conditional_env_cfg
from smp.rl.tasks.force_unloading.mdp import force_unloading_task_reward, smp_component
from smp.rl.tasks.force_unloading.mdp.conditions import conditional_metric
from smp.rl.tasks.force_unloading.mdp.controller import RailPusherCfg

TASK_ID = "Smp-Force-Unloading-Conditional-ShoulderMixedV1-G1"
PRIOR_PATH = "datasets/pretrain_ckpt/pretrained_g1_shoulder_mixed_conditional_v1.pt"
CALIBRATION_PATH = (
  "logs/reproduction/g1_shoulder_mixed_prior_ppo_v1/calibration/final.pt"
)


def mixed_response_metrics(env):
  """Track torso-fixed shoulder markers, permitting coordinated pelvis rotation."""
  c = env._rail_controller
  robot = c.robot
  n = env.num_envs
  if not hasattr(c, "mixed_response"):
    # Same marker definition as the accepted personal data: shoulder geom centers
    # transformed into the torso frame at the model reference pose.
    import mujoco

    from smp.data.shoulder_ik import _build_model

    model = _build_model(soft_limits=False)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    torso = model.body("torso_link").id
    rotation = data.xmat[torso].reshape(3, 3)
    offsets = [
      rotation.T
      @ (
        data.geom_xpos[model.geom(f"{side}_shoulder_yaw_collision").id]
        - data.xpos[torso]
      )
      for side in ("left", "right")
    ]
    c.mixed_offsets = torch.tensor(
      __import__("numpy").stack(offsets),
      device=env.device,
      dtype=robot.data.root_link_pos_w.dtype,
    )
    c.mixed_response = {
      "started": torch.zeros(n, device=env.device, dtype=torch.bool),
      "root": torch.zeros(n, 3, device=env.device),
      "shoulder": torch.zeros(n, 3, device=env.device),
      "heading": torch.zeros(n, device=env.device),
      "torso_yaw": torch.zeros(n, device=env.device),
      "world_retreat": torch.zeros(n, device=env.device),
      "compensated_retreat": torch.zeros(n, device=env.device),
      "torso_signed_peak": torch.zeros(n, device=env.device),
      "direction": torch.zeros(n, device=env.device),
    }
  state = c.mixed_response
  reset = c.response_condition == 0
  state["started"][reset] = False
  for key in ("world_retreat", "compensated_retreat", "torso_signed_peak", "direction"):
    state[key][reset] = 0
  condition = c.response_condition
  side = (condition == 2).long()
  q = robot.data.body_link_quat_w[:, c.torso_body_id]
  marker = robot.data.body_link_pos_w[:, c.torso_body_id] + quat_apply(
    q, c.mixed_offsets[side]
  )
  matrix = matrix_from_quat(q)
  yaw = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
  root = robot.data.root_link_pos_w
  active = (condition == 1) | (condition == 2)
  first = active & ~state["started"]
  for key, value in (
    ("root", root),
    ("shoulder", marker),
    ("heading", robot.data.heading_w),
    ("torso_yaw", yaw),
  ):
    state[key][first] = value[first]
  state["started"] |= first
  dx = marker - state["shoulder"]
  compensated = dx - (root - state["root"])
  axis = torch.stack((torch.cos(state["heading"]), torch.sin(state["heading"])), -1)
  world = -(dx[:, :2] * axis).sum(-1)
  relative = -(compensated[:, :2] * axis).sum(-1)
  delta = torch.atan2(
    torch.sin(yaw - state["torso_yaw"]), torch.cos(yaw - state["torso_yaw"])
  )
  signed = torch.where(condition == 1, delta, -delta)
  response_window = active & (c.response_steps * env.step_dt <= 2)
  for key, value in (
    ("world_retreat", world),
    ("compensated_retreat", relative),
    ("torso_signed_peak", signed),
  ):
    state[key] = torch.maximum(state[key], torch.where(response_window, value, 0.0))
  correct = response_window & (world >= 0.015) & (relative >= 0.015) & (signed > 0)
  state["direction"] = torch.maximum(state["direction"], correct.float())


def mixed_task_product(env, task_terms, fixed_timesteps=(8, 15, 22)):
  reward = task_conditional_contrastive_smp_product(env, task_terms, fixed_timesteps)
  mixed_response_metrics(env)
  # One call per control frame; each 24 frames is one PPO rollout/iteration.
  if not hasattr(env, "_mixed_zero_guard"):
    env._mixed_zero_guard = {"frames": 0, "zero": 0, "count": 0, "streak": 0}
  guard = env._mixed_zero_guard
  supported = env._smp_unexpected_condition < 0.5
  guard["frames"] += 1
  guard["zero"] += int(((env._smp_reward == 0) & supported).sum())
  guard["count"] += int(supported.sum())
  if guard["frames"] == 24:
    ratio = guard["zero"] / max(guard["count"], 1)
    guard["streak"] = guard["streak"] + 1 if ratio >= 0.99 else 0
    guard.update(frames=0, zero=0, count=0)
    if guard["streak"] >= 20:
      raise RuntimeError("EXP-035 online SMP gate: >=99% exact zeros for 20 rollouts")
  return reward


def mixed_metric(env, name):
  if not hasattr(env._rail_controller, "mixed_response"):
    mixed_response_metrics(env)
  return env._rail_controller.mixed_response[name]


def shoulder_mixed_env_cfg(
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
    func=mixed_task_product,
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
  for output, field in (
    ("response_direction_correct", "direction"),
    ("shoulder_world_retreat_peak_m", "world_retreat"),
    ("shoulder_compensated_retreat_peak_m", "compensated_retreat"),
    ("torso_signed_yaw_peak_rad", "torso_signed_peak"),
  ):
    cfg.metrics[output] = MetricsTermCfg(
      func=mixed_metric, params={"name": field}, reduce="max"
    )
  return cfg


class ShoulderMixedRunner(MjlabOnPolicyRunner):
  """PPO runner with immutable Prior/calibration identity and full resume state."""

  def save(self, path, infos=None):
    env = self.env.unwrapped
    infos = {
      **(infos or {}),
      "env_state": {"common_step_counter": env.common_step_counter},
      "smp_prior": env._smp_prior_identity,
      "smp_gsi_prior": env._smp_gsi_prior_identity,
      "smp_score": env._smp_score_identity,
      "algorithm_learning_rate": getattr(self.alg, "learning_rate", None),
      "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
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
      raise ValueError("PPO checkpoint mixed-source Prior/schema mismatch")
    if infos.get("smp_gsi_prior") != env._smp_gsi_prior_identity:
      raise ValueError("PPO checkpoint mixed-source GSI Prior mismatch")
    if infos.get("smp_score") != env._smp_score_identity:
      raise ValueError("PPO checkpoint mixed-source score calibration mismatch")
    env_state = infos.get("env_state")
    if not isinstance(env_state, dict) or "common_step_counter" not in env_state:
      raise ValueError("PPO checkpoint is missing the environment step counter")
    env.common_step_counter = int(env_state["common_step_counter"])
    loaded = super().load(path, load_cfg, strict, map_location)
    if load_cfg is None and "rng" in infos:
      self.current_learning_iteration = int(state["iter"]) + 1
      if infos.get("algorithm_learning_rate") is not None:
        self.alg.learning_rate = float(infos["algorithm_learning_rate"])
      elif hasattr(self.alg, "optimizer"):
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]["lr"])
      torch.set_rng_state(infos["rng"]["torch"].cpu())
      torch.cuda.set_rng_state_all([value.cpu() for value in infos["rng"]["cuda"]])
    return loaded

  def learn(self, *args, **kwargs):
    try:
      return super().learn(*args, **kwargs)
    except BaseException:
      if self.logger.log_dir:
        self.save(
          Path(self.logger.log_dir)
          / f"interrupted_{self.current_learning_iteration}.pt"
        )
      raise


def register_shoulder_mixed_task() -> None:
  rl = unitree_g1_smp_ppo_runner_cfg()
  rl.experiment_name = "smp_force_unloading_conditional_shoulder_mixed"
  rl.run_name = "g1_shoulder_mixed"
  rl.actor.distribution_cfg["init_std"] = 0.10
  rl.algorithm.learning_rate = 1.0e-4
  rl.save_interval = 1000
  rl.upload_model = False
  register_mjlab_task(
    TASK_ID,
    shoulder_mixed_env_cfg(),
    shoulder_mixed_env_cfg(play=True),
    rl,
    runner_cls=ShoulderMixedRunner,
  )
