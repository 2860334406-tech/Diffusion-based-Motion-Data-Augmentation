"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.conditional_score import (
  contrastive_score_from_energies,
  opposite_shoulder_conditions,
  paired_condition_predictions,
  shoulder_score_condition_ids,
  weighted_denoising_mse,
)
from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _update_buffer_from_sim(env: ManagerBasedRlEnv) -> None:
  """Push current sim kinematics onto the buffer tail, env-origin-relative
  (matching ``_prime_sim_and_buffer``) so features are placement-invariant."""
  robot = env.scene["robot"]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins
  buffer.update(
    robot.data.root_link_pos_w - origins,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :],
    robot.data.joint_pos,
    robot.data.joint_vel,
  )


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
) -> torch.Tensor:
  """SDS-style guidance reward over fixed timesteps ``K``:
  ``exp(-w_s/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)``.  ``normalize`` divides each MSE by a
  ``DiffNormalizer`` running mean (policy-relative) vs. raw (absolute scale);
  always stashes the mean raw MSE on ``env._smp_raw_err``."""
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  _update_buffer_from_sim(env)

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]
  condition_ids = None
  if getattr(env, "_smp_num_conditions", 0):
    controller = env._rail_controller
    controller.refresh_contact()
    condition_ids = controller.condition_state.reward_ids

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t) if condition_ids is None else model(x_t, t, condition_ids)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        if condition_ids is None:
          total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
        else:
          total_err += normalizer.update_and_normalize(
            t_scalar, mse_per_env, condition_ids
          )
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  """``(Σ wᵢ · taskᵢ(env)) · r_smp`` — multiplicative SMP gating; ``task_terms`` is
  a tuple of ``(func, weight, kwargs)``.  Calls ``smp_guidance_reward`` once (the
  sole SMP-buffer update), so it must be the task's only SMP reward term."""
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  smp_reward = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
  env._smp_reward = smp_reward  # type: ignore[attr-defined]
  return task * smp_reward


def conditional_contrastive_smp_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
) -> torch.Tensor:
  """Frozen calibrated SMP score contrasting the paired shoulder condition.

  Correct and opposite conditions see the exact same noised motion. During a
  debounced contact transition, only frames already assigned to the current
  condition contribute to the energy.
  """
  from smp.rl.tasks.force_unloading.mdp.controller import get_controller

  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
  calibration = env._smp_score_calibration  # type: ignore[attr-defined]
  calibration_timesteps = tuple(int(value) for value in calibration["fixed_timesteps"])
  if tuple(fixed_timesteps) != calibration_timesteps:
    raise ValueError("Runtime score timesteps do not match frozen calibration")

  controller = get_controller(env)
  controller.refresh_contact()
  state = controller.condition_state
  condition_ids, supported = shoulder_score_condition_ids(state.reward_ids)
  opposite_ids = opposite_shoulder_conditions(condition_ids)
  time_mask = state.reward_mask

  _update_buffer_from_sim(env)
  features = env._smp_buffer.compute_features()  # type: ignore[attr-defined]
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  correct_energies: list[torch.Tensor] = []
  opposite_energies: list[torch.Tensor] = []
  weights = calibration["feature_weights"]
  with torch.no_grad():
    for timestep in fixed_timesteps:
      if not 0 <= timestep < scheduler.num_timesteps:
        raise ValueError(
          f"fixed_timestep {timestep} out of range [0, {scheduler.num_timesteps})"
        )
      t = torch.full((env.num_envs,), timestep, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      correct_prediction, opposite_prediction = paired_condition_predictions(
        model,
        x_t,
        t,
        condition_ids,
        opposite_ids,
      )
      correct_energies.append(
        weighted_denoising_mse(correct_prediction, noise, weights, time_mask)
      )
      opposite_energies.append(
        weighted_denoising_mse(opposite_prediction, noise, weights, time_mask)
      )

  correct = torch.stack(correct_energies, dim=-1)
  opposite = torch.stack(opposite_energies, dim=-1)
  timestep_indexes = torch.tensor(
    [calibration_timesteps.index(value) for value in fixed_timesteps],
    device=device,
  )
  references = calibration["reference_mean"][condition_ids][:, timestep_indexes]
  reward, components = contrastive_score_from_energies(
    correct,
    opposite,
    references,
    shoulder_mask=(condition_ids == 1) | (condition_ids == 2),
  )
  reward = torch.where(supported, reward, torch.zeros_like(reward))
  components["base_reward"] = torch.where(
    supported, components["base_reward"], torch.zeros_like(reward)
  )
  components["contrast_reward"] = torch.where(
    supported, components["contrast_reward"], torch.zeros_like(reward)
  )
  if not bool(torch.isfinite(reward).all()):
    raise FloatingPointError("Conditional contrastive SMP reward contains NaN/Inf")

  env._smp_reward = reward  # type: ignore[attr-defined]
  env._smp_raw_err = components["correct_energy"]  # type: ignore[attr-defined]
  for name, value in components.items():
    setattr(env, f"_smp_{name}", value)
  env._smp_unexpected_condition = (~supported).float()  # type: ignore[attr-defined]
  env._smp_valid_history_frames = time_mask.sum(-1).float()  # type: ignore[attr-defined]
  env._smp_condition_id = state.ids.float()  # type: ignore[attr-defined]
  return reward


def task_conditional_contrastive_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
) -> torch.Tensor:
  """Task reward multiplied by the calibrated conditional contrast score."""
  task = sum(weight * func(env, **kwargs) for func, weight, kwargs in task_terms)
  return task * conditional_contrastive_smp_reward(env, fixed_timesteps)
