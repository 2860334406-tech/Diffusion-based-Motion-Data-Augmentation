"""Deterministic, per-episode evaluation for the G1 force-unloading task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from tensordict import TensorDict

import smp.rl.tasks  # noqa: F401  # register Smp-* tasks


@dataclass(frozen=True)
class EvalCfg:
  checkpoint: str
  output: str
  task_id: str = "Smp-Force-Unloading-G1"
  stationary_reward_weight: float = 0.1
  stage: Literal["a", "b1", "b2", "c"] = "c"
  prior_checkpoint: str = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_balance.pt"
  conditional_score_calibration: str = ""
  shoulder_gate: bool = False
  gsi_max_root_planar_speed: float = 0.5
  stationary_speed_scale: float = 1.0
  sphere_radius: float = 0.175
  target_probabilities: tuple[float, ...] = (1.0, 0.0, 0.0)
  push_direction_mode: Literal["target_default", "front_to_back"] = "target_default"
  num_episodes: int = 1024
  num_envs: int = 256
  seed: int = 42
  device: str = "cuda:0"
  reference_speed: float = 0.05
  max_force_weight_ratio: float = 0.15
  ramp_time: float = 0.25
  wait_range_s: tuple[float, float] = (1.0, 3.0)
  max_steps: int = 20_000
  video: bool = False
  video_length: int = 600
  video_folder: str = ""
  video_height: int = 720
  video_width: int = 1280


def _episode_metric_values(env: ManagerBasedRlEnv, env_ids: torch.Tensor):
  manager = env.metrics_manager
  counts = manager._step_count[env_ids].float().clamp_min(1.0)
  values: dict[str, torch.Tensor] = {}
  for idx, name in enumerate(manager._term_names):
    reduce = manager._term_cfgs[idx].reduce
    if reduce == "max":
      value = manager._episode_max[name][env_ids]
    elif reduce == "last":
      value = manager._step_values[env_ids, idx]
    elif reduce == "sum":
      value = manager._episode_sums[name][env_ids]
    else:
      value = manager._episode_sums[name][env_ids] / counts
    values[name] = value.detach().cpu()
  return values


def _summarize(values: torch.Tensor) -> dict[str, float | int]:
  array = values.numpy()
  if array.size == 0:
    return {
      "count": 0,
      "mean": 0.0,
      "p50": 0.0,
      "p90": 0.0,
      "p95": 0.0,
      "max": 0.0,
    }
  return {
    "count": int(array.size),
    "mean": float(np.mean(array)),
    "p50": float(np.quantile(array, 0.50)),
    "p90": float(np.quantile(array, 0.90)),
    "p95": float(np.quantile(array, 0.95)),
    "max": float(np.max(array)),
  }


def _gate(
  stage: str,
  metrics: dict[str, dict[str, float | int]],
  shoulder_gate: bool = False,
) -> dict:
  if stage == "a":
    checks = {
      "precontact_speed_p95_le_0.30": (
        metrics["precontact_root_planar_speed_m_s"]["p95"] <= 0.30
      ),
      "nonmechanical_failure_rate_le_0.05": (
        metrics["failure_nonmechanical_union"]["mean"] <= 0.05
      ),
      "displacement_p50_le_0.25": (
        metrics["precontact_root_displacement_m"]["p50"] <= 0.25
      ),
    }
  elif stage == "b1":
    checks = {
      "emergency_rate_le_0.10": metrics["failure_emergency_force"]["mean"] <= 0.10,
      "event_success_rate_ge_0.20": metrics["event_success"]["mean"] >= 0.20,
    }
  elif stage == "b2":
    checks = {
      "emergency_rate_le_0.20": metrics["failure_emergency_force"]["mean"] <= 0.20,
      "event_success_rate_ge_0.20": metrics["event_success"]["mean"] >= 0.20,
    }
  else:
    checks = {
      "event_success_rate_ge_0.50": metrics["event_success"]["mean"] >= 0.50,
      "emergency_rate_le_0.10": metrics["failure_emergency_force"]["mean"] <= 0.10,
      "nonmechanical_failure_rate_le_0.05": (
        metrics["failure_nonmechanical_union"]["mean"] <= 0.05
      ),
    }
  if shoulder_gate:
    checks.update(
      {
        "confirmed_contact_coverage_ge_0.80": metrics["contact_coverage"]["mean"]
        >= 0.80,
        "correct_target_contact_coverage_ge_0.80": metrics[
          "contact_condition_classification_correct"
        ]["mean"]
        >= 0.80,
        "direction_correct_given_target_contact_ge_0.80": metrics[
          "response_direction_correct_given_target_contact"
        ]["mean"]
        >= 0.80,
        "signed_waist_peak_given_target_contact_p50_ge_0.08": metrics[
          "signed_waist_response_peak_rad_given_target_contact"
        ]["p50"]
        >= 0.08,
        "pelvis_yaw_abs_peak_given_target_contact_p95_le_0.175": metrics[
          "pelvis_yaw_abs_peak_rad_given_target_contact"
        ]["p95"]
        <= 0.175,
      }
    )
  return {"passed": all(checks.values()), "checks": checks}


def main(cfg: EvalCfg) -> None:
  checkpoint = Path(cfg.checkpoint)
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.auto_reset = False
  init_params = env_cfg.events["init_smp_state"].params
  if cfg.conditional_score_calibration:
    init_params["conditional_score_calibration_path"] = (
      cfg.conditional_score_calibration
    )
  init_params["ckpt_path"] = cfg.prior_checkpoint
  init_params["gsi_max_root_planar_speed"] = cfg.gsi_max_root_planar_speed
  rail_params = env_cfg.events["rail_pusher_controller"].params
  rail_params["rail_cfg"] = replace(
    rail_params["rail_cfg"],
    sphere_radius=cfg.sphere_radius,
    reference_speed=cfg.reference_speed,
    max_force_weight_ratio=cfg.max_force_weight_ratio,
    ramp_time=cfg.ramp_time,
    wait_range_s=cfg.wait_range_s,
    stationary_speed_scale=cfg.stationary_speed_scale,
    stationary_reward_weight=cfg.stationary_reward_weight,
    target_probabilities=cfg.target_probabilities,
    push_direction_mode=cfg.push_direction_mode,
  )

  if cfg.video:
    env_cfg.viewer.height = cfg.video_height
    env_cfg.viewer.width = cfg.video_width
  render_mode = "rgb_array" if cfg.video else None
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=render_mode)
  wrapped_env = base_env
  if cfg.video:
    video_folder = (
      Path(cfg.video_folder)
      if cfg.video_folder
      else (checkpoint.parent / "videos" / "evaluation")
    )
    wrapped_env = VideoRecorder(
      base_env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
  vec_env = RslRlVecEnvWrapper(wrapped_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )
  policy = runner.get_inference_policy(device=cfg.device)
  obs = vec_env.get_observations()

  samples: dict[str, list[torch.Tensor]] = {}
  completed = 0
  steps = 0
  while completed < cfg.num_episodes and steps < cfg.max_steps:
    with torch.inference_mode():
      actions = policy(obs)
    obs, _, dones, _ = vec_env.step(actions)
    steps += 1
    env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
      continue

    metric_values = _episode_metric_values(base_env, env_ids)
    for name, values in metric_values.items():
      samples.setdefault(name, []).append(values)
    controller = getattr(base_env, "_rail_controller", None)
    if controller is not None and hasattr(controller, "response_condition"):
      response_condition = controller.response_condition[env_ids].detach().cpu()
      expected_by_target = torch.tensor([3, 1, 2, 4])
      expected_condition = expected_by_target[controller.target[env_ids].detach().cpu()]
      samples.setdefault("actual_contact_condition", []).append(
        response_condition.float()
      )
      samples.setdefault("contact_condition_classification_correct", []).append(
        ((response_condition > 0) & (response_condition == expected_condition)).float()
      )
      for peak_name in (
        "signed_waist_response_peak_rad",
        "pelvis_yaw_abs_peak_rad",
      ):
        if peak_name not in metric_values:
          samples.setdefault(peak_name, []).append(
            controller.conditional_metrics[peak_name][env_ids].detach().cpu()
          )
    termination = {
      name: base_env.termination_manager.get_term(name)[env_ids].detach().cpu()
      for name in base_env.termination_manager.active_terms
    }
    for name, values in termination.items():
      samples.setdefault("termination/" + name, []).append(values.float())
    failure_union = (
      termination["nonfoot_ground"]
      | termination["root_height"]
      | termination["torso_tilt"]
    )
    samples.setdefault("failure_nonmechanical_union", []).append(failure_union.float())
    lengths = (
      base_env.episode_length_buf[env_ids].detach().cpu().float() * base_env.step_dt
    )
    samples.setdefault("episode_length_s", []).append(lengths)

    completed += int(env_ids.numel())
    obs_dict, _ = base_env.reset(env_ids=env_ids)
    obs = TensorDict(obs_dict, batch_size=[cfg.num_envs])

  wrapped_env.close()
  if completed < cfg.num_episodes:
    raise RuntimeError(
      f"Only completed {completed}/{cfg.num_episodes} episodes in {steps} steps"
    )

  merged = {
    name: torch.cat(chunks)[: cfg.num_episodes] for name, chunks in samples.items()
  }
  if cfg.shoulder_gate:
    confirmed = merged["contact_coverage"] >= 0.5
    target_contact = confirmed & (
      merged["contact_condition_classification_correct"] >= 0.5
    )
    merged["response_direction_correct_given_target_contact"] = merged[
      "response_direction_correct"
    ][target_contact]
    for name in (
      "signed_waist_response_peak_rad",
      "pelvis_yaw_abs_peak_rad",
    ):
      merged[f"{name}_given_target_contact"] = merged[name][target_contact]
  if not all(torch.isfinite(values).all() for values in merged.values()):
    raise RuntimeError("Evaluation produced NaN/Inf metrics")
  metrics = {name: _summarize(values) for name, values in merged.items()}
  report = {
    "config": asdict(cfg),
    "completed_episodes": cfg.num_episodes,
    "environment_steps": steps,
    "finite": True,
    "metrics": metrics,
    "gate": _gate(cfg.stage, metrics, shoulder_gate=cfg.shoulder_gate),
  }
  output = Path(cfg.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main(tyro.cli(EvalCfg))
