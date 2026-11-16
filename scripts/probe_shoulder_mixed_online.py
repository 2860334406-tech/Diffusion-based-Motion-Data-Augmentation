"""Actual-simulation reference and scratch-PPO reward gates, never formal weights."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from smp.data.shoulder_ik import atomic_json
from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_mixed_dataset import ShoulderMixedDataset
from smp.rl.events import _prime_sim_and_buffer
from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.force_unloading.conditional_shoulder_mixed import (
  ShoulderMixedRunner,
  shoulder_mixed_env_cfg,
)
from smp.utils import seed_everything


def reward_gate(values):
  values = torch.as_tensor(values).cpu().float()
  finite = bool(torch.isfinite(values).all())
  count = len(values)
  median = float(torch.quantile(values, 0.5)) if count and finite else 0.0
  zero = float((values == 0).float().mean()) if count else 1.0
  checks = {
    "at_least_1024_supported_nonterminal_windows": count >= 1024,
    "finite": finite,
    "smp_median_ge_0.01": median >= 0.01,
    "exact_zero_fraction_le_0.01": zero <= 0.01,
  }
  return {
    "samples": count,
    "median": median,
    "exact_zero_fraction": zero,
    "checks": checks,
    "passed": all(checks.values()),
  }


def configuration(checkpoint, calibration):
  cfg = shoulder_mixed_env_cfg(
    prior_path=str(checkpoint), calibration_path=str(calibration)
  )
  cfg.scene.num_envs = 64
  cfg.seed = 42
  init = cfg.events["init_smp_state"].params
  init.update(
    compile_model=False,
    gsi_buffer_size=256,
    gsi_batch_size=128,
    gsi_max_root_planar_speed=0.15,
  )
  event = cfg.events["rail_pusher_controller"].params
  event["rail_cfg"] = replace(
    event["rail_cfg"], wait_range_s=(1e9, 1e9), stationary_reward_weight=0.9
  )
  return cfg


def attach_capture(env, destination):
  original = env.step

  def step(actions):
    result = original(actions)
    _, reward, terminated, truncated, _ = result
    supported = env._smp_unexpected_condition < 0.5
    valid = supported & ~terminated & ~truncated
    destination["smp"].append(env._smp_reward[valid].detach().cpu())
    destination["reward"].append(reward.detach().cpu())
    destination["normalized_energy"].append(
      env._smp_normalized_correct_energy.detach().cpu()
    )
    features = env._smp_buffer.compute_features()
    low, high = env._smp_bundle[2:4]
    normalized = 2 * (features - low) / (high - low + 1e-8) - 1
    destination["feature_abs_max"].append(normalized.abs().amax((0, 1)).cpu())
    for name in env.termination_manager.active_terms:
      destination["terminations"].setdefault(name, []).append(
        env.termination_manager.get_term(name).detach().cpu()
      )
    return result

  env.step = step


def capture():
  return {
    "smp": [],
    "reward": [],
    "normalized_energy": [],
    "feature_abs_max": [],
    "terminations": {},
  }


def summary(data):
  gate = reward_gate(torch.cat(data["smp"]))
  energies = torch.cat(data["normalized_energy"])
  rewards = torch.cat(data["reward"])
  feature_max = torch.stack(data["feature_abs_max"]).amax(0)
  if not all(bool(torch.isfinite(v).all()) for v in (energies, rewards, feature_max)):
    raise FloatingPointError("Nonfinite online energy, reward or features")
  gate.update(
    normalized_energy_p95=float(torch.quantile(energies, 0.95)),
    normalized_energy_max=float(energies.max()),
    mean_final_reward=float(rewards.mean()),
    feature_normalized_abs_max=feature_max.tolist(),
    termination_counts={
      k: int(torch.stack(v).sum()) for k, v in data["terminations"].items()
    },
  )
  return gate


@torch.no_grad()
def feature_energy(env):
  model, scheduler, low, high, _, _ = env._smp_bundle
  features = env._smp_buffer.compute_features()
  x = 2 * (features - low) / (high - low + 1e-8) - 1
  generator = torch.Generator(device=env.device).manual_seed(42035)
  noise = torch.randn(x.shape, generator=generator, device=env.device)
  t = torch.full((env.num_envs,), 8, device=env.device, dtype=torch.long)
  prediction = model(scheduler.add_noise(x, noise, t), t, torch.zeros_like(t))
  error = (prediction - noise).square().mean((0, 1))
  error[21] *= 10
  error /= 68
  return {
    "timestep": 8,
    "condition": 0,
    "weighted_feature_energy": error.cpu().tolist(),
    "largest_feature_indices": error.argsort(descending=True)[:10].cpu().tolist(),
  }


def probe(checkpoint, calibration, output):
  output.mkdir(parents=True, exist_ok=True)
  seed_everything(42)
  torch.set_num_threads(2)
  training = ShoulderMixedDataset(split="train")
  collected = capture()
  reference_cfg = configuration(checkpoint, calibration)
  reference_cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg=reference_cfg, device="cuda:0")
  try:
    env.reset()
    ids = torch.arange(64, device="cuda:0")
    chosen = training.sample_indices(0, 64, 42035)
    references = training.raw[chosen].cuda()
    _prime_sim_and_buffer(env, ids, references)
    env.sim.forward()
    env.scene.update(env.step_dt)
    parity = float((env._smp_buffer.compute_features() - references).abs().max())
    if parity > 5e-5:
      raise RuntimeError(f"Offline/online window round-trip mismatch: {parity}")
    attach_capture(env, collected)
    action_term = env.action_manager.get_term("joint_pos")
    target = references[:, -1, 9:38].clone()
    action = (target - action_term.offset) / action_term.scale
    with torch.inference_mode():
      for step in range(48):
        _, _, terminated, truncated, _ = env.step(action)
        reset_ids = (terminated | truncated).nonzero().flatten()
        if len(reset_ids):
          env.reset(env_ids=reset_ids)
          new_ids = training.sample_indices(0, len(reset_ids), 42035 + step + 1)
          new_reference = training.raw[new_ids].cuda()
          _prime_sim_and_buffer(env, reset_ids, new_reference)
          env.sim.forward()
          env.scene.update(env.step_dt)
          target[reset_ids] = new_reference[:, -1, 9:38]
          action = (target - action_term.offset) / action_term.scale
    reference_report = summary(collected)
    reference_report["feature_energy_diagnostic"] = feature_energy(env)
    reference_report["feature_roundtrip_max_error"] = parity
  finally:
    env.close()

  seed_everything(42)
  collected = capture()
  env = ManagerBasedRlEnv(cfg=configuration(checkpoint, calibration), device="cuda:0")
  try:
    attach_capture(env, collected)
    agent = unitree_g1_smp_ppo_runner_cfg()
    agent.actor.distribution_cfg["init_std"] = 0.1
    agent.algorithm.learning_rate = 1e-4
    agent.upload_model = False
    agent.logger = "tensorboard"
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = ShoulderMixedRunner(
      wrapper, asdict(agent), log_dir=str(output / "scratch_smoke"), device="cuda:0"
    )
    # The blind-test caller is no_grad; PPO's optimizer still needs gradients.
    with torch.enable_grad():
      runner.learn(2)
    scratch = output / "scratch_smoke/model_1.pt"
    state = torch.load(scratch, map_location="cpu", weights_only=True)
    from scripts.train_force_unloading_curriculum import validate_checkpoint

    validate_checkpoint(scratch)
    dims = (
      state["actor_state_dict"]["mlp.0.weight"].shape[1],
      state["critic_state_dict"]["mlp.0.weight"].shape[1],
    )
    if dims != (128, 992):
      raise RuntimeError(f"Unexpected actor/critic input dimensions: {dims}")
    scratch_report = summary(collected)
    scratch_report["feature_energy_diagnostic"] = feature_energy(env)
    scratch_report.update(
      resume=False,
      actor_input=128,
      critic_input=992,
      optimizer_updates=2,
      checkpoint=str(scratch),
      checkpoint_sha256=sha256(scratch),
    )
  finally:
    env.close()
  report = {
    "experiment": "EXP-035",
    "prior_sha256": sha256(checkpoint),
    "calibration_sha256": sha256(calibration),
    "reference": reference_report,
    "scratch_ppo": scratch_report,
    "formal_policy_uses_smoke_weights": False,
    "gate": {"passed": reference_report["passed"] and scratch_report["passed"]},
  }
  atomic_json(output / "report.json", report)
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--calibration", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = probe(args.checkpoint, args.calibration, args.output)
  if not report["gate"]["passed"]:
    raise SystemExit(2)
