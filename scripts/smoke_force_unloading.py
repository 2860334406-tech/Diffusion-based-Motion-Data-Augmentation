"""Headless physical smoke test for ``Smp-Force-Unloading-G1``.

This intentionally removes the diffusion prior and PPO reward so mechanical,
contact, observation, and safety issues can be diagnosed independently.
"""

from __future__ import annotations

import argparse
import json

import torch
from mjlab.envs import ManagerBasedRlEnv

from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)
from smp.rl.tasks.force_unloading.mdp import RailPusherCfg


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=6)
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  rail_cfg = RailPusherCfg(
    wait_range_s=(0.0, 0.0),
    target_probabilities=(1.0 / 3.0,) * 3,
  )
  cfg = g1_force_unloading_smp_env_cfg(rail_cfg=rail_cfg)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  for name in ("init_smp_state", "gsi_reset", "gsi_refresh"):
    cfg.events.pop(name, None)
  cfg.rewards = {}

  env = ManagerBasedRlEnv(cfg, device=args.device)
  observations, _ = env.reset()
  actions = torch.zeros(
    (args.num_envs, env.action_manager.total_action_dim), device=args.device
  )
  controller = env._rail_controller  # type: ignore[attr-defined]
  targets_seen: set[int] = set()
  max_contact_force = 0.0
  min_contact_velocity = float("inf")
  max_cross_axis = 0.0
  finite = True
  for _ in range(args.steps):
    observations, _, _, _, _ = env.step(actions)
    targets_seen.update(controller.target.tolist())
    max_cross_axis = max(
      max_cross_axis, float(controller.rail_cross_axis_error.max().item())
    )
    max_contact_force = max(
      max_contact_force, float(controller.raw_force_peak.max().item())
    )
    contact = controller.contact_obs[:, 0].bool()
    if contact.any():
      velocity = controller.pusher.data.joint_vel[:, 0]
      min_contact_velocity = min(
        min_contact_velocity, float(velocity[contact].min().item())
      )
    finite &= bool(
      torch.isfinite(observations["actor"]).all()
      and torch.isfinite(observations["critic"]).all()
    )

  checkpoint = torch.load(
    "datasets/pretrain_ckpt/pretrained_loco.pt",
    map_location="cpu",
    weights_only=False,
  )
  report = {
    "actor_shape": list(observations["actor"].shape),
    "critic_shape": list(observations["critic"].shape),
    "smp_feature_dim": checkpoint["cfg"]["feature_dim"],
    "smp_window_size": checkpoint["cfg"]["window_size"],
    "targets_seen": sorted(targets_seen),
    "max_cross_axis_error_m": max_cross_axis,
    "max_contact_force_n": max_contact_force,
    "min_pusher_velocity_during_contact_m_s": (
      None if min_contact_velocity == float("inf") else min_contact_velocity
    ),
    "finite": finite,
  }
  print(json.dumps(report, indent=2))
  env.close()

  assert report["actor_shape"] == [args.num_envs, 128]
  assert report["critic_shape"] == [args.num_envs, 992]
  assert report["smp_feature_dim"] == 59
  assert report["smp_window_size"] == 10
  assert max_cross_axis < 1e-5
  assert finite


if __name__ == "__main__":
  main()
