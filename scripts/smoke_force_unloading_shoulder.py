"""Headless mechanical smoke test for front-to-back left/right shoulder pushes."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv

from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)
from smp.rl.tasks.force_unloading.mdp import RailPusherCfg


@dataclass(frozen=True)
class Cfg:
  num_envs: int = 64
  steps: int = 400
  device: str = "cuda"
  seed: int = 42


def main(cfg: Cfg) -> None:
  rail_cfg = RailPusherCfg(
    wait_range_s=(0.0, 0.0),
    target_probabilities=(0.0, 0.5, 0.5),
    push_direction_mode="front_to_back",
    start_distance=0.10,
  )
  env_cfg = g1_force_unloading_smp_env_cfg(rail_cfg=rail_cfg)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  for name in ("init_smp_state", "gsi_reset", "gsi_refresh"):
    env_cfg.events.pop(name, None)
  env_cfg.rewards = {}

  env_cfg.terminations = {}
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  observations, _ = env.reset()
  actions = torch.zeros(
    (cfg.num_envs, env.action_manager.total_action_dim), device=cfg.device
  )
  controller = env._rail_controller  # type: ignore[attr-defined]
  targets_seen: set[int] = set()
  max_contact_force = 0.0
  min_center_distance = float("inf")
  contact_found = False
  finite = True
  for _ in range(cfg.steps):
    observations, _, _, _, _ = env.step(actions)
    targets_seen.update(controller.target.tolist())
    max_contact_force = max(
      max_contact_force, float(controller.raw_force_peak.max().item())
    )
    target_pos = controller.robot.data.geom_pose_w[
      torch.arange(cfg.num_envs, device=controller.device),
      controller.target_geom_ids[controller.target],
      :3,
    ]
    pusher_pos = controller.pusher.data.geom_pose_w[:, 0, :3]
    distance = torch.linalg.vector_norm(pusher_pos - target_pos, dim=-1)
    min_center_distance = min(min_center_distance, float(distance.min().item()))
    sensor_found = controller.contact_sensor.data.found
    sensor_force = controller.contact_sensor.data.force
    assert sensor_found is not None
    assert sensor_force is not None
    contact_found |= bool(sensor_found.bool().any())
    max_contact_force = max(
      max_contact_force, float(sensor_force[..., 0].abs().max().item())
    )
    finite &= bool(
      torch.isfinite(observations["actor"]).all()
      and torch.isfinite(observations["critic"]).all()
    )

  report = {
    "actor_shape": list(observations["actor"].shape),
    "critic_shape": list(observations["critic"].shape),
    "targets_seen": sorted(targets_seen),
    "max_contact_force_n": max_contact_force,
    "max_cross_axis_error_m": float(controller.rail_cross_axis_error.max().item()),
    "min_center_distance_m": min_center_distance,
    "contact_found": contact_found,
    "finite": finite,
  }
  print(json.dumps(report, indent=2))
  env.close()

  assert report["actor_shape"] == [cfg.num_envs, 128]
  assert report["critic_shape"] == [cfg.num_envs, 992]
  assert targets_seen == {1, 2}
  assert max_contact_force > 0.0
  assert report["max_cross_axis_error_m"] < 1.0e-5
  assert finite


if __name__ == "__main__":
  main(tyro.cli(Cfg))
