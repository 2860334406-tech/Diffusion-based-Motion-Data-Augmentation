"""Tests for GSI-0.15, steep stationary reward, and shoulder-front commands."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from scripts.train_force_unloading_curriculum import (
  CONTACT_STAGES,
  STAGE_A,
  CurriculumCfg,
  build_evaluation_command,
  build_train_command,
)

from smp.rl.tasks.force_unloading.mdp import (
  RailPusherCfg,
  canonical_drive_direction,
  force_unloading_task_reward,
)


class NewForceUnloadingExperimentTest(unittest.TestCase):
  def test_steep_stationary_reward_matches_formula(self) -> None:
    controller = SimpleNamespace(
      robot=SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_w=torch.tensor(((0.15, 0.0, 0.0),)))
      ),
      cfg=RailPusherCfg(
        stationary_reward_weight=1.0,
        stationary_speed_scale=10.0,
      ),
      force_drop=torch.zeros(1),
      emergency=torch.zeros(1, dtype=torch.bool),
      refresh_contact=lambda: None,
    )
    reward = force_unloading_task_reward(SimpleNamespace(_rail_controller=controller))
    self.assertAlmostEqual(float(reward[0]), math.exp(-10.0 * 0.15**2), places=6)
    self.assertEqual(RailPusherCfg().stationary_speed_scale, 1.0)

  def test_front_to_back_shoulders_use_longitudinal_direction(self) -> None:
    targets = torch.tensor((0, 1, 2), dtype=torch.long)
    default = canonical_drive_direction(targets, "target_default")
    front = canonical_drive_direction(targets, "front_to_back")
    torch.testing.assert_close(
      default,
      torch.tensor(((-1.0, 0.0), (0.0, -1.0), (0.0, 1.0))),
    )
    torch.testing.assert_close(
      front,
      torch.tensor(((-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0))),
    )

  def test_train_and_eval_commands_share_new_parameters(self) -> None:
    cfg = CurriculumCfg(
      gsi_max_root_planar_speed=0.15,
      stationary_speed_scale=10.0,
      target_probabilities=(0.0, 0.5, 0.5),
      push_direction_mode="front_to_back",
    )
    train = " ".join(build_train_command(cfg, STAGE_A, "shoulder_test", resume=None))
    self.assertIn("gsi-max-root-planar-speed 0.15", train)
    self.assertIn("stationary-speed-scale 10.0", train)
    self.assertIn("target-probabilities (0,0.5,0.5)", train)
    self.assertIn("push-direction-mode front_to_back", train)

    evaluate = " ".join(
      build_evaluation_command(
        cfg,
        CONTACT_STAGES[0],
        Path("/tmp/model.pt"),
        Path("/tmp/eval.json"),
        target_probabilities=(0.0, 1.0, 0.0),
      )
    )
    self.assertIn("gsi-max-root-planar-speed 0.15", evaluate)
    self.assertIn("stationary-speed-scale 10.0", evaluate)
    self.assertIn("target-probabilities 0 1 0", evaluate)
    self.assertIn("push-direction-mode front_to_back", evaluate)


if __name__ == "__main__":
  unittest.main()
