"""Fast deterministic checks for the force-unloading task math and config."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import torch

from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)
from smp.rl.tasks.force_unloading.mdp import (
  RailContactNoiseCfg,
  RailPusherCfg,
  force_unloading_task_reward,
  mapped_force_drop,
  rail_pusher_entity_cfg,
)


class ForceUnloadingMathTest(unittest.TestCase):
  def test_force_drop_is_mapped_before_product(self) -> None:
    previous = torch.tensor((0.05, 0.05, 0.05, 0.05))
    current = torch.tensor((0.04, 0.05, 0.06, 0.04))
    mask = torch.tensor((1.0, 1.0, 1.0, 0.0))
    reward = mapped_force_drop(previous, current, mask)
    self.assertGreater(float(reward[0]), 0.5)
    self.assertAlmostEqual(float(reward[1]), 0.5, places=6)
    self.assertLess(float(reward[2]), 0.5)
    self.assertEqual(float(reward[3]), 0.0)
    self.assertTrue(torch.all((reward >= 0.0) & (reward <= 1.0)))

  def test_stationary_reward_weight_and_emergency_mask(self) -> None:
    controller = SimpleNamespace(
      robot=SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_w=torch.zeros((2, 3)))
      ),
      cfg=RailPusherCfg(stationary_reward_weight=0.9),
      force_drop=torch.full((2,), 0.5),
      emergency=torch.tensor((False, True)),
      refresh_contact=lambda: None,
    )
    env = SimpleNamespace(_rail_controller=controller)

    reward = force_unloading_task_reward(env)

    self.assertAlmostEqual(float(reward[0]), 0.95, places=6)
    self.assertEqual(float(reward[1]), 0.0)

  def test_stationary_reward_weight_range_is_validated(self) -> None:
    with self.assertRaisesRegex(ValueError, "stationary_reward_weight"):
      RailPusherCfg(stationary_reward_weight=1.01)

  def test_contact_noise_preserves_no_contact_zero(self) -> None:
    obs = torch.zeros((2, 8))
    obs[1] = torch.tensor((1.0, 0.1, 1.0, 0.0, 0.0, 0.2, 0.0, 0.8))
    noisy = RailContactNoiseCfg().apply(obs)
    self.assertTrue(torch.equal(noisy[0], torch.zeros(8)))
    self.assertEqual(float(noisy[1, 0]), 1.0)
    self.assertAlmostEqual(
      float(torch.linalg.vector_norm(noisy[1, 2:5])), 1.0, places=5
    )

  def test_rail_has_exactly_one_slide_dof_and_no_rotation(self) -> None:
    entity = rail_pusher_entity_cfg(RailPusherCfg()).build()
    model = entity.compile()
    self.assertEqual(model.nv, 1)
    self.assertEqual(model.njnt, 1)
    self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_SLIDE)

  def test_observation_history_contract(self) -> None:
    cfg = g1_force_unloading_smp_env_cfg()
    actor = cfg.observations["actor"]
    critic = cfg.observations["critic"]
    self.assertEqual(actor.terms["rail_contact"].history_length, 4)
    self.assertEqual(critic.terms["rail_contact"].history_length, 4)
    self.assertIsNone(critic.history_length)
    self.assertTrue(
      all(
        term.history_length == 10
        for name, term in critic.terms.items()
        if name != "rail_contact"
      )
    )
    self.assertEqual(
      cfg.events["init_smp_state"].params["ckpt_path"],
      "datasets/pretrain_ckpt/pretrained_loco.pt",
    )

  def test_rail_limit_is_telemetry_not_a_failure_termination(self) -> None:
    cfg = g1_force_unloading_smp_env_cfg()
    self.assertNotIn("pusher_travel", cfg.terminations)
    self.assertNotIn("failure_travel", cfg.metrics)
    self.assertIn("travel_limit_reached", cfg.metrics)


if __name__ == "__main__":
  unittest.main()
