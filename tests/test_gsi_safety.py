"""Deterministic unit tests for GSI safety rejection."""

from __future__ import annotations

import unittest

import torch

from smp.rl.events import _root_planar_speed_mask, _window_safety_masks
from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)


class GsiSafetyTest(unittest.TestCase):
  def test_last_frame_joint_limits_margin_and_nonfinite(self) -> None:
    windows = torch.zeros((4, 10, 59))
    limits = torch.stack((torch.full((29,), -1.0), torch.full((29,), 1.0)), dim=-1)
    windows[1, -1, 9] = -1.01
    windows[2, -1, 9] = 0.99995
    windows[3, 0, 0] = float("nan")

    finite, within = _window_safety_masks(windows, limits, joint_limit_margin=1.0e-4)

    self.assertEqual(finite.tolist(), [True, True, True, False])
    self.assertEqual(within.tolist(), [True, False, False, True])

  def test_per_environment_limits_use_intersection(self) -> None:
    windows = torch.zeros((2, 10, 59))
    windows[1, -1, 9] = 0.75
    limits = torch.zeros((2, 29, 2))
    limits[0, :, 0] = -1.0
    limits[0, :, 1] = 1.0
    limits[1, :, 0] = -0.5
    limits[1, :, 1] = 0.5

    _, within = _window_safety_masks(windows, limits, joint_limit_margin=0.0)

    self.assertEqual(within.tolist(), [True, False])

  def test_root_planar_speed_mask(self) -> None:
    windows = torch.zeros((3, 10, 59))
    windows[1, -1, -6] = 1.19
    windows[2, -1, -6] = 1.21

    mask = _root_planar_speed_mask(windows, max_root_planar_speed=1.2)

    self.assertEqual(mask.tolist(), [True, True, False])
    self.assertTrue(_root_planar_speed_mask(windows, None).all())

  def test_force_unloading_enables_safe_gsi_pool(self) -> None:
    cfg = g1_force_unloading_smp_env_cfg()
    params = cfg.events["init_smp_state"].params
    self.assertTrue(params["gsi_reject_unsafe"])
    self.assertEqual(params["gsi_self_collision_sensor"], "self_collision")
    self.assertGreater(params["gsi_joint_limit_margin"], 0.0)
    self.assertEqual(params["gsi_max_root_planar_speed"], 1.2)
    self.assertEqual(params["gsi_buffer_size"], 1024)
    self.assertEqual(cfg.events["gsi_refresh"].params["num_samples"], 256)


if __name__ == "__main__":
  unittest.main()
