"""Tests for the reduced-radius GPU0 speed-gradient suite."""

from __future__ import annotations

import unittest

from scripts.run_force_unloading_speed_gradient_suite import (
  GPU_ID,
  SPEEDS,
  SPHERE_RADIUS,
  curriculum_command,
  smoke_command,
)


class ForceUnloadingSpeedGradientSuiteTest(unittest.TestCase):
  def test_chest_curriculum_has_fixed_gpu_geometry_and_speed_gradient(self) -> None:
    command = curriculum_command("chest")
    rendered = " ".join(command)
    self.assertEqual(GPU_ID, 0)
    self.assertEqual(SPHERE_RADIUS, 0.10)
    self.assertEqual(SPEEDS, (0.05, 0.10, 0.20))
    self.assertIn("--gpu-id 0", rendered)
    self.assertIn("--sphere-radius 0.1", rendered)
    self.assertIn("--contact-reference-speeds 0.05 0.1 0.2", rendered)
    self.assertIn("--target-probabilities 1.0 0.0 0.0", rendered)
    self.assertIn("pretrained_lafan_low_speed_static80.pt", rendered)
    self.assertIn("--gsi-max-root-planar-speed 0.15", rendered)
    self.assertIn("--stationary-speed-scale 10.0", rendered)

  def test_shoulder_curriculum_is_front_to_back_and_split_evaluated(self) -> None:
    command = curriculum_command("shoulders")
    rendered = " ".join(command)
    self.assertIn("--target-probabilities 0.0 0.5 0.5", rendered)
    self.assertIn("--push-direction-mode front_to_back", rendered)
    self.assertIn("--shoulder-split-evaluation", command)

  def test_contact_smokes_are_bound_to_local_cuda_zero(self) -> None:
    for target in ("chest", "shoulders"):
      rendered = " ".join(smoke_command(target))
      self.assertIn(f"--target {target}", rendered)
      self.assertIn("--device cuda:0", rendered)
      self.assertIn("--sphere-radius 0.1", rendered)
      self.assertIn("--reference-speed 0.2", rendered)


if __name__ == "__main__":
  unittest.main()
