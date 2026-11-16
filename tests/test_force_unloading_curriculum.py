"""Unit tests for the low-speed-prior force-unloading curriculum driver."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.train_force_unloading_curriculum import (
  CONTACT_STAGES,
  EXPECTED_PRIOR_SHA256,
  STAGE_A,
  STAGE_A_EXTENSION,
  CurriculumCfg,
  ResumeSource,
  build_evaluation_command,
  build_train_command,
  contact_stages_for_cfg,
  curriculum_stages,
  expected_terminal_iteration,
  is_recoverable_interruption,
  remaining_iterations,
  should_extend_stage_a,
)


class ForceUnloadingCurriculumTest(unittest.TestCase):
  def setUp(self) -> None:
    self.cfg = CurriculumCfg()

  def test_stage_table_matches_approved_budget_and_contact_schedule(self) -> None:
    stages = curriculum_stages()
    self.assertEqual(
      [(stage.key, stage.iterations) for stage in stages],
      [
        ("a", 3000),
        ("a_extension", 2000),
        ("b1", 4000),
        ("b2", 4000),
        ("c", 19000),
      ],
    )
    self.assertEqual(
      sum(stage.iterations for stage in (STAGE_A, *CONTACT_STAGES)), 30000
    )
    self.assertEqual(sum(stage.iterations for stage in stages), 32000)
    self.assertEqual(STAGE_A.wait_range_s, (1.0e9, 1.0e9))
    self.assertEqual(STAGE_A.stationary_reward_weight, 0.9)
    self.assertEqual(
      [
        (
          stage.reference_speed,
          stage.max_force_weight_ratio,
          stage.ramp_time,
        )
        for stage in CONTACT_STAGES
      ],
      [(0.01, 0.05, 1.0), (0.03, 0.10, 0.5), (0.05, 0.15, 0.25)],
    )
    self.assertEqual(
      self.cfg.prior_sha256,
      EXPECTED_PRIOR_SHA256,
    )

  def test_speed_gradient_and_sphere_radius_are_explicit_overrides(self) -> None:
    cfg = CurriculumCfg(
      sphere_radius=0.10,
      contact_reference_speeds=(0.05, 0.10, 0.20),
    )
    stages = contact_stages_for_cfg(cfg)
    self.assertEqual([stage.reference_speed for stage in stages], [0.05, 0.10, 0.20])
    train = " ".join(build_train_command(cfg, stages[0], "gradient", None))
    evaluate = " ".join(
      build_evaluation_command(
        cfg, stages[-1], Path("/tmp/model.pt"), Path("/tmp/eval.json")
      )
    )
    self.assertIn("sphere-radius 0.1", train)
    self.assertIn("--sphere-radius 0.1", evaluate)

  def test_stage_a_command_is_scratch_and_uses_low_speed_prior(self) -> None:
    command = build_train_command(
      self.cfg,
      STAGE_A,
      "scratch_test",
      resume=None,
    )
    rendered = " ".join(command)

    self.assertIn("--agent.resume False", rendered)
    self.assertNotIn("--agent.load-run", command)
    self.assertNotIn("--agent.load-checkpoint", command)
    self.assertIn(self.cfg.prior_checkpoint, command)
    self.assertIn("(1e+09,1e+09)", command)
    self.assertIn(
      "--env.events.rail-pusher-controller.params.rail-cfg.stationary-reward-weight",
      command,
    )
    self.assertIn("0.9", command)

  def test_later_stage_resumes_immediately_preceding_checkpoint(self) -> None:
    source = ResumeSource(
      run_dir=Path("/tmp/previous_run"),
      checkpoint=Path("/tmp/previous_run/model_2999.pt"),
    )
    command = build_train_command(
      self.cfg,
      CONTACT_STAGES[0],
      "b1_test",
      resume=source,
    )
    rendered = " ".join(command)

    self.assertIn("--agent.resume True", rendered)
    self.assertIn("--agent.load-run previous_run", rendered)
    self.assertIn("--agent.load-checkpoint model_2999.pt", rendered)

  def test_a_gate_controls_only_the_optional_extension(self) -> None:
    self.assertTrue(should_extend_stage_a({"gate": {"passed": False}}))
    self.assertFalse(should_extend_stage_a({"gate": {"passed": True}}))
    self.assertEqual(STAGE_A_EXTENSION.iterations, 2000)

  def test_interrupted_resume_preserves_original_terminal_iteration(self) -> None:
    self.assertEqual(expected_terminal_iteration(CONTACT_STAGES[0], 4998), 8997)
    self.assertEqual(remaining_iterations(8997, 8000), 998)
    self.assertTrue(
      is_recoverable_interruption(
        {"status": "failed", "error": "[Errno 32] Broken pipe"}
      )
    )
    self.assertTrue(
      is_recoverable_interruption(
        {"status": "failed", "error": "Command failed with exit code -2"}
      )
    )
    self.assertTrue(
      is_recoverable_interruption(
        {"status": "failed", "verified_recoverable_interruption": True}
      )
    )
    self.assertEqual(remaining_iterations(12996, 11000), 1997)
    self.assertFalse(
      is_recoverable_interruption(
        {"status": "failed", "error": "Checkpoint contains NaN/Inf"}
      )
    )

  def test_final_evaluation_uses_nominal_mechanics_and_video(self) -> None:
    stage_c = CONTACT_STAGES[-1]
    command = build_evaluation_command(
      self.cfg,
      stage_c,
      Path("/tmp/model.pt"),
      Path("/tmp/result.json"),
      video_folder=Path("/tmp/video"),
    )
    rendered = " ".join(command)

    self.assertIn("--reference-speed 0.05", rendered)
    self.assertIn("--max-force-weight-ratio 0.15", rendered)
    self.assertIn("--ramp-time 0.25", rendered)
    self.assertIn("--video", command)
    self.assertIn("--video-length 600", rendered)


if __name__ == "__main__":
  unittest.main()
