"""Scratch/resume and immutable-input checks for the exploratory EXP-037."""

from pathlib import Path

import pytest
from scripts import run_shoulder_mixed_p01p02_ppo_override as driver
from scripts.train_force_unloading_curriculum import (
  CONTACT_STAGES,
  STAGE_A,
  ResumeSource,
  build_train_command,
)

from smp.pretrain.conditioning import sha256


def argument(command, name):
  return command[command.index(name) + 1]


def test_a_starts_scratch_while_smoke_stays_separate():
  cfg = driver.curriculum_cfg()
  formal = build_train_command(cfg, STAGE_A, "a", None)
  smoke = build_train_command(cfg, STAGE_A, "smoke", None, smoke=True)
  for command in (formal, smoke):
    assert argument(command, "--agent.resume") == "False"
    assert "--agent.load-run" not in command
    assert "--agent.load-checkpoint" not in command
    assert argument(command, "--agent.upload-model") == "False"
    assert argument(command, "--env.events.init-smp-state.params.ckpt-path") == str(
      driver.PRIOR
    )
    assert "forward" not in " ".join(command).lower()
  assert argument(formal, "--agent.max-iterations") == "3000"
  assert argument(formal, "--env.scene.num-envs") == "4096"
  assert argument(formal, "--agent.save-interval") == "1000"
  assert argument(smoke, "--agent.max-iterations") == "2"
  assert argument(smoke, "--env.scene.num-envs") == "64"
  assert argument(smoke, "--agent.logger") == "tensorboard"
  assert argument(formal, "--agent.logger") == "wandb"


def test_b2_uses_only_its_explicit_b1_parent():
  cfg = driver.curriculum_cfg()
  run = Path(cfg.log_root) / cfg.experiment_name / "new_experiment_b1"
  parent = ResumeSource(run, run / "model_6999.pt")
  command = build_train_command(cfg, CONTACT_STAGES[1], "b2", parent)
  assert argument(command, "--agent.resume") == "True"
  assert argument(command, "--agent.load-run") == parent.run_dir.name
  assert argument(command, "--agent.load-checkpoint") == parent.checkpoint.name
  assert argument(command, "--agent.max-iterations") == "4000"
  assert (
    argument(
      command, "--env.events.rail-pusher-controller.params.rail-cfg.reference-speed"
    )
    == "0.03"
  )
  assert cfg.target_probabilities == (0.0, 0.5, 0.5, 0.0)
  assert cfg.gsi_max_root_planar_speed == 0.15
  assert cfg.stationary_speed_scale == 10.0


def test_prior_and_calibration_are_copied_without_recalibration(tmp_path, monkeypatch):
  if not driver.PRIOR.exists():
    pytest.skip("EXP-036 artifacts not present")
  isolated = tmp_path / "runtime.pt"
  monkeypatch.setattr(driver, "CALIBRATION", isolated)
  result = driver.validate_sources()
  assert sha256(isolated) == driver.CALIBRATION_SHA
  assert isolated.read_bytes() == driver.SOURCE_CALIBRATION.read_bytes()
  assert set(result["failed_checks"]) == {
    "left_shoulder/correct_gt_stand_ge_0.95",
    "right_shoulder/correct_gt_stand_ge_0.95",
  }
  # Once prepared, a damaged runtime artifact must stop the driver.
  isolated.write_bytes(b"corrupted")
  with pytest.raises(ValueError, match="runtime calibration changed"):
    driver.validate_sources()


def test_changed_prior_is_rejected_before_loading(tmp_path, monkeypatch):
  path = tmp_path / "wrong_prior.pt"
  path.write_bytes(b"wrong candidate")
  monkeypatch.setattr(driver, "PRIOR", path)
  with pytest.raises(ValueError, match="Frozen source identity mismatch"):
    driver.validate_sources()
