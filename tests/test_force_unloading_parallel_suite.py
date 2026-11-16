"""Unit tests for the dual-GPU force-unloading suite scheduler."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_gsi_static80_shoulder_suite import (
  _curriculum_command,
  choose_experiment3_gpu,
  terminal_stage_a,
  validate_gpu_ids,
)


class ForceUnloadingParallelSuiteTest(unittest.TestCase):
  def test_two_distinct_physical_gpus_are_required(self) -> None:
    inventory = {0: {"uuid": "gpu-0"}, 1: {"uuid": "gpu-1"}}
    validate_gpu_ids((0, 1), inventory)
    with self.assertRaises(ValueError):
      validate_gpu_ids((0, 0), inventory)
    with self.assertRaises(RuntimeError):
      validate_gpu_ids((0, 2), inventory)

  def test_parallel_curricula_receive_independent_gpu_ids(self) -> None:
    common = {
      "prior": "prior.pt",
      "prior_sha": "abc",
      "prefix": "test",
      "state_dir": "state",
    }
    command_0 = _curriculum_command(**common, gpu_id=0)
    command_1 = _curriculum_command(**common, gpu_id=1)
    gpu_flag = command_0.index("--gpu-id")
    self.assertEqual(command_0[gpu_flag + 1], "0")
    self.assertEqual(command_1[gpu_flag + 1], "1")

  def test_experiment_3_waits_for_actual_terminal_stage_a(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      manifest = Path(directory) / "curriculum_state.json"
      manifest.write_text(
        json.dumps(
          {
            "stages": {
              "a": {"status": "complete", "gate_passed": False},
              "a_extension": {"status": "running"},
            }
          }
        ),
        encoding="utf-8",
      )
      self.assertIsNone(terminal_stage_a(manifest))
      payload = json.loads(manifest.read_text(encoding="utf-8"))
      payload["stages"]["a_extension"]["status"] = "complete"
      manifest.write_text(json.dumps(payload), encoding="utf-8")
      self.assertEqual(terminal_stage_a(manifest), "a_extension")

  def test_passed_a_is_terminal_without_extension(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      manifest = Path(directory) / "curriculum_state.json"
      manifest.write_text(
        json.dumps(
          {"stages": {"a": {"status": "complete", "gate_passed": True}}}
        ),
        encoding="utf-8",
      )
      self.assertEqual(terminal_stage_a(manifest), "a")

  def test_first_free_gpu_is_selected_only_after_parent_ready(self) -> None:
    self.assertIsNone(choose_experiment3_gpu([0], parent_ready=False))
    self.assertEqual(choose_experiment3_gpu([1, 0], parent_ready=True), 1)
    self.assertIsNone(choose_experiment3_gpu([], parent_ready=True))


if __name__ == "__main__":
  unittest.main()
