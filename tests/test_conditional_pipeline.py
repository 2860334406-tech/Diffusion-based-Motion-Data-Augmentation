"""CPU tests for the conditional experiment orchestration and full PPO state."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from scripts.train_force_unloading_curriculum import (
  CONTACT_STAGES,
  STAGE_A,
  CurriculumCfg,
  _evaluate_all_contacts,
  build_train_command,
  validate_checkpoint,
)

from smp.rl.conditional_normalizer import ConditionalDiffNormalizer
from smp.rl.tasks.force_unloading.conditional import ConditionalSmpRunner
from smp.rl.tasks.force_unloading.mdp.conditions import capsule_start_distance


class ConditionalPipelineTests(unittest.TestCase):
  def test_task_cli_keeps_scratch_and_four_targets(self):
    import mjlab
    import tyro
    from mjlab.scripts.train import TrainConfig
    from scripts.run_conditional_contact_suite import TASK, curriculum_command
    cfg = tyro.cli(CurriculumCfg, args=curriculum_command("unit-test-sha")[3:])
    self.assertEqual(cfg.task_id, TASK)
    self.assertTrue(cfg.all_contact_evaluation)
    self.assertEqual(cfg.gpu_id, 1)
    self.assertEqual(cfg.target_probabilities, (.25,)*4)
    command = build_train_command(cfg, STAGE_A, "unit_smoke", None, smoke=True)
    parsed = tyro.cli(TrainConfig, args=command[3:],
                     default=TrainConfig.from_task(TASK), config=mjlab.TYRO_FLAGS)
    self.assertFalse(parsed.agent.resume)
    self.assertEqual(parsed.env.scene.num_envs, 64)
    self.assertEqual(parsed.agent.max_iterations, 2)
    self.assertFalse(parsed.agent.upload_model)
    rail = parsed.env.events["rail_pusher_controller"].params["rail_cfg"]
    self.assertTrue(rail.conditional)
    self.assertEqual(rail.target_probabilities, (.25,)*4)
    self.assertEqual(rail.start_surface_gap, .02)
    self.assertEqual(rail.wait_range_s, (1e9, 1e9))
    self.assertEqual(parsed.env.events["init_smp_state"].params["gsi_max_root_planar_speed"], .15)
    model = parsed.env.scene.entities["rail_pusher"].spec_fn().compile()
    self.assertAlmostEqual(model.geom("sphere").size[0], .1)

  def test_complete_ppo_and_conditional_state_roundtrip(self):
    normalizer = ConditionalDiffNormalizer(50, "cpu")
    normalizer.update_and_normalize(8, torch.tensor([2., 4.]), torch.tensor([0, 1]))
    original = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in normalizer.state_dict().items()}
    state = {
      "actor_state_dict": {"obs_normalizer.mean": torch.tensor([.5]), "weight": torch.tensor([1.])},
      "critic_state_dict": {"obs_normalizer.mean": torch.tensor([.6]), "weight": torch.tensor([2.])},
      "optimizer_state_dict": {"state": {0: {"exp_avg": torch.tensor([.7])}}},
    }
    class Algorithm:
      def save(self):
        return dict(state)
      def load(self, loaded, load_cfg, strict):
        self.restored = loaded
        return True
    raw = SimpleNamespace(_smp_normalizer=normalizer, _smp_prior_identity={"sha256": "test", "schema": 1},
                          common_step_counter=91)
    runner = ConditionalSmpRunner.__new__(ConditionalSmpRunner)
    runner.env = SimpleNamespace(unwrapped=raw)
    runner.alg, runner.cfg = Algorithm(), {"upload_model": False}
    runner.current_learning_iteration = 17
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model_17.pt"
      runner.save(str(path))
      self.assertEqual(validate_checkpoint(path), 17)
      normalizer.mean.zero_()
      normalizer.count.zero_()
      raw.common_step_counter = 0
      runner.load(str(path), map_location="cpu")
      self.assertEqual(raw.common_step_counter, 91)
      torch.testing.assert_close(normalizer.mean, original["mean"])
      torch.testing.assert_close(normalizer.count, original["count"])
      torch.testing.assert_close(runner.alg.restored["optimizer_state_dict"]["state"][0]["exp_avg"],
                                 state["optimizer_state_dict"]["state"][0]["exp_avg"])
      raw._smp_prior_identity["sha256"] = "changed"
      with self.assertRaises(ValueError):
        runner.load(str(path), map_location="cpu")

  def test_each_contact_gets_independent_evaluation(self):
    calls = []
    def fake_process(command, *_args, **_kwargs):
      calls.append(command)
      output = Path(command[command.index("--output")+1])
      output.write_text(json.dumps({"finite": True, "metrics": {},
                                   "gate": {"passed": False, "checks": {"quality": False}}}))
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      cfg = CurriculumCfg(task_id="Smp-Force-Unloading-Conditional-G1", all_contact_evaluation=True)
      with patch("scripts.train_force_unloading_curriculum._run_process", fake_process):
        _, report, _ = _evaluate_all_contacts(cfg, CONTACT_STAGES[0], root/"policy.pt", root, root)
      self.assertEqual(len(calls), 5)
      self.assertEqual(set(report["conditions"]), {"stand", "left_shoulder", "right_shoulder", "chest", "back"})
      self.assertFalse(report["gate"]["passed"])
      self.assertEqual(calls[0][calls[0].index("--stage")+1], "a")
      for command in calls:
        self.assertEqual(command[command.index("--num-episodes")+1], "1024")

  def test_shoulder_endpoint_gap(self):
    direction = torch.tensor([[-1., 0., 0.]])
    axis = torch.tensor([[.3, 0., (1-.3**2)**.5]])
    half, radius = torch.tensor([.07]), torch.tensor([.04])
    offset = axis*half[:, None]
    distance = capsule_start_distance(direction, axis, half, radius, .10, .02, offset=offset)
    point = offset-distance[:, None]*direction
    projected = (point*axis).sum(-1).clamp(-half, half)
    gap = torch.linalg.vector_norm(point-projected[:, None]*axis, dim=-1)-radius-.10
    torch.testing.assert_close(gap, torch.tensor([.02]), atol=1e-6, rtol=0)


if __name__ == "__main__":
  unittest.main()
