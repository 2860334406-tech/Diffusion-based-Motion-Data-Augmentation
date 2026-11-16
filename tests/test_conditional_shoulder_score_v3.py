"""Unit tests for frozen conditional shoulder score V3."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from scripts.migrate_conditional_score_checkpoint import Cfg as MigrationCfg
from scripts.migrate_conditional_score_checkpoint import migrate

from smp.pretrain.conditioning import CONDITION_SCHEMA, sha256
from smp.rl.conditional_score import (
  FIXED_TIMESTEPS,
  SCORE_MODE,
  SCORE_SCHEMA_VERSION,
  contrastive_score_from_energies,
  opposite_shoulder_conditions,
  paired_condition_predictions,
  score_feature_weights,
  shoulder_score_condition_ids,
  weighted_denoising_mse,
)
from smp.rl.tasks.force_unloading.mdp.conditions import ContactConditionState
from smp.rl.tasks.force_unloading.mdp.controller import RailPusherCfg


class ConditionalShoulderScoreV3Tests(unittest.TestCase):
  def test_mirror_mapping_and_feature_weights(self):
    ids = torch.tensor([0, 1, 2, 3, 4])
    self.assertEqual(opposite_shoulder_conditions(ids).tolist(), [0, 2, 1, 3, 4])
    weights = score_feature_weights()
    self.assertEqual(weights.shape, (59,))
    self.assertEqual(float(weights[21]), 10.0)
    self.assertTrue(torch.equal(weights[torch.arange(59) != 21], torch.ones(58)))

  def test_incidental_non_shoulder_contacts_use_safe_fallback(self):
    mapped, supported = shoulder_score_condition_ids(torch.tensor([0, 1, 2, 3, 4]))
    self.assertEqual(mapped.tolist(), [0, 1, 2, 0, 0])
    self.assertEqual(supported.tolist(), [True, True, True, False, False])

  def test_paired_conditions_receive_identical_xt_and_timestep(self):
    class Recorder(torch.nn.Module):
      def forward(self, x, timestep, condition):
        self.x = x.clone()
        self.timestep = timestep.clone()
        self.condition = condition.clone()
        return torch.zeros_like(x)

    model = Recorder()
    x_t = torch.randn(3, 10, 59)
    timestep = torch.tensor([8, 15, 22])
    correct = torch.tensor([1, 2, 1])
    opposite = torch.tensor([2, 1, 2])
    first, second = paired_condition_predictions(
      model, x_t, timestep, correct, opposite
    )
    torch.testing.assert_close(model.x[:3], model.x[3:], rtol=0, atol=0)
    torch.testing.assert_close(model.timestep[:3], model.timestep[3:], rtol=0, atol=0)
    self.assertEqual(model.condition.tolist(), [1, 2, 1, 2, 1, 2])
    self.assertEqual(first.shape, x_t.shape)
    self.assertEqual(second.shape, x_t.shape)

  def test_time_mask_and_waist_weight(self):
    target = torch.zeros(1, 10, 59)
    prediction = torch.zeros_like(target)
    prediction[:, :-1] = 100.0
    prediction[:, -1, 21] = 1.0
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, -1] = True
    energy = weighted_denoising_mse(prediction, target, score_feature_weights(), mask)
    self.assertAlmostEqual(float(energy), 10.0 / 68.0, places=7)

  def test_score_formula_bounds_and_no_contact_base_only(self):
    correct = torch.tensor([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])
    opposite = torch.tensor([[1.5, 1.5, 1.5], [0.1, 0.1, 0.1]])
    references = torch.ones_like(correct)
    reward, parts = contrastive_score_from_energies(
      correct, opposite, references, torch.tensor([True, False])
    )
    self.assertTrue(bool(((reward >= 0) & (reward <= 1)).all()))
    self.assertGreater(float(parts["contrast_reward"][0]), 0.5)
    self.assertEqual(float(parts["contrast_reward"][1]), 1.0)
    self.assertAlmostEqual(
      float(reward[1]), float(torch.exp(torch.tensor(-1.0))), places=7
    )

  def test_masked_transition_activates_on_third_frame(self):
    forces = torch.tensor([[0.01, 0.0, 0.0, 0.0]])
    state = ContactConditionState(1, "cpu", transition_mode="masked_current")
    state.update(forces)
    state.update(forces)
    self.assertEqual(int(state.reward_ids[0]), 0)
    state.update(forces)
    self.assertEqual(int(state.ids[0]), 1)
    self.assertEqual(int(state.reward_ids[0]), 1)
    self.assertEqual(int(state.reward_mask.sum()), 1)
    legacy = ContactConditionState(1, "cpu")
    for _ in range(3):
      legacy.update(forces)
    self.assertEqual(int(legacy.ids[0]), 1)
    self.assertEqual(int(legacy.reward_ids[0]), 5)
    self.assertEqual(RailPusherCfg().condition_transition_mode, "null_until_stable")

  def test_checkpoint_migration_preserves_all_ppo_tensors(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      prior = root / "prior.pt"
      prior.write_bytes(b"frozen-prior")
      prior_sha = sha256(prior)
      calibration = root / "calibration.pt"
      torch.save(
        {
          "score_schema_version": SCORE_SCHEMA_VERSION,
          "mode": SCORE_MODE,
          "prior_sha256": prior_sha,
          "data_manifest_sha256": "data-sha",
          "fixed_timesteps": list(FIXED_TIMESTEPS),
          "feature_weights": score_feature_weights(),
          "reference_mean": torch.ones(5, 3),
          "samples_per_condition": 4096,
          "seed": 42,
        },
        calibration,
      )
      source = root / "model_2999.pt"
      original = {
        "actor_state_dict": {"w": torch.randn(3, 2)},
        "critic_state_dict": {"w": torch.randn(4, 2)},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(7)}}},
        "iter": 2999,
        "infos": {
          "env_state": {"common_step_counter": 72000},
          "smp_prior": {"sha256": prior_sha, "schema": CONDITION_SCHEMA},
          "smp_normalizer": {"mean": torch.ones(6, 50)},
        },
      }
      torch.save(original, source)
      destination = root / "migrated.pt"
      report = root / "report.json"
      migrate(
        MigrationCfg(
          source=str(source),
          destination=str(destination),
          calibration=str(calibration),
          report=str(report),
          prior=str(prior),
          expected_source_sha256=sha256(source),
          expected_prior_sha256=prior_sha,
        )
      )
      restored = torch.load(destination, map_location="cpu", weights_only=True)
      for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        self.assertEqual(original[key].keys(), restored[key].keys())
      torch.testing.assert_close(
        original["actor_state_dict"]["w"],
        restored["actor_state_dict"]["w"],
        rtol=0,
        atol=0,
      )
      self.assertNotIn("smp_normalizer", restored["infos"])
      self.assertIn("smp_score", restored["infos"])


if __name__ == "__main__":
  unittest.main()
