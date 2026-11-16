"""Regression tests for the phase-free G1 shoulder IK-v2 Prior."""

from __future__ import annotations

import unittest

import torch
from scripts.pretrain_shoulder_ik_v2 import FORMAL

from smp.pretrain.conditioning import CONDITION_SCHEMA as FIVE_SCHEMA
from smp.pretrain.schema import resolve_condition, validate_schema
from smp.pretrain.shoulder80 import CONDITION_SCHEMA as OLD_THREE_SCHEMA
from smp.pretrain.shoulder_ik_v2 import (
  CONDITION_SCHEMA,
  ShoulderIkV2Sampler,
)
from smp.pretrain.shoulder_ik_v2_dataset import (
  EXPECTED_COUNTS,
  ShoulderIkV2Dataset,
)


class ShoulderIkV2PriorTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)
    cls.train = ShoulderIkV2Dataset(split="train")

  def test_schema_is_distinct_and_backward_compatible(self):
    self.assertEqual(resolve_condition("null", CONDITION_SCHEMA), 3)
    self.assertEqual(resolve_condition("left_shoulder", CONDITION_SCHEMA), 1)
    self.assertIs(validate_schema(CONDITION_SCHEMA, 3), CONDITION_SCHEMA)
    self.assertIs(validate_schema(OLD_THREE_SCHEMA, 3), OLD_THREE_SCHEMA)
    self.assertIs(validate_schema(FIVE_SCHEMA, 5), FIVE_SCHEMA)
    with self.assertRaises(ValueError):
      resolve_condition("chest", CONDITION_SCHEMA)

  def test_dataset_contract_counts_normalization_and_mirror_pairs(self):
    for split, counts in EXPECTED_COUNTS.items():
      dataset = ShoulderIkV2Dataset(split=split)
      self.assertEqual(dataset.windows.shape[1:], (10, 59))
      self.assertEqual(torch.bincount(dataset.labels, minlength=3).tolist(), counts)
      self.assertTrue(torch.isfinite(dataset.windows).all())
      self.assertFalse(hasattr(dataset, "phase_ids"))
      mirror = dataset.mirror_indices()
      shoulder = dataset.labels != 0
      self.assertTrue(
        torch.equal(mirror[mirror[shoulder]], torch.arange(len(dataset))[shoulder])
      )
      self.assertTrue(
        torch.equal(dataset.labels[mirror[shoulder]], 3 - dataset.labels[shoulder])
      )

  def test_sampler_exact_quota_amplitude_balance_and_resume(self):
    sampler = ShoulderIkV2Sampler(
      self.train.labels,
      self.train.clip_indices,
      self.train.target_amplitudes_deg,
    )
    indices = torch.tensor(list(sampler))
    labels = self.train.labels[indices]
    self.assertEqual(torch.bincount(labels, minlength=3).tolist(), [672, 1344, 1344])
    edges = torch.tensor(tuple(float(value) for value in range(20, 70, 5)))
    for condition in (1, 2):
      bins = torch.bucketize(
        self.train.target_amplitudes_deg[indices[labels == condition]], edges
      )
      counts = torch.bincount(bins, minlength=11)
      self.assertEqual(int(counts.sum()), 1344)
      self.assertLessEqual(int(counts.max() - counts.min()), 1)
    state = sampler.state_dict()
    expected = list(sampler)
    restored = ShoulderIkV2Sampler(
      self.train.labels,
      self.train.clip_indices,
      self.train.target_amplitudes_deg,
    )
    restored.load_state_dict(state)
    self.assertEqual(expected, list(restored))
    self.assertNotEqual(indices.tolist(), expected)

  def test_training_budget_matches_approved_configuration(self):
    self.assertEqual(
      (
        FORMAL["num_epochs"],
        FORMAL["batch_size"],
        FORMAL["d_model"],
        FORMAL["num_layers"],
      ),
      (10000, 1024, 128, 2),
    )
    self.assertEqual(FORMAL["condition_dropout"], 0.1)
    self.assertFalse(FORMAL["ema"])
    self.assertEqual(
      (FORMAL["checkpoint_interval"], FORMAL["save_interval"]), (100, 1000)
    )


if __name__ == "__main__":
  unittest.main()
