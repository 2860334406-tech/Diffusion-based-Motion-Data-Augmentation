"""CPU regression tests for the independently gated shoulder80 data stage."""

import copy
import unittest

import numpy as np
import torch
from scripts.curate_conditional_shoulder80 import (
  Cfg,
  classify_windows,
  context_motion,
  data_gate,
  link_early_windows,
  nonoverlap_count,
)
from scripts.curate_lafan_low_speed import mirror_g1_motion

from smp.pretrain.conditioning import CONDITION_SCHEMA as OLD_SCHEMA
from smp.pretrain.conditioning import condition_index as old_condition_index
from smp.pretrain.shoulder80 import (
  CONDITION_SCHEMA,
  Shoulder80Sampler,
  condition_index,
  subject_split,
)


def motion(length=200):
  result = np.zeros((length, 36))
  result[:, 2], result[:, 6] = 0.75, 1.0
  return result


def complete_lineage():
  records = []
  for split, subjects in (("train", (1, 2)), ("validation", (4,)), ("test", (5,))):
    for subject in subjects:
      for condition in range(3):
        for phase in (0,) if condition == 0 else (1, 2, 3):
          start = condition * 100 + phase * 20
          records.append(
            {
              "source": f"dance1_subject{subject}.csv",
              "split": split,
              "condition_id": condition,
              "phase_id": phase,
              "frame50_start": start,
              "frame50_end_exclusive": start + 10,
              "mirrored": False,
            }
          )
  return records


class Shoulder80Tests(unittest.TestCase):
  def pools(self):
    return torch.tensor([0, 1, 1, 1, 2, 2, 2]).repeat_interleave(3), torch.tensor(
      [0, 1, 2, 3, 1, 2, 3]
    ).repeat_interleave(3)

  def test_exact_sampling_and_resume(self):
    labels, phases = self.pools()
    sampler = Shoulder80Sampler(labels, phases)
    indices = torch.tensor(list(sampler))
    self.assertEqual(torch.bincount(labels[indices]).tolist(), [672, 1344, 1344])
    for condition in (1, 2):
      self.assertEqual(
        torch.bincount(
          phases[indices[labels[indices] == condition]], minlength=4
        ).tolist(),
        [0, 336, 336, 672],
      )
    state = sampler.state_dict()
    expected = list(sampler)
    restored = Shoulder80Sampler(labels, phases)
    restored.load_state_dict(state)
    self.assertEqual(expected, list(restored))
    self.assertNotEqual(indices.tolist(), expected)

  def test_sampler_rejects_missing_pools_and_legacy_labels(self):
    labels, phases = self.pools()
    keep = ~((labels == 1) & (phases == 1))
    with self.assertRaisesRegex(ValueError, "Missing sampling pool"):
      Shoulder80Sampler(labels[keep], phases[keep])
    with self.assertRaisesRegex(ValueError, "NULL"):
      Shoulder80Sampler(
        torch.cat((labels, torch.tensor([3]))), torch.cat((phases, torch.tensor([0])))
      )
    with self.assertRaisesRegex(ValueError, "identity"):
      Shoulder80Sampler(labels, phases).load_state_dict(
        {"seed": 43, "samples": 3360, "epoch": 1}
      )

  def test_new_null_and_old_chest_stay_distinct(self):
    self.assertEqual(condition_index("stand"), 0)
    self.assertEqual(condition_index("null"), 3)
    self.assertEqual(
      CONDITION_SCHEMA["names"], ["stand", "left_shoulder", "right_shoulder"]
    )
    self.assertEqual(old_condition_index("chest"), 3)
    self.assertEqual(old_condition_index("null"), 5)
    self.assertEqual(OLD_SCHEMA["version"], 1)
    for value in ("chest", "back", 4, 5, -1):
      with self.assertRaises(ValueError):
        condition_index(value)

  def test_three_way_subject_split_and_mirror(self):
    for subject, expected in (
      (1, "train"),
      (2, "train"),
      (3, "train"),
      (4, "validation"),
      (5, "test"),
    ):
      for suffix in ("", "_mirror"):
        self.assertEqual(
          subject_split(f"dance2_subject{subject}{suffix}.csv"), expected
        )
    with self.assertRaises(ValueError):
      subject_split("dance2_subject50.csv")

  def test_phase_definitions_and_mirror_mapping(self):
    source = motion()
    source[40:90, 19] = np.linspace(0, 0.20, 50)
    source[90:, 19] = 0.20
    left = classify_windows(source, Cfg())
    right = classify_windows(mirror_g1_motion(source), Cfg())
    self.assertEqual(set(left["phases"][left["labels"] == 1]), {1, 2, 3})
    np.testing.assert_array_equal(
      right["labels"], np.where(left["labels"] == 1, 2, left["labels"])
    )
    np.testing.assert_array_equal(right["phases"], left["phases"])
    np.testing.assert_allclose(left["metrics"][:, 1], -right["metrics"][:, 1])

  def test_static_rejects_turning_and_large_waist(self):
    source = motion()
    source[:, 19] = np.arange(len(source)) * 0.0008
    result = classify_windows(source, Cfg())
    self.assertFalse((result["labels"] == 0).any())
    source[:, 19] = 0.36
    self.assertTrue((classify_windows(source, Cfg())["labels"] == -1).all())

  def test_exact_one_second_context_and_boundaries(self):
    positions = motion()[:, :3]
    starts = np.array([0, 20, 25, 50, 170, 175, 190])
    keep, available, displacement, path = context_motion(positions, starts, Cfg())
    np.testing.assert_array_equal(
      available, [False, False, True, True, True, False, False]
    )
    np.testing.assert_array_equal(keep, available)
    self.assertTrue((displacement == 0).all() and (path == 0).all())
    positions[:, 0] = np.arange(len(positions)) * 0.12 / 50
    keep, _, displacement, _ = context_motion(positions, np.array([50]), Cfg())
    self.assertAlmostEqual(displacement[0], 0.12)
    self.assertFalse(keep[0])

  def test_path_rejects_back_and_forth_despite_small_net_motion(self):
    positions = np.zeros((200, 3))
    positions[:, 0] = 0.05 * np.sin(2 * np.pi * np.arange(200) / 50)
    keep, _, displacement, path = context_motion(positions, np.array([50]), Cfg())
    self.assertLess(displacement[0], 1e-10)
    self.assertGreater(path[0], 0.15)
    self.assertFalse(keep[0])

  def test_sustained_translation_is_not_admitted_as_shoulder(self):
    source = motion()
    source[:, 0] = np.arange(len(source)) * 0.12 / 50
    source[:, 19] = 0.15
    result = classify_windows(source, Cfg())
    self.assertGreater(result["audit"]["pre_context"][1][3], 0)
    self.assertFalse((result["labels"] >= 0).any())

  def test_early_entry_requires_safe_future_mature_window(self):
    starts = np.array([10, 40, 70])
    labels, phases = np.ones(3, dtype=int), np.array([1, 2, 3])
    waist = np.linspace(0, 0.2, 100)
    keep, links = link_early_windows(
      starts, labels, phases, np.ones(3, dtype=bool), waist, Cfg()
    )
    self.assertTrue(keep[0])
    self.assertEqual(links[0], 40)
    keep, _ = link_early_windows(
      starts, labels, phases, np.array([True, False, True]), waist, Cfg()
    )
    self.assertFalse(keep[0])
    waist[30] = -0.01
    keep, _ = link_early_windows(
      starts, labels, phases, np.ones(3, dtype=bool), waist, Cfg()
    )
    self.assertFalse(keep[0])

  def test_mirrors_and_overlap_are_not_independent(self):
    spans = [
      ("a.csv", 0, 10),
      ("a.csv", 0, 10),
      ("a.csv", 5, 15),
      ("a.csv", 10, 20),
      ("b.csv", 0, 10),
    ]
    self.assertEqual(nonoverlap_count(spans), 3)

  def test_data_gate_requires_diversity_and_novel_nonoverlap(self):
    rows = complete_lineage()
    baseline = {"train": [("dance1_subject1.csv", 0, 10)], "validation": [], "test": []}
    gate = data_gate(rows, baseline)
    self.assertTrue(gate["passed"])
    removed = [
      r
      for r in rows
      if not (
        r["split"] == "train" and r["phase_id"] == 1 and "subject2" in r["source"]
      )
    ]
    mirrored = [{**r, "mirrored": True} for r in removed]
    gate = data_gate(removed + mirrored, baseline)
    self.assertFalse(gate["passed"])
    self.assertIn(
      "train/left_shoulder/early_entry/at_least_two_original_files",
      gate["failed_checks"],
    )
    same = {
      split: [
        (r["source"], r["frame50_start"], r["frame50_end_exclusive"])
        for r in rows
        if r["split"] == split and r["condition_id"]
      ]
      for split in baseline
    }
    self.assertFalse(data_gate(rows, same)["passed"])

  def test_data_gate_rejects_subject_leakage(self):
    rows = copy.deepcopy(complete_lineage())
    rows[0]["source"] = "dance1_subject5.csv"
    with self.assertRaisesRegex(ValueError, "leakage"):
      data_gate(rows, {"train": [], "validation": [], "test": []})


if __name__ == "__main__":
  unittest.main()
