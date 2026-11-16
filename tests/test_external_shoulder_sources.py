"""Adapters must not silently relax shoulder80 thresholds or leak actors."""

import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scripts.curate_conditional_shoulder80 import Cfg, classify_windows, data_gate
from scripts.review_external_shoulder_sources import (
  actor_group,
  coverage_gate,
  group_split,
  load_motion,
  resample_motion,
  source_fps,
)


def raw_motion(length=121):
  result = np.zeros((length, 36), dtype=np.float64)
  result[:, 6] = 1
  result[:, 2] = 0.75
  return result


class ExternalSourceTests(unittest.TestCase):
  def test_fps_not_assumed_to_be_lafan(self):
    self.assertEqual(source_fps("g1_moves", "a.csv"), 60)
    self.assertEqual(source_fps("amass_g1", "a_120_jpos.npy"), 120)
    self.assertEqual(source_fps("amass_g1", "a_59_jpos.npy"), 59)
    with self.assertRaises(ValueError):
      source_fps("amass_g1", "a.npy")
    raw = raw_motion()
    raw[:, 0] = np.arange(len(raw)) * 0.12 / 60
    result = resample_motion(raw, 60)
    self.assertEqual(len(result), 100)
    np.testing.assert_allclose(np.gradient(result[:, 0], 1 / 50), 0.12)

  def test_coordinate_offset_is_fixed_and_only_amass(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      raw = raw_motion()
      raw[:, 2] = -0.01
      np.save(path / "motion.npy", raw)
      original, _ = load_motion(path / "motion.npy", "amass_g1", 120)
      np.testing.assert_allclose(original[:, 2], 0.783)
      np.testing.assert_array_equal(original[:, 7:], raw[:, 7:])
      np.savetxt(path / "motion.csv", raw, delimiter=",")
      original, _ = load_motion(path / "motion.csv", "g1_moves", 60)
      np.testing.assert_allclose(original[:, 2], -0.01)
      np.testing.assert_array_equal(np.load(path / "motion.npy"), raw)

  def test_numeric_only_finite_and_quaternion_validation(self):
    raw = raw_motion()
    raw[4, 0] = np.nan
    with self.assertRaisesRegex(ValueError, "NaN"):
      resample_motion(raw, 60)
    raw = raw_motion()
    raw[:, 6] = 2
    with self.assertRaisesRegex(ValueError, "quaternion"):
      resample_motion(raw, 60)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "object.npy"
      np.save(path, {"untrusted": "object"})
      with self.assertRaises(ValueError):
        load_motion(path, "amass_g1", 120)

  def test_accad_actor_shared_across_activity_folders(self):
    a = actor_group("amass_g1", "g1/ACCAD/Male2MartialArtsPunches_c3d/a_120_jpos.npy")
    b = actor_group("amass_g1", "g1/ACCAD/Male2Walking_c3d/b_120_jpos.npy")
    self.assertEqual(a, b)
    self.assertEqual(group_split(a), group_split(b))
    with self.assertRaisesRegex(ValueError, "Unverified"):
      actor_group("amass_g1", "g1/ACCAD/MartialArtsWalksTurns_c3d/a_120_jpos.npy")

  def test_g1_clips_same_performer_do_not_cross_split(self):
    groups = [
      actor_group("g1_moves", f"karate/M_Move{i}/retarget/M_Move{i}.csv")
      for i in (3, 5, 19)
    ]
    self.assertEqual(len(set(groups)), 1)
    self.assertEqual(len({group_split(g) for g in groups}), 1)
    with self.assertRaisesRegex(ValueError, "Unverified"):
      actor_group("g1_moves", "bonus/V_Rocamena/retarget/V_Rocamena.csv")

  def test_threshold_defaults_not_changed(self):
    cfg = asdict(Cfg())
    for key, expected in {
      "fps": 50,
      "window_size": 10,
      "stride": 5,
      "early_min_delta": 0.01,
      "entry_min_yaw": 0.08,
      "entry_min_delta": 0.02,
      "hold_min_last": 0.12,
      "hold_min_window": 0.10,
      "hold_max_delta": 0.01,
      "max_yaw": 0.35,
      "max_shoulder_speed": 0.20,
      "context_max_displacement": 0.10,
      "context_max_path": 0.15,
      "early_link_seconds": 1.0,
      "stand_max_yaw_delta": 0.005,
    }.items():
      self.assertEqual(cfg[key], expected)
    raw = raw_motion(300)
    raw[:, 19] = 0.15
    raw[:, 0] = np.arange(len(raw)) * 0.12 / 50
    self.assertFalse((classify_windows(raw, Cfg())["labels"] >= 0).any())

  def lineage(self):
    rows = []
    for split, subjects in (("train", (1, 2)), ("validation", (4,)), ("test", (5,))):
      for subject in subjects:
        for c in range(3):
          for p in (0,) if c == 0 else (1, 2, 3):
            rows.append(
              {
                "dataset": "lafan",
                "source": f"dance_subject{subject}.csv",
                "source_id": f"lafan/dance_subject{subject}.csv",
                "source_sha256": f"hash{subject}",
                "source_group": f"lafan:subject:{subject}",
                "split": split,
                "condition_id": c,
                "phase_id": p,
                "frame50_start": c * 100 + p * 20,
                "frame50_end_exclusive": c * 100 + p * 20 + 10,
              }
            )
    return rows

  def test_coverage_gates_match_historical_implementation(self):
    rows = self.lineage()
    baseline = {s: [] for s in ("train", "validation", "test")}
    self.assertEqual(
      coverage_gate(rows, baseline)["checks"], data_gate(rows, baseline)["checks"]
    )
    missing = [r for r in rows if not (r["split"] == "test" and r["phase_id"] == 1)]
    result = coverage_gate(missing, baseline)
    self.assertFalse(result["passed"])
    self.assertEqual(len(result["failed_checks"]), 2)

  def test_mirrors_are_not_new_independent_samples(self):
    rows = self.lineage()
    baseline = {s: [] for s in ("train", "validation", "test")}
    first = coverage_gate(rows, baseline)["baseline_comparison"]
    self.assertEqual(
      coverage_gate(rows + copy.deepcopy(rows), baseline)["baseline_comparison"], first
    )

  def test_identical_source_cannot_cross_splits(self):
    rows = self.lineage()
    rows[-1]["source_sha256"] = "hash1"
    with self.assertRaisesRegex(ValueError, "cross splits"):
      coverage_gate(rows, {s: [] for s in ("train", "validation", "test")})


if __name__ == "__main__":
  unittest.main()
