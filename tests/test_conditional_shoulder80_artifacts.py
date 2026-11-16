"""Validate saved data-review artifacts independently of the curation loop."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts.curate_conditional_shoulder80 import baseline_intervals, data_gate
from scripts.run_conditional_shoulder80_data_review import (
  GPU_UUID,
  require_gpu,
  review_environment,
)

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder80 import CONDITION_SCHEMA, SPLIT_SUBJECTS, subject_split

DATA = Path("logs/reproduction/conditional_shoulder80_prior_v1/data")


class Shoulder80ArtifactTests(unittest.TestCase):
  def dataset(self):
    path = DATA / "manifest.json"
    if not path.is_file():
      self.skipTest("Data review has not completed")
    return json.loads(path.read_text())

  def test_all_shards_and_lineage_are_finite_hash_verified_and_disjoint(self):
    manifest = self.dataset()
    self.assertEqual(manifest["condition_schema"], CONDITION_SCHEMA)
    self.assertEqual(manifest["split_subjects"], SPLIT_SUBJECTS)
    self.assertEqual(
      sha256("scripts/curate_conditional_shoulder80.py"), manifest["curator_sha256"]
    )
    source_splits = {}
    counts = {name: np.zeros(3, dtype=int) for name in SPLIT_SUBJECTS}
    for item in manifest["files"]:
      self.assertEqual(sha256(DATA / item["path"]), item["sha256"])
      self.assertEqual(subject_split(item["source"]), item["split"])
      self.assertEqual(
        source_splits.setdefault(item["source"], item["split"]), item["split"]
      )
      with np.load(DATA / item["path"], allow_pickle=False) as shard:
        raw, labels, phases = shard["windows"], shard["condition_id"], shard["phase_id"]
        self.assertEqual(raw.shape[1:], (10, 59))
        self.assertTrue(np.isfinite(raw).all())
        self.assertTrue(np.isin(labels, (0, 1, 2)).all())
        self.assertTrue((phases[labels == 0] == 0).all())
        self.assertTrue(np.isin(phases[labels != 0], (1, 2, 3)).all())
        counts[item["split"]] += np.bincount(labels, minlength=3)
    self.assertEqual(
      {key: value.tolist() for key, value in counts.items()}, manifest["counts"]
    )
    for key, name in (
      ("windows_manifest_sha256", "windows.csv"),
      ("audit_sha256", "audit.json"),
      ("norm_stats_sha256", "norm_stats.npz"),
    ):
      self.assertEqual(sha256(DATA / name), manifest[key])
    for source, digest in manifest["sources"].items():
      self.assertEqual(sha256(Path(manifest["config"]["input_dir"]) / source), digest)

  def test_normalization_uses_only_subjects_one_to_three(self):
    manifest = self.dataset()
    arrays = []
    for item in manifest["files"]:
      if item["split"] == "train":
        with np.load(DATA / item["path"], allow_pickle=False) as shard:
          arrays.append(shard["windows"])
    low, high = np.percentile(
      np.concatenate(arrays).reshape(-1, 59), [1, 99], axis=0
    ).astype(np.float32)
    high = np.where(high - low < 1e-6, low + 1.0, high)
    with np.load(DATA / "norm_stats.npz") as stats:
      np.testing.assert_array_equal(stats["q_low"], low)
      np.testing.assert_array_equal(stats["q_high"], high)

  def test_every_early_window_links_to_saved_same_direction_mature_window(self):
    manifest = self.dataset()
    seen = 0
    for item in manifest["files"]:
      with np.load(DATA / item["path"], allow_pickle=False) as shard:
        labels, phases, starts, links = (
          shard[key]
          for key in ("condition_id", "phase_id", "frame50_start", "early_link_frame50")
        )
        for index in np.flatnonzero(phases == 1):
          candidates = np.flatnonzero(starts == links[index])
          self.assertEqual(len(candidates), 1)
          target = candidates[0]
          self.assertEqual(labels[index], labels[target])
          self.assertIn(phases[target], (2, 3))
          self.assertGreater(links[index], starts[index])
          self.assertLessEqual(links[index] - starts[index], 50)
          seen += 1
        self.assertTrue((links[phases != 1] == -1).all())
    self.assertEqual(
      seen,
      sum(counts[1][1] + counts[2][1] for counts in manifest["phase_counts"].values()),
    )

  def test_saved_gate_recomputes_and_blocks_training(self):
    manifest = self.dataset()
    with (DATA / "windows.csv").open(newline="") as stream:
      rows = list(csv.DictReader(stream))
    for row in rows:
      for key in ("condition_id", "phase_id", "frame50_start", "frame50_end_exclusive"):
        row[key] = int(row[key])
    baseline, _ = baseline_intervals(Path(manifest["config"]["baseline_dir"]))
    self.assertEqual(data_gate(rows, baseline), manifest["gate"])
    state = json.loads((DATA.parent / "manifest.json").read_text())
    if not manifest["gate"]["passed"]:
      self.assertEqual(state["status"], "blocked_data_gate")
      self.assertEqual(state["completed_formal_prior_epochs"], 0)
      self.assertEqual(state["stages"]["prior_smoke"], "not_started")
      self.assertFalse(state["prior_published"])
    self.assertFalse(state["ppo_authorized"])

  def test_gpu_identity_and_occupancy_are_not_silently_overridden(self):
    with patch(
      "scripts.run_conditional_shoulder80_data_review.subprocess.check_output",
      return_value="wrong-uuid\n",
    ):
      with self.assertRaisesRegex(RuntimeError, "UUID mismatch"):
        require_gpu()
    with patch(
      "scripts.run_conditional_shoulder80_data_review.subprocess.check_output",
      side_effect=[GPU_UUID + "\n", "12345\n"],
    ):
      with self.assertRaisesRegex(RuntimeError, "no preemption"):
        require_gpu()

  def test_data_review_runs_cpu_only_and_isolates_caches(self):
    with (
      tempfile.TemporaryDirectory() as directory,
      patch("scripts.run_conditional_shoulder80_data_review.OUTPUT", Path(directory)),
    ):
      env = review_environment()
      self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
      self.assertEqual(env["MUJOCO_EGL_DEVICE_ID"], "0")
      for key in (
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "MPLCONFIGDIR",
        "WARP_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TMPDIR",
      ):
        self.assertTrue(Path(env[key]).is_relative_to(directory))


if __name__ == "__main__":
  unittest.main()
