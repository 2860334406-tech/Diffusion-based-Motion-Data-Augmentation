"""Tests for exact 80/20 prior sampling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts.pretrain import ExactCategorySampler

from smp.pretrain.dataset import MotionWindowDataset


class Static80SamplingTest(unittest.TestCase):
  def test_exact_ratio_and_reproducibility(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      np.savez_compressed(
        root / "dynamic.npz",
        windows=np.zeros((4, 10, 59), dtype=np.float32),
      )
      np.savez_compressed(
        root / "stand.npz",
        windows=np.ones((4, 10, 59), dtype=np.float32),
      )
      manifest = root / "sampling.json"
      manifest.write_text(
        json.dumps(
          {
            "schema_version": 1,
            "samples_per_epoch": 10,
            "category_fractions": {"stand": 0.8, "dynamic": 0.2},
            "files": {
              "dynamic.npz": "dynamic",
              "stand.npz": "stand",
            },
          }
        ),
        encoding="utf-8",
      )
      dataset = MotionWindowDataset(root)
      first = ExactCategorySampler(dataset, manifest, seed=42)
      second = ExactCategorySampler(dataset, manifest, seed=42)

      indices = list(first)
      self.assertEqual(indices, list(second))
      dynamic_slice = dataset.file_slices["dynamic.npz"]
      dynamic = sum(
        dynamic_slice.start <= index < dynamic_slice.stop for index in indices
      )
      self.assertEqual(dynamic, 2)
      self.assertEqual(len(indices) - dynamic, 8)
      self.assertEqual(first.metadata()["category_counts"], {"stand": 8, "dynamic": 2})

      next_epoch = list(first)
      self.assertNotEqual(indices, next_epoch)
      self.assertEqual(len(next_epoch), 10)

  def test_manifest_must_cover_every_npz(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      np.savez_compressed(
        root / "only.npz",
        windows=np.zeros((2, 10, 59), dtype=np.float32),
      )
      manifest = root / "sampling.json"
      manifest.write_text(
        json.dumps(
          {
            "schema_version": 1,
            "samples_per_epoch": 2,
            "category_fractions": {"stand": 1.0},
            "files": {},
          }
        ),
        encoding="utf-8",
      )
      dataset = MotionWindowDataset(root)
      with self.assertRaisesRegex(ValueError, "file mismatch"):
        ExactCategorySampler(dataset, manifest, seed=42)


if __name__ == "__main__":
  unittest.main()
