"""Tests for low-speed LAFAN curation and scalar-only W&B logging."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scripts.curate_lafan_low_speed import (
  Cfg as CurationCfg,
)
from scripts.curate_lafan_low_speed import (
  crop_clip,
  load_manifest,
  mirror_g1_motion,
)
from scripts.pretrain import _maybe_upload_checkpoint

from smp.pretrain.pretrain_cfg import PretrainCfg
from smp.rl.utils import load_denoiser


class _FakeRun:
  def __init__(self) -> None:
    self.saved: list[tuple[str, str]] = []

  def save(self, path: str, base_path: str) -> None:
    self.saved.append((path, base_path))


class LafanLowSpeedCurationTest(unittest.TestCase):
  def test_crop_is_half_open_ninety_frames(self) -> None:
    motion = np.arange(200 * 36, dtype=np.float64).reshape(200, 36)
    clip = crop_clip(motion, 1.0, 4.0, input_fps=30, expected_duration_s=3.0)
    self.assertEqual(clip.shape, (90, 36))
    np.testing.assert_array_equal(clip[0], motion[30])
    np.testing.assert_array_equal(clip[-1], motion[119])

  def test_mirror_is_an_involution(self) -> None:
    rng = np.random.default_rng(42)
    motion = rng.normal(size=(90, 36))
    np.testing.assert_allclose(
      mirror_g1_motion(mirror_g1_motion(motion)), motion, atol=1.0e-12
    )

  def test_matches_shipped_forward_mirror_convention(self) -> None:
    source = np.loadtxt("datasets/csv/forward/g1_walk.csv", delimiter=",")
    expected = np.loadtxt("datasets/csv/forward/g1_walk_mirror.csv", delimiter=",")
    np.testing.assert_allclose(mirror_g1_motion(source), expected, atol=1.0e-7)

  def test_manifest_has_balanced_accepted_clips(self) -> None:
    clips = load_manifest(CurationCfg().manifest)
    accepted = [clip for clip in clips if clip.status == "accepted"]
    self.assertEqual(sum(clip.category == "stand" for clip in accepted), 6)
    self.assertEqual(sum(clip.category == "dynamic" for clip in accepted), 6)

  def test_checkpoint_upload_requires_explicit_opt_in(self) -> None:
    run = _FakeRun()
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      checkpoint = base / "pretrained.pt"
      _maybe_upload_checkpoint(run, checkpoint, base, upload_model=False)
      self.assertEqual(run.saved, [])
      _maybe_upload_checkpoint(run, checkpoint, base, upload_model=True)
      self.assertEqual(run.saved, [(str(checkpoint), str(base))])
    self.assertFalse(PretrainCfg(device="cpu").upload_model)

  def test_low_speed_prior_checkpoint_loads(self) -> None:
    checkpoint = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_balance.pt"
    model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(
      checkpoint, "cpu"
    )
    self.assertEqual((window_size, feature_dim), (10, 59))
    self.assertEqual(scheduler.num_timesteps, 50)
    self.assertEqual((q_low.numel(), q_high.numel()), (59, 59))
    self.assertFalse(model.training)
    self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
    self.assertTrue(torch.isfinite(q_low).all() and torch.isfinite(q_high).all())


if __name__ == "__main__":
  unittest.main()
