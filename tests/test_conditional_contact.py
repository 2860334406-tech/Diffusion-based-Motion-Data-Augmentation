"""Behavioral tests for the five-condition experiment and legacy compatibility."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scripts.curate_conditional_contact import Cfg, classify_windows, subject_split
from scripts.curate_lafan_low_speed import mirror_g1_motion
from scripts.pretrain import _diffusion_loss

from smp.pretrain.conditioning import BalancedConditionSampler, condition_index
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.rl.conditional_normalizer import ConditionalDiffNormalizer
from smp.rl.tasks.force_unloading.mdp.conditions import (
  ContactConditionState,
  capsule_start_distance,
)
from smp.rl.tasks.force_unloading.mdp.controller import canonical_drive_direction
from smp.rl.utils import load_denoiser


class ConditionalContactTests(unittest.TestCase):
  def test_condition_interface_and_legacy_weights(self):
    torch.manual_seed(42)
    x, t = torch.randn(3, 10, 59), torch.zeros(3, dtype=torch.long)
    model = DiffusionDenoiser(59, 10, d_model=32, num_conditions=5)
    with self.assertRaises(ValueError):
      model(x, t)
    result = model(x[:1].expand(3, -1, -1), t, torch.tensor([0, 1, 5]))
    self.assertFalse(torch.allclose(result[0], result[1]))
    self.assertFalse(torch.allclose(result[0], result[2]))
    old, *_ = load_denoiser("datasets/pretrain_ckpt/pretrained_lafan_low_speed_static80.pt", "cpu")
    self.assertEqual(old.num_conditions, 0)
    self.assertTrue(torch.isfinite(old(x, t)).all())
    self.assertFalse(any("condition_embedding" in key for key in old.state_dict()))

  def test_label_expansion_matches_noise_samples(self):
    class Recorder(torch.nn.Module):
      def forward(self, x, t, ids):
        self.ids = ids
        return torch.zeros_like(x)
    model = Recorder()
    labels = torch.tensor([0, 2, 4])
    _diffusion_loss(model, DDPMScheduler(), torch.zeros(3, 10, 59), 4, labels)
    torch.testing.assert_close(model.ids, labels.repeat_interleave(4))

  def test_balanced_sampling_and_resume(self):
    labels = torch.tensor([0]*2 + [1]*9 + [2]*5 + [3]*20 + [4]*3)
    sampler = BalancedConditionSampler(labels)
    first = torch.tensor(list(sampler))
    self.assertEqual(torch.bincount(labels[first]).tolist(), [672]*5)
    second = list(sampler)
    resumed = BalancedConditionSampler(labels)
    resumed.epoch = 1
    self.assertEqual(second, list(resumed))
    self.assertEqual(condition_index("stand"), 0)
    self.assertEqual(condition_index("null"), 5)
    with self.assertRaises(ValueError):
      condition_index("invalid")

  def test_mirror_exchanges_shoulder_labels(self):
    a = np.zeros((60, 36))
    a[:, 2], a[:, 6] = .8, 1
    a[:, 19] = np.linspace(.10, .20, len(a))
    cfg = Cfg()
    left = classify_windows(a, cfg)[2]
    right = classify_windows(mirror_g1_motion(a), cfg)[2]
    self.assertTrue((left == 1).all())
    self.assertTrue((right == 2).all())
    self.assertEqual(subject_split("dance2_subject5_mirror.csv"), "validation")
    self.assertEqual(subject_split("dance2_subject4.csv"), "train")

  def test_contact_debounce_transition_and_reset(self):
    state = ContactConditionState(2, "cpu")
    zero = torch.zeros(2, 4)
    for _ in range(15):
      state.update(zero)
    self.assertEqual(state.reward_ids.tolist(), [0, 0])
    forces = zero.clone()
    forces[0, 0], forces[1, 3] = .01, .02
    for _ in range(2):
      state.update(forces)
    self.assertEqual(state.ids.tolist(), [0, 0])
    state.update(forces)
    self.assertEqual(state.ids.tolist(), [1, 4])
    self.assertEqual(state.reward_ids.tolist(), [5, 5])
    for _ in range(9):
      state.update(forces)
    self.assertEqual(state.reward_ids.tolist(), [1, 4])
    for _ in range(3):
      state.update(zero)
    self.assertEqual(state.ids.tolist(), [0, 0])
    state.reset(torch.tensor([0, 1]))
    self.assertEqual(state.reward_ids.tolist(), [0, 0])
    self.assertFalse(state.history.any())

  def test_capsule_gap_and_back_direction(self):
    dirs = canonical_drive_direction(torch.tensor([0, 1, 2, 3]), "front_to_back")
    torch.testing.assert_close(dirs[:, 0], torch.tensor([-1., -1., -1., 1.]))
    d = torch.cat((dirs, torch.zeros(4, 1)), -1)
    axes = torch.tensor([[0., 0., 1.], [.6, 0., .8], [1., 0., 0.], [0., 0., 1.]])
    h, r = torch.full((4,), .06), torch.full((4,), .09)
    distance = capsule_start_distance(d, axes, h, r, .10, .02)
    point = -distance[:, None]*d
    projection = (point*axes).sum(-1).clamp(-h, h)
    gap = torch.linalg.vector_norm(point-projection[:, None]*axes, dim=-1)-r-.10
    torch.testing.assert_close(gap, torch.full((4,), .02), atol=1e-6, rtol=0)

  def test_normalization_separates_conditions_and_restores(self):
    norm = ConditionalDiffNormalizer(50, "cpu")
    values = norm.update_and_normalize(8, torch.tensor([1., 9.]), torch.tensor([0, 1]))
    torch.testing.assert_close(values, torch.ones(2))
    self.assertEqual(norm.count[:, 8].tolist(), [1, 1, 0, 0, 0, 0])
    restored = ConditionalDiffNormalizer(50, "cpu")
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / "state.pt"
      torch.save(norm.state_dict(), path)
      restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.mean, norm.mean)
    torch.testing.assert_close(restored.count, norm.count)

  def test_curated_split_and_mirror_counts(self):
    path = Path("logs/reproduction/conditional_contact_v1/data/manifest.json")
    if not path.exists():
      self.skipTest("Data curation has not run")
    manifest = json.loads(path.read_text())
    train, validation = set(), set()
    for item in manifest["files"]:
      (train if item["split"] == "train" else validation).add(item["source"])
    self.assertFalse(train & validation)
    for counts in manifest["counts"].values():
      self.assertGreater(min(counts), 0)
      self.assertEqual(counts[1], counts[2])


if __name__ == "__main__":
  unittest.main()
