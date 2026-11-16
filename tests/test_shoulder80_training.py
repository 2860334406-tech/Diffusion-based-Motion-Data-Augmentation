"""New trainer isolation, schema compatibility, sampling, and complete resume."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scripts.pretrain import _diffusion_loss
from scripts.pretrain_shoulder80 import FORMAL, restore_training, rng_state

from smp.pretrain.conditioning import CONDITION_SCHEMA as FIVE_SCHEMA
from smp.pretrain.conditioning import sample_condition
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.pretrain.schema import resolve_condition, validate_schema
from smp.pretrain.shoulder80 import CONDITION_SCHEMA, SAMPLING, Shoulder80Sampler
from smp.pretrain.shoulder80_dataset import DATA_DIR, DATA_SHA, Shoulder80Dataset
from smp.pretrain.shoulder80_score import paired_scores
from smp.rl.utils import load_denoiser


class ShoulderTrainingTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)

  def test_schemas_do_not_conflate_chest_and_null(self):
    self.assertEqual(resolve_condition("null", CONDITION_SCHEMA), 3)
    self.assertEqual(resolve_condition("chest", FIVE_SCHEMA), 3)
    self.assertEqual(resolve_condition("null", FIVE_SCHEMA), 5)
    for value in ("chest", "back", 4, 5, -1):
      with self.assertRaises(ValueError):
        resolve_condition(value, CONDITION_SCHEMA)
    with self.assertRaises(ValueError):
      validate_schema(FIVE_SCHEMA, 3)

  def test_loader_and_named_sampling_three_five_and_unconditional(self):
    for number, schema in ((0, None), (3, CONDITION_SCHEMA), (5, FIVE_SCHEMA)):
      model = DiffusionDenoiser(
        59, 10, d_model=16, nhead=4, num_layers=1, num_conditions=number
      )
      with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        torch.save(
          dict(
            model=model.state_dict(),
            condition_schema=schema,
            q_low=np.zeros(59, np.float32),
            q_high=np.ones(59, np.float32),
            cfg=dict(
              feature_dim=59,
              window_size=10,
              d_model=16,
              nhead=4,
              num_layers=1,
              num_conditions=number,
              num_timesteps=2,
            ),
          ),
          path,
        )
        bundle = load_denoiser(str(path), "cpu")
        if number:
          self.assertTrue(torch.isfinite(sample_condition(bundle, "stand", 2)).all())
          self.assertTrue(
            torch.isfinite(
              sample_condition(bundle, "null", 2, guidance_scale=2.0)
            ).all()
          )
          if number == 3:
            with self.assertRaises(ValueError):
              sample_condition(bundle, "chest", 2)
        else:
          self.assertTrue(
            torch.isfinite(
              bundle[0](torch.randn(2, 10, 59), torch.zeros(2, dtype=torch.long))
            ).all()
          )

  def test_noise_repeat_label_alignment(self):
    class Capture(torch.nn.Module):
      def forward(self, x, t, conditions):
        self.conditions = conditions.clone()
        return torch.zeros_like(x)

    model = Capture()
    _diffusion_loss(
      model, DDPMScheduler(50), torch.zeros(3, 10, 59), 10, torch.tensor([0, 1, 3])
    )
    self.assertEqual(model.conditions.tolist(), [0] * 10 + [1] * 10 + [3] * 10)

  def test_training_uses_approved_budget_and_no_extra_losses(self):
    self.assertEqual(
      (FORMAL["num_epochs"], FORMAL["num_layers"], FORMAL["batch_size"]),
      (10000, 2, 1024),
    )
    self.assertFalse(FORMAL["ema"])
    self.assertEqual(
      (FORMAL["checkpoint_interval"], FORMAL["save_interval"]), (100, 1000)
    )

  def test_complete_restore_reproduces_next_update_and_sampling(self):
    torch.manual_seed(42)
    labels, phases = (
      torch.tensor([0, 1, 1, 1, 2, 2, 2]),
      torch.tensor([0, 1, 2, 3, 1, 2, 3]),
    )
    sampler = Shoulder80Sampler(labels, phases)
    model = DiffusionDenoiser(59, 10, d_model=16, num_layers=1, num_conditions=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = DDPMScheduler(50)
    x = torch.randn(7, 10, 59)

    def update(m, o):
      o.zero_grad()
      loss = _diffusion_loss(m, scheduler, x, 2, labels)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
      o.step()
      return loss.detach()

    for _ in range(4):
      update(model, optimizer)
    list(sampler)
    payload = copy.deepcopy(
      dict(
        cfg=FORMAL,
        condition_schema=CONDITION_SCHEMA,
        data_sha256=DATA_SHA,
        sampling=SAMPLING,
        initialization="scratch",
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        sampler=sampler.state_dict(),
        epoch=1,
        optimizer_steps=4,
        rng=rng_state(),
      )
    )
    expected_draw = list(sampler)
    expected_loss = update(model, optimizer)
    restored = DiffusionDenoiser(59, 10, d_model=16, num_layers=1, num_conditions=3)
    opt = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    other_sampler = Shoulder80Sampler(labels, phases)
    restore_training(payload, restored, opt, other_sampler, FORMAL, DATA_SHA)
    self.assertEqual(list(other_sampler), expected_draw)
    self.assertTrue(torch.equal(update(restored, opt), expected_loss))
    for name, value in model.state_dict().items():
      self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

  def test_paired_score_same_noise_mask_and_range(self):
    model = DiffusionDenoiser(59, 10, d_model=16, num_layers=1, num_conditions=3).eval()
    scheduler, means = DDPMScheduler(50), torch.ones(3, 3)
    windows = torch.randn(3, 10, 59)
    a = paired_scores(model, scheduler, windows, 1, means, repeats=2)
    b = paired_scores(model, scheduler, windows, 2, means, repeats=2)
    torch.testing.assert_close(
      a["correct_energy"], b["opposite_energy"], rtol=0, atol=0
    )
    torch.testing.assert_close(
      a["contrast_margin"], -b["contrast_margin"], rtol=0, atol=0
    )
    for frames in (1, 3, 5, 10):
      score = paired_scores(
        model, scheduler, windows, 1, means, repeats=2, valid_frames=frames
      )
      self.assertTrue(((score["reward"] >= 0) & (score["reward"] <= 1)).all())

  @unittest.skipUnless((DATA_DIR / "manifest.json").exists(), "Expanded data absent")
  def test_combined_dataset_includes_lafan_and_external_and_mirrors(self):
    for split, counts in (
      ("train", [3660, 1232, 1232]),
      ("validation", [1898, 590, 590]),
      ("test", [482, 889, 889]),
    ):
      dataset = Shoulder80Dataset(split=split)
      self.assertEqual(torch.bincount(dataset.labels).tolist(), counts)
      self.assertIn("lafan", {r["dataset"] for r in dataset.rows})
      self.assertIn("amass_g1", {r["dataset"] for r in dataset.rows})
      mirror = dataset.mirror_indices()
      self.assertTrue(torch.equal(mirror[mirror], torch.arange(len(dataset))))
      self.assertTrue(torch.equal(dataset.phase_ids[mirror], dataset.phase_ids))
      sampler = Shoulder80Sampler(dataset.labels, dataset.phase_ids)
      sampled = torch.tensor(list(sampler))
      self.assertEqual(
        torch.bincount(dataset.labels[sampled]).tolist(), [672, 1344, 1344]
      )


if __name__ == "__main__":
  unittest.main()
