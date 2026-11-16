"""CPU checks for stronger shoulder curation and the isolated v2 experiment."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import tyro
from scripts.curate_conditional_contact_v2 import Cfg, shoulder_phases
from scripts.pretrain_conditional_contact import Cfg as PriorCfg
from scripts.run_conditional_contact_v2_suite import (
  BUDGET,
  OUTPUT,
  PRIOR,
  child_environment,
  curriculum_command,
  prior_command,
  require_storage,
)
from scripts.train_force_unloading_curriculum import CurriculumCfg

from smp.pretrain.conditional_dataset import ConditionalMotionDataset
from smp.pretrain.conditioning import ShoulderPhaseBalancedSampler, sha256


class ConditionalV2Tests(unittest.TestCase):
  def test_exact_class_and_phase_counts(self):
    labels = torch.tensor([0, 1, 1, 1, 2, 2, 3, 4])
    phases = torch.tensor([0, 1, 1, 2, 1, 2, 0, 0])
    sampler = ShoulderPhaseBalancedSampler(labels, phases)
    indices = torch.tensor(list(sampler))
    self.assertEqual(len(indices), 3360)
    self.assertEqual(torch.bincount(labels[indices]).tolist(), [672]*5)
    for condition in (1, 2):
      selected = indices[labels[indices] == condition]
      self.assertEqual(torch.bincount(phases[selected], minlength=3).tolist(), [0, 336, 336])
    expected = list(sampler)
    restored = ShoulderPhaseBalancedSampler(labels, phases)
    restored.epoch = 1
    self.assertEqual(expected, list(restored))

  def test_invalid_phase_pools_fail(self):
    labels = torch.tensor([0, 1, 1, 2, 2, 3, 4])
    with self.assertRaisesRegex(ValueError, "Missing shoulder phase"):
      ShoulderPhaseBalancedSampler(labels, torch.tensor([0, 1, 1, 1, 2, 0, 0]))
    with self.assertRaisesRegex(ValueError, "entering or holding"):
      ShoulderPhaseBalancedSampler(labels, torch.tensor([0, 0, 2, 1, 2, 0, 0]))
    with self.assertRaisesRegex(ValueError, "aligned labels"):
      ShoulderPhaseBalancedSampler(labels, torch.zeros(3), samples=3360)

  def test_curation_mirror_and_neutral_rejection(self):
    windows = np.zeros((6, 10, 59))
    windows[0, :, 21] = np.linspace(.05, .10, 10)
    windows[1, :, 21] = .14
    windows[2, :, 21] = np.linspace(.01, .04, 10)
    windows[3, :, 21] = np.linspace(.30, .40, 10)
    windows[4, :, 21] = np.linspace(.05, .14, 10)
    windows[5, :, 21] = 0
    labels = np.array([1, 1, 1, 1, 1, 0])
    keep, phases = shoulder_phases(windows, labels, Cfg())
    np.testing.assert_array_equal(keep, [True, True, False, False, True, True])
    np.testing.assert_array_equal(phases[keep], [1, 2, 1, 0])
    mirrored = windows.copy()
    mirrored[..., 21] *= -1
    mirror_labels = np.where(labels == 1, 2, labels)
    mirror_keep, mirror_phases = shoulder_phases(mirrored, mirror_labels, Cfg())
    np.testing.assert_array_equal(mirror_keep, keep)
    np.testing.assert_array_equal(mirror_phases, phases)

  def test_prior_cli_keeps_budget_and_architecture(self):
    # -u is an interpreter argument, not part of the dataclass CLI.
    cfg = tyro.cli(PriorCfg, args=prior_command()[4:])
    self.assertEqual(cfg.num_epochs, BUDGET)
    self.assertEqual((cfg.batch_size, cfg.d_model, cfg.num_layers), (1024, 128, 2))
    self.assertEqual((cfg.num_timesteps, cfg.num_noise_samples), (50, 10))
    self.assertEqual((cfg.lr, cfg.condition_dropout, cfg.seed), (3e-4, .1, 42))
    self.assertEqual(cfg.data_dir, str(OUTPUT / "data"))
    self.assertEqual(cfg.output_dir, str(OUTPUT / "prior"))
    smoke = tyro.cli(PriorCfg, args=prior_command(smoke=True)[4:])
    self.assertEqual(smoke.num_epochs, 1)
    self.assertFalse(smoke.use_wandb)
    self.assertNotEqual(smoke.output_dir, cfg.output_dir)

  def test_curriculum_keeps_scratch_and_isolated_paths(self):
    cfg = tyro.cli(CurriculumCfg, args=curriculum_command("test-sha")[3:])
    self.assertEqual(cfg.prior_checkpoint, str(PRIOR))
    self.assertEqual(cfg.prior_sha256, "test-sha")
    self.assertEqual(cfg.gpu_id, 1)
    self.assertEqual((cfg.gsi_max_root_planar_speed, cfg.stationary_speed_scale), (.15, 10))
    self.assertEqual(cfg.target_probabilities, (.25,)*4)
    self.assertTrue(cfg.all_contact_evaluation)
    self.assertEqual(cfg.num_envs, 4096)
    self.assertEqual(cfg.save_interval, 1000)
    self.assertEqual(cfg.run_prefix, "conditional_contact_v2")
    for value in (cfg.log_root, cfg.state_dir, cfg.evaluation_dir):
      self.assertTrue(Path(value).is_relative_to(OUTPUT))

  def test_gpu_and_new_caches_isolated(self):
    env = child_environment()
    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
    self.assertEqual(env["MUJOCO_EGL_DEVICE_ID"], "1")
    for name in ("WANDB_DIR", "WANDB_CACHE_DIR", "WARP_CACHE_PATH", "MPLCONFIGDIR",
                 "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                 "XDG_CACHE_HOME", "TMPDIR"):
      self.assertTrue(Path(env[name]).is_relative_to(OUTPUT), name)

  def test_storage_never_falls_back_to_root(self):
    with tempfile.TemporaryDirectory() as root:
      with patch("scripts.run_conditional_contact_v2_suite.OUTPUT", Path(root)):
        with self.assertRaisesRegex(RuntimeError, "dedicated persistent volume"):
          require_storage()

  def test_curated_lineage_keeps_original_windows_and_splits(self):
    root = OUTPUT / "data"
    if not (root / "manifest.json").exists():
      self.skipTest("v2 data not curated yet")
    manifest = json.loads((root / "manifest.json").read_text())
    parent_path = Path(manifest["parent_manifest"])
    self.assertEqual(sha256(parent_path), manifest["parent_sha256"])
    parent = json.loads(parent_path.read_text())
    originals = {item["path"]: item for item in parent["files"]}
    seen = {"train": set(), "validation": set()}
    for item in manifest["files"]:
      old = originals[item["path"]]
      self.assertEqual((item["split"], item["mirrored"]), (old["split"], old["mirrored"]))
      self.assertEqual(item["parent_shard_sha256"], old["sha256"])
      with np.load(root / item["path"]) as shard, np.load(parent_path.parent / old["path"]) as raw:
        np.testing.assert_array_equal(shard["windows"], raw["windows"][shard["parent_index"]])
        np.testing.assert_array_equal(shard["condition_id"], raw["condition_id"][shard["parent_index"]])
      seen[item["split"]].add(item["source"])
    self.assertFalse(seen["train"] & seen["validation"])
    for split in ("train", "validation"):
      data = ConditionalMotionDataset(root, split)
      self.assertTrue(torch.isfinite(data.windows).all())
      self.assertEqual(torch.bincount(data.labels).tolist(), manifest["counts"][split])
      for condition in (0, 3, 4):
        self.assertEqual(manifest["counts"][split][condition], parent["counts"][split][condition])
      self.assertEqual(manifest["counts"][split][1], manifest["counts"][split][2])


if __name__ == "__main__":
  unittest.main()
