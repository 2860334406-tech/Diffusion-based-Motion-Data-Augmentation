"""Read-only acceptance checks for the new external motion review artifacts."""

import csv
import json
import unittest
from dataclasses import asdict

import numpy as np
import torch
from scripts.curate_conditional_shoulder80 import Cfg, baseline_intervals
from scripts.review_external_shoulder_sources import (
  BASE_DATA,
  BASE_DATA_SHA,
  MAX_DOWNLOAD_BYTES,
  OUTPUT,
  REPOS,
  ROOT,
  actor_group,
  coverage_gate,
  digest,
  group_split,
)

from smp.pretrain.shoulder80 import SAMPLING, Shoulder80Sampler


@unittest.skipUnless((OUTPUT / "review/manifest.json").exists(), "Review not complete")
class ExternalArtifactsTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.directory = OUTPUT / "review"
    cls.manifest = json.loads((cls.directory / "manifest.json").read_text())
    cls.plan = json.loads((OUTPUT / "plan.json").read_text())
    cls.downloads = json.loads((OUTPUT / "downloads.json").read_text())
    with (cls.directory / "windows.csv").open() as stream:
      cls.rows = list(csv.DictReader(stream))

  def test_frozen_rules_and_historical_identities(self):
    self.assertEqual(digest(BASE_DATA / "manifest.json"), BASE_DATA_SHA)
    self.assertEqual(self.manifest["filter_config"], asdict(Cfg()))
    self.assertEqual(self.plan["filter_config"], asdict(Cfg()))
    self.assertEqual(
      self.manifest["filter_source_sha256"],
      digest(ROOT / "scripts/curate_conditional_shoulder80.py"),
    )
    self.assertFalse(self.manifest["training_started"])
    self.assertFalse(self.manifest["prior_acceptance_evaluated"])
    self.assertEqual(self.manifest["sampling"], SAMPLING)
    self.assertEqual(
      digest(
        ROOT / "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
      ),
      "ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734",
    )

  def test_revisions_source_bytes_and_download_budget(self):
    self.assertEqual(self.plan["repos"], {k: list(v) for k, v in REPOS.items()})
    self.assertEqual(self.downloads["plan_sha256"], digest(OUTPUT / "plan.json"))
    self.assertEqual(self.downloads["errors"], [])
    self.assertLessEqual(
      sum(v["bytes"] for v in self.downloads["files"].values()), MAX_DOWNLOAD_BYTES
    )
    for source in self.plan["sources"]:
      item = self.downloads["files"][source["source_id"]]
      self.assertEqual(digest(OUTPUT / "raw" / source["source_id"]), item["sha256"])
      self.assertEqual(
        actor_group(source["dataset"], source["remote_path"]), source["source_group"]
      )
      self.assertEqual(group_split(source["source_group"]), source["split"])

  def test_all_shards_finite_labels_lineage_and_hashes(self):
    for item in self.manifest["external_files"]:
      path = self.directory / item["path"]
      self.assertEqual(digest(path), item["sha256"])
      with np.load(path, allow_pickle=False) as data:
        self.assertEqual(data["windows"].shape[1:], (10, 59))
        self.assertTrue(np.isfinite(data["windows"]).all())
        self.assertTrue(np.isfinite(data["metrics"]).all())
        self.assertTrue(np.isin(data["condition_id"], (0, 1, 2)).all())
        rows = [r for r in self.rows if r["shard"] == str(path)]
        self.assertEqual(len(rows), len(data["windows"]))
        for row in rows:
          i = int(row["index"])
          self.assertEqual(int(row["condition_id"]), data["condition_id"][i])
          self.assertEqual(int(row["phase_id"]), data["phase_id"][i])
          self.assertEqual(int(row["frame50_start"]), data["frame50_start"][i])
    for field, name in (
      ("windows_manifest_sha256", "windows.csv"),
      ("audit_sha256", "audit.json"),
      ("norm_stats_sha256", "norm_stats.npz"),
    ):
      self.assertEqual(self.manifest[field], digest(self.directory / name))

  def test_early_entries_link_to_saved_mature_windows(self):
    mature = {
      (r["source_id"], r["mirrored"], r["condition_id"], int(r["frame50_start"]))
      for r in self.rows
      if int(r["phase_id"]) in (2, 3)
    }
    for row in self.rows:
      if int(row["phase_id"]) != 1:
        continue
      start, link = int(row["frame50_start"]), int(row["early_link_frame50"])
      self.assertTrue(start < link <= start + 50)
      self.assertIn(
        (row["source_id"], row["mirrored"], row["condition_id"], link), mature
      )

  def test_gate_and_counts_recompute_without_moving_subjects(self):
    baseline, _ = baseline_intervals(
      ROOT / "logs/reproduction/conditional_contact_v2/data"
    )
    self.assertEqual(coverage_gate(self.rows, baseline), self.manifest["gate"])
    for split in ("train", "validation", "test"):
      rows = [r for r in self.rows if r["split"] == split]
      counts = [sum(int(r["condition_id"]) == c for r in rows) for c in range(3)]
      self.assertEqual(counts, self.manifest["counts"][split])

  def test_training_only_normalization_and_exact_sampling(self):
    windows, labels, phases = [], [], []
    for shard in sorted({r["shard"] for r in self.rows if r["split"] == "train"}):
      with np.load(shard, allow_pickle=False) as data:
        windows.append(data["windows"])
        labels.append(data["condition_id"])
        phases.append(data["phase_id"])
    low, high = np.percentile(
      np.concatenate(windows).reshape(-1, 59), [1, 99], axis=0
    ).astype(np.float32)
    high = np.where(high - low < 1e-6, low + 1, high)
    with np.load(self.directory / "norm_stats.npz", allow_pickle=False) as data:
      np.testing.assert_array_equal(low, data["q_low"])
      np.testing.assert_array_equal(high, data["q_high"])
    labels, phases = (
      torch.tensor(np.concatenate(labels)),
      torch.tensor(np.concatenate(phases)),
    )
    sampler = Shoulder80Sampler(labels, phases)
    selection = torch.tensor(list(sampler))
    self.assertEqual(torch.bincount(labels[selection]).tolist(), [672, 1344, 1344])
    for c in (1, 2):
      self.assertEqual(
        torch.bincount(phases[selection[labels[selection] == c]], minlength=4).tolist(),
        [0, 336, 336, 672],
      )
    state = sampler.state_dict()
    restored = Shoulder80Sampler(labels, phases)
    restored.load_state_dict(state)
    self.assertEqual(list(sampler), list(restored))


if __name__ == "__main__":
  unittest.main()
