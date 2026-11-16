"""EXP-038: LAFAN supplies 90% of stand draws; EXP-036 evaluation stays frozen."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from smp.data.shoulder_ik import atomic_json
from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder_mixed import (
  SAMPLING as EVALUATION_SAMPLING,
  ShoulderMixedSampler,
  condition_weights,
)
from smp.pretrain.shoulder_mixed_dataset import _verify
from smp.pretrain.shoulder_mixed_p01p02_dataset import (
  POLICY,
  ShoulderMixedP01P02Dataset,
)

PARENT = Path("logs/reproduction/g1_shoulder_mixed_p01p02_prior_v1/data")
PARENT_SHA = "41c2b9f9c466f8a12d86cb812157958b83ccd990533f416d173aaa053b2902f0"
NORM_SHA = "5f3900e1049ea683709c16badfd201db884739431e8814c79738ad4d9e529ee8"
DATA_DIR = Path("logs/reproduction/g1_shoulder_mixed_lafan90_prior_v1/data")
STAND_SOURCE_ORDER = ("lafan", "ik", "personal", "g1_moves", "amass_g1")
STAND_MASS = dict(zip(STAND_SOURCE_ORDER, (0.90, 0.025, 0.025, 0.025, 0.025)))
SAMPLING = {
  "algorithm": "mixed_lafan90_five_epoch_source_rotation_v1",
  "samples_per_epoch": 3360,
  "per_condition": [672, 1344, 1344],
  "stand_source_mass": STAND_MASS,
  "stand_quota_base": dict(zip(STAND_SOURCE_ORDER, (605, 17, 17, 17, 17))),
  "stand_subtract_one_order": list(STAND_SOURCE_ORDER),
  "stand_rotation_period": 5,
  "stand_five_epoch_totals": dict(zip(STAND_SOURCE_ORDER, (3024, 84, 84, 84, 84))),
  "shoulder_source_quotas": {"1": {"ik": 672, "personal": 672},
                             "2": {"ik": 672, "personal": 672}},
  "ik_amplitude_bins": 11,
  "legacy_hierarchy": EVALUATION_SAMPLING["legacy_hierarchy"],
  "personal_hierarchy": "clip, window",
  "seed": 42,
  "proportions_apply_before_null_dropout": True,
  "normalization_policy": "reuse_EXP036_train_only_statistics_byte_for_byte",
  "evaluation_policy": "unchanged_EXP036_weights_and_negative_windows",
}


def epoch_quotas(epoch: int) -> dict[int, dict[str, int]]:
  if type(epoch) is not int or epoch < 0:
    raise ValueError("Invalid quota epoch")
  stand = dict(SAMPLING["stand_quota_base"])
  stand[STAND_SOURCE_ORDER[epoch % 5]] -= 1
  return {0: stand, 1: {"ik": 672, "personal": 672},
          2: {"ik": 672, "personal": 672}}


def training_condition_weights(rows, condition):
  """Change source mass only; within-source actor/clip/window weights stay fixed."""
  weights = condition_weights(rows, condition)
  if condition == 0:
    for source, mass in STAND_MASS.items():
      mask = np.array([int(r["condition_id"]) == 0 and r["dataset"] == source
                       for r in rows])
      original_mass = weights[mask].sum()
      if original_mass <= 0:
        raise ValueError(f"Missing required stand training source: {source}")
      weights[mask] *= mass / original_mass
  if not np.isclose(weights.sum(), 1):
    raise ValueError("LAFAN90 conditional weights do not sum to one")
  return weights


def data_sha(root=DATA_DIR):
  return sha256(Path(root) / "manifest.json")


def prepare(root=DATA_DIR):
  """Reference immutable EXP-036 shards without rewriting data or normalization."""
  root = Path(root)
  _verify(PARENT / "manifest.json", PARENT_SHA)
  _verify(PARENT / "norm_stats.npz", NORM_SHA)
  parent = json.loads((PARENT / "manifest.json").read_text())
  if parent["identity"]["policy"] != POLICY:
    raise ValueError("Parent complete-video split policy changed")
  references = {"norm_stats.npz": NORM_SHA,
                "windows.csv": parent["windows_index_sha256"],
                "personal_split.json": parent["personal_split_sha256"]}
  for shard in parent["shards"]:
    references[shard["path"]] = shard["sha256"]
    references[shard["rows"]] = shard["rows_sha256"]
  identity = {
    "experiment": "EXP-038",
    "parent_manifest_sha256": PARENT_SHA,
    "policy": POLICY,
    "sampling": SAMPLING,
    "normalization_sha256": NORM_SHA,
    "normalization_recomputed": False,
    "evaluation_sampling": EVALUATION_SAMPLING,
    "code_sha256": sha256(__file__),
  }
  manifest = copy.deepcopy(parent)
  manifest.update(
    experiment="EXP-038", identity=identity, parent_manifest_sha256=PARENT_SHA,
    training_sampling=SAMPLING, evaluation_sampling=EVALUATION_SAMPLING,
    normalization_recomputed=False, raw_features_changed=False,
    split_changed=False, reference_files=references,
  )
  path = root / "manifest.json"
  if path.exists() and json.loads(path.read_text()) != manifest:
    raise ValueError("Existing LAFAN90 dataset identity mismatch")
  root.mkdir(parents=True, exist_ok=True)
  for name, digest in references.items():
    source, destination = PARENT / name, root / name
    _verify(source, digest)
    if destination.exists() or destination.is_symlink():
      _verify(destination, digest)
    else:
      destination.symlink_to(source.resolve())
  if not path.exists():
    atomic_json(path, manifest)
  return manifest


class Lafan90Dataset(ShoulderMixedP01P02Dataset):
  """Only the training/calibration view gets new source weights."""

  def __init__(self, root=DATA_DIR, split="train", expected_sha=None):
    super().__init__(root, split, expected_sha)
    identity = self.manifest.get("identity", {})
    if (identity.get("parent_manifest_sha256") != PARENT_SHA
        or identity.get("sampling") != SAMPLING
        or identity.get("evaluation_sampling") != EVALUATION_SAMPLING
        or self.manifest["norm_stats_sha256"] != NORM_SHA):
      raise ValueError("LAFAN90 recipe/normalization identity mismatch")
    if split == "train":
      self._weights = [torch.from_numpy(training_condition_weights(self.rows, c))
                       for c in range(3)]


class Lafan90Sampler(ShoulderMixedSampler):
  def __init__(self, dataset, *, seed=42):
    super().__init__(dataset, seed=seed)
    self.weights = [torch.from_numpy(training_condition_weights(self.rows, c))
                    for c in range(3)]
    self.last_epoch_audit = None

  def __iter__(self):
    epoch = self.epoch
    generator = torch.Generator().manual_seed(self.seed + epoch)
    quotas = epoch_quotas(epoch)
    draws, observed = [], {}
    for condition, sources in quotas.items():
      observed[str(condition)] = {}
      for source, quota in sources.items():
        pools = self.pools[condition, source]
        base, remainder = divmod(quota, len(pools))
        extras = {(epoch + i) % len(pools) for i in range(remainder)}
        source_draws = []
        for b, ids in enumerate(pools):
          count = base + int(b in extras)
          offsets = torch.multinomial(self.weights[condition][ids], count, True,
                                      generator=generator)
          source_draws.append(ids[offsets])
        selected = torch.cat(source_draws)
        if len(selected) != quota or any(
          int(self.rows[i]["condition_id"]) != condition
          or self.rows[i]["dataset"] != source for i in selected.tolist()
        ):
          raise RuntimeError("LAFAN90 actual source quota mismatch")
        observed[str(condition)][source] = len(selected)
        draws.append(selected)
    indices = torch.cat(draws)
    if len(indices) != self.samples:
      raise RuntimeError("LAFAN90 epoch sample count mismatch")
    self.last_epoch_audit = {"epoch": epoch, "observed": observed,
                             "before_null_dropout": True}
    self.epoch += 1
    return iter(indices[torch.randperm(len(indices), generator=generator)].tolist())

  def state_dict(self):
    return {"seed": self.seed, "samples": self.samples, "epoch": self.epoch,
            "sampling": copy.deepcopy(SAMPLING),
            "stand_rotation_offset": self.epoch % 5,
            "ik_amplitude_offset": self.epoch % 11}

  def load_state_dict(self, state):
    epoch = state.get("epoch")
    if type(epoch) is not int or epoch < 0:
      raise ValueError("Invalid LAFAN90 sampler epoch")
    expected = {**self.state_dict(), "epoch": epoch,
                "stand_rotation_offset": epoch % 5,
                "ik_amplitude_offset": epoch % 11}
    if state != expected:
      raise ValueError("LAFAN90 sampler resume identity/rotation mismatch")
    self.epoch = epoch
