"""EXP-038 source weighting, frozen evaluation, and resumable quota rotation."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smp.pretrain.shoulder_mixed import (
  QUOTAS,
  ShoulderMixedSampler,
  condition_weights,
)
from smp.pretrain.shoulder_mixed import (
  SAMPLING as OLD_SAMPLING,
)
from smp.pretrain.shoulder_mixed_lafan90 import (
  NORM_SHA,
  PARENT,
  SAMPLING,
  STAND_MASS,
  STAND_SOURCE_ORDER,
  Lafan90Dataset,
  Lafan90Sampler,
  epoch_quotas,
  prepare,
  training_condition_weights,
)


def toy():
  rows = []
  for condition, sources in QUOTAS.items():
    for source in sources:
      for b in range(11) if source == "ik" and condition else range(1):
        for clip in range(2):
          for window in range(2 + clip):
            rows.append(
              {
                "condition_id": condition,
                "dataset": source,
                "clip_id": f"{condition}/{source}/{b}/{clip}",
                "source_group": f"{source}/{clip}",
                "target_yaw_amplitude_deg": 17.5 + b * 5,
                "frame50_start": window * 5,
              }
            )
  return SimpleNamespace(
    rows=rows, labels=torch.tensor([r["condition_id"] for r in rows])
  )


def test_exact_five_epoch_source_and_condition_quotas():
  dataset = toy()
  sampler = Lafan90Sampler(dataset)
  totals = Counter()
  for epoch in range(5):
    indices = list(sampler)
    observed = Counter(
      (dataset.rows[i]["condition_id"], dataset.rows[i]["dataset"]) for i in indices
    )
    assert observed == {
      (c, source): n
      for c, sources in epoch_quotas(epoch).items()
      for source, n in sources.items()
    }
    assert Counter(dataset.rows[i]["condition_id"] for i in indices) == {
      0: 672,
      1: 1344,
      2: 1344,
    }
    totals.update({source: observed[0, source] for source in STAND_SOURCE_ORDER})
    assert sampler.last_epoch_audit["epoch"] == epoch
  assert totals == {
    "lafan": 3024,
    "ik": 84,
    "personal": 84,
    "g1_moves": 84,
    "amass_g1": 84,
  }


def test_source_weights_preserve_within_source_clip_mass_and_shoulders():
  dataset = toy()
  original = condition_weights(dataset.rows, 0)
  updated = training_condition_weights(dataset.rows, 0)
  for source, target in STAND_MASS.items():
    mask = np.array(
      [r["dataset"] == source and r["condition_id"] == 0 for r in dataset.rows]
    )
    assert updated[mask].sum() == pytest.approx(target)
    np.testing.assert_allclose(
      updated[mask] / updated[mask].sum(), original[mask] / original[mask].sum()
    )
  for c in (1, 2):
    np.testing.assert_array_equal(
      training_condition_weights(dataset.rows, c), condition_weights(dataset.rows, c)
    )


def test_shoulder_amplitude_rotation_and_old_sampler_are_unchanged():
  dataset = toy()
  old, new = ShoulderMixedSampler(dataset), Lafan90Sampler(dataset)
  for _ in range(11):
    old_ids, new_ids = list(old), list(new)
    for c in (1, 2):

      def bins(indices, c=c):
        return Counter(
          int((dataset.rows[i]["target_yaw_amplitude_deg"] - 15) // 5)
          for i in indices
          if dataset.rows[i]["condition_id"] == c and dataset.rows[i]["dataset"] == "ik"
        )

      assert bins(old_ids) == bins(new_ids)
  assert OLD_SAMPLING["source_quotas"]["0"]["lafan"] == 112
  assert QUOTAS[0]["ik"] == QUOTAS[0]["personal"] == 168


def test_resume_restores_both_non_aligned_cycles_and_next_draws():
  dataset = toy()
  sampler = Lafan90Sampler(dataset)
  for _ in range(7):
    list(sampler)
  saved = sampler.state_dict()
  assert saved["stand_rotation_offset"] == 2
  assert saved["ik_amplitude_offset"] == 7
  expected = [list(sampler) for _ in range(6)]
  restored = Lafan90Sampler(dataset)
  restored.load_state_dict(saved)
  assert [list(restored) for _ in range(6)] == expected
  assert list(Lafan90Sampler(dataset, seed=43)) != list(Lafan90Sampler(dataset))
  for field in ("seed", "stand_rotation_offset", "ik_amplitude_offset"):
    invalid = deepcopy(saved)
    invalid[field] += 1
    with pytest.raises(ValueError, match="identity/rotation"):
      restored.load_state_dict(invalid)
  with pytest.raises(ValueError, match="identity/rotation"):
    restored.load_state_dict(ShoulderMixedSampler(dataset).state_dict())


def test_missing_source_is_a_hard_error():
  dataset = toy()
  dataset.rows = [r for r in dataset.rows if r["dataset"] != "g1_moves"]
  with pytest.raises(ValueError, match="stand training source"):
    training_condition_weights(dataset.rows, 0)
  with pytest.raises(ValueError, match="required training pool"):
    Lafan90Sampler(dataset)


@pytest.fixture(scope="module")
def frozen_views(tmp_path_factory):
  if not (PARENT / "manifest.json").exists():
    pytest.skip("Frozen EXP-036 data unavailable")
  root = tmp_path_factory.mktemp("lafan90_frozen_views")
  prepare(root)
  return root, Lafan90Dataset(root, "train"), Lafan90Dataset(root, "validation")


def test_real_data_norm_raw_shards_and_splits_are_frozen(frozen_views):
  from smp.pretrain.conditioning import sha256

  root, train, validation = frozen_views
  assert sha256(root / "norm_stats.npz") == NORM_SHA
  assert train.manifest["normalization_recomputed"] is False
  assert train.manifest["split_changed"] is False
  assert len(train) == 85641 and len(validation) == 19483
  np.testing.assert_array_equal(train.q_low, validation.q_low)
  with np.load(PARENT / "norm_stats.npz") as norm:
    np.testing.assert_array_equal(train.q_low, norm["q_low"])
    np.testing.assert_array_equal(train.q_high, norm["q_high"])
  for row in train.rows + validation.rows:
    assert row["subject_id"] != "P03"
  for shard in train.manifest["shards"]:
    assert sha256(root / shard["path"]) == sha256(PARENT / shard["path"])


def test_validation_weights_and_negative_indices_are_exactly_original(frozen_views):
  _, train, validation = frozen_views
  for c in (0, 1, 2):
    old = torch.from_numpy(condition_weights(validation.rows, c))
    assert torch.equal(validation.clip_equal_weights(c), old)
    expected = torch.multinomial(
      old, 137, True, generator=torch.Generator().manual_seed(42000 + c)
    )
    assert torch.equal(validation.sample_indices(c, 137, 42000 + c), expected)
  for source, target in STAND_MASS.items():
    mask = torch.tensor(
      [r["dataset"] == source and r["condition_id"] == 0 for r in train.rows]
    )
    assert float(train.clip_equal_weights(0)[mask].sum()) == pytest.approx(target)


def test_optimizer_rng_and_sampler_checkpoint_roundtrip():
  import random

  from scripts.pretrain_shoulder_mixed_lafan90 import restore_training

  from smp.pretrain.shoulder_mixed import CONDITION_SCHEMA

  model = torch.nn.Linear(3, 2)
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  sampler = Lafan90Sampler(toy())
  list(sampler)

  def step(m, o):
    o.zero_grad()
    loss = m(torch.randn(4, 3)).square().mean()
    loss.backward()
    o.step()

  for _ in range(4):
    step(model, optimizer)
  payload = deepcopy(
    {
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "sampler": sampler.state_dict(),
      "epoch": 1,
      "optimizer_steps": 4,
      "cfg": {"test": True},
      "condition_schema": CONDITION_SCHEMA,
      "data_sha256": "frozen-data",
      "sampling": SAMPLING,
      "initialization": "scratch",
      "rng": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [],
      },
    }
  )
  expected_random = (random.random(), np.random.rand())
  expected_indices = list(sampler)
  step(model, optimizer)
  recovered = torch.nn.Linear(3, 2)
  recovered_optimizer = torch.optim.AdamW(recovered.parameters(), lr=9e-3)
  recovered_sampler = Lafan90Sampler(toy())
  restore_training(
    payload,
    recovered,
    recovered_optimizer,
    recovered_sampler,
    {"test": True},
    "frozen-data",
  )
  assert (random.random(), np.random.rand()) == expected_random
  assert list(recovered_sampler) == expected_indices
  step(recovered, recovered_optimizer)
  assert all(
    torch.equal(a, b)
    for a, b in zip(model.parameters(), recovered.parameters(), strict=True)
  )
  assert recovered_optimizer.param_groups[0]["lr"] == 3e-4


def test_budget_score_schema_and_null_are_unchanged():
  from scripts import pretrain_shoulder_mixed_lafan90 as new
  from scripts import pretrain_shoulder_mixed_p01p02 as old
  from scripts import run_shoulder_mixed_lafan90_prior as driver

  assert new.FORMAL == old.FORMAL
  assert new.CONDITION_SCHEMA == old.CONDITION_SCHEMA
  assert new.CONDITION_SCHEMA["null_id"] == 3
  assert new.CONDITION_SCHEMA["names"][0] == "stand"
  assert new.calibrate is old.calibrate
  assert new.evaluate_scores is old.evaluate_scores
  assert new.EVALUATION_SAMPLING == old.SAMPLING
  assert "lafan90" in str(driver.OUTPUT)
  assert not hasattr(driver, "run_ppo")


def test_distribution_diagnostic_detects_joint_motion_despite_stationary_root():
  from scripts.assess_shoulder_mixed_lafan90 import summarize

  windows = np.zeros((32, 10, 59), dtype=np.float32)
  windows[..., 2] = 0.8
  windows[..., 3] = windows[..., 8] = 1
  windows[..., 21] = np.arange(10) * 0.01
  metrics = summarize(windows, 1)
  assert metrics["window_max_root_planar_speed_p95_m_s"] == 0
  assert metrics["window_max_joint_speed_p95_rad_s"] == pytest.approx(0.5)
  assert metrics["window_max_waist_speed_p95_rad_s"] == pytest.approx(0.5)
  assert metrics["expected_direction_rate"] == 1
  assert summarize(windows, 2)["expected_direction_rate"] == 0
  windows[0, 0, 0] = np.nan
  with pytest.raises(FloatingPointError, match="Invalid generated"):
    summarize(windows, 0)


def test_gsi_probe_checks_prior_identity_and_preserves_quality_failure(
  monkeypatch, tmp_path
):
  import mjlab.envs
  from scripts.assess_shoulder_mixed_lafan90 import gsi_check

  from smp.pretrain.conditioning import sha256
  from smp.pretrain.shoulder_mixed import CONDITION_SCHEMA

  checkpoint = tmp_path / "checkpoint.pt"
  checkpoint.write_bytes(b"probe-only")
  observed = {}
  stats = {"acceptance_rate": 0.8, "nonfinite_rejected": 0}

  class FakeEnv:
    def __init__(self, cfg, device):
      observed.update(params=cfg.events["init_smp_state"].params, device=device)
      self._smp_gsi_safety_stats = dict(stats)
      self._smp_gsi_prior_identity = {
        "sha256": sha256(checkpoint),
        "schema": CONDITION_SCHEMA,
      }

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", FakeEnv)
  result = gsi_check(checkpoint, tmp_path / "calibration.pt")
  assert result["acceptance_rate"] == 0.8 and result["pool_filled"]
  assert observed["closed"] and observed["device"] == "cuda:0"
  assert observed["params"]["ckpt_path"] == str(checkpoint)
  assert observed["params"]["gsi_buffer_size"] == 4096
  assert observed["params"]["gsi_max_root_planar_speed"] == 0.15

  def unable_to_fill(**kwargs):
    raise RuntimeError(
      "Unable to fill the safe GSI pool within the sampling budget: "
      "{'acceptance_rate': 0.01, 'nonfinite_rejected': 0}. Increase gsi_max_sampling_multiplier only after checking the prior."
    )

  monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", unable_to_fill)
  assert gsi_check(checkpoint, tmp_path / "calibration.pt")["pool_filled"] is False

  def gpu_fault(**kwargs):
    raise RuntimeError("CUDA failure")

  monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", gpu_fault)
  with pytest.raises(RuntimeError, match="CUDA failure"):
    gsi_check(checkpoint, tmp_path / "calibration.pt")
