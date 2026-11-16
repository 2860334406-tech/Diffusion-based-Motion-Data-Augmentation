"""Complete-video split and unchanged prior/score recipe for EXP-036."""

from collections import Counter

import pytest

from smp.pretrain.shoulder_mixed_p01p02_dataset import (
  PARENT,
  POLICY,
  ShoulderMixedP01P02Dataset,
  split_personal_clips,
  validate_split_records,
)


def clips():
  result = []
  for subject in ("P01", "P02", "P03"):
    for condition, name in enumerate(("stand", "left_shoulder", "right_shoulder")):
      for take in range(1, (2 if subject == "P02" and condition == 2 else 3) + 1):
        clip = f"{subject}_session01_{name}_take{take:03d}"
        result.append(
          {
            "clip_id": clip,
            "subject_id": subject,
            "condition_id": condition,
            "split": {"P01": "train", "P02": "validation", "P03": "test"}[subject],
            "source_identity": {"video_sha256": "video/" + clip},
          }
        )
  return result


def test_seed42_split_is_stable_and_independent_of_inventory_order():
  expected = {
    "P02_session01_stand_take001",
    "P02_session01_left_shoulder_take003",
    "P02_session01_right_shoulder_take002",
  }
  records = split_personal_clips(clips())
  assert {r["clip_id"] for r in records if r["split"] == "validation"} == expected
  assert records == split_personal_clips(list(reversed(clips())))
  assert Counter(r["split"] for r in records) == {
    "train": 14,
    "validation": 3,
    "test": 9,
  }


def test_both_participants_train_in_all_three_conditions():
  records = split_personal_clips(clips())
  train = [r for r in records if r["split"] == "train"]
  assert Counter(r["subject_id"] for r in train) == {"P01": 9, "P02": 5}
  assert Counter(r["condition_id"] for r in train) == {0: 5, 1: 5, 2: 4}
  assert all(
    any(r["subject_id"] == subject and r["condition_id"] == condition for r in train)
    for subject in ("P01", "P02")
    for condition in range(3)
  )
  assert POLICY["cross_subject_generalization_claim"] is False


def test_p03_is_reserved_and_original_subject_identifiers_are_preserved():
  records = split_personal_clips(clips())
  assert all(r["split"] == "test" for r in records if r["subject_id"] == "P03")
  assert all(r["clip_id"].startswith(r["subject_id"] + "_") for r in records)
  assert len({r["clip_id"] for r in records}) == 26


def test_alternate_seed_changes_holdout_without_changing_budget():
  first = split_personal_clips(clips(), seed=42)
  other = split_personal_clips(clips(), seed=43)
  assert {r["clip_id"] for r in first if r["split"] == "validation"} != {
    r["clip_id"] for r in other if r["split"] == "validation"
  }
  assert Counter(r["split"] for r in other) == {"train": 14, "validation": 3, "test": 9}


def test_same_original_video_cannot_cross_train_validation():
  records = split_personal_clips(clips())
  train = next(r for r in records if r["split"] == "train")
  validation = next(r for r in records if r["split"] == "validation")
  validation["source_video_sha256"] = train["source_video_sha256"]
  with pytest.raises(ValueError, match="video crosses"):
    validate_split_records(records)


def test_p03_cannot_enter_training():
  records = split_personal_clips(clips())
  next(r for r in records if r["subject_id"] == "P03")["split"] = "train"
  with pytest.raises(ValueError, match="independently reserved"):
    validate_split_records(records)


def test_unaccepted_or_missing_clip_cannot_silently_change_inventory():
  extra = clips()
  extra.append(
    {
      **extra[-1],
      "clip_id": "P02_session01_right_shoulder_take003",
      "subject_id": "P02",
      "condition_id": 2,
    }
  )
  with pytest.raises(ValueError, match="inventory"):
    split_personal_clips(extra)
  with pytest.raises(ValueError, match="inventory"):
    split_personal_clips(clips()[:-1])


def test_legacy_dataset_is_rejected_before_loading_arrays():
  if not (PARENT / "manifest.json").exists():
    pytest.skip("Historical data is unavailable")
  with pytest.raises(ValueError, match="policy identity"):
    ShoulderMixedP01P02Dataset(root=PARENT)


def test_only_data_split_changes_not_network_budget_or_score():
  from scripts import pretrain_shoulder_mixed as old
  from scripts import pretrain_shoulder_mixed_p01p02 as new
  from scripts import run_shoulder_mixed_p01p02_prior as driver

  assert new.FORMAL == old.FORMAL
  assert new.CONDITION_SCHEMA == old.CONDITION_SCHEMA
  assert new.SAMPLING == old.SAMPLING
  assert new.calibrate is old.calibrate
  assert new.evaluate_scores is old.evaluate_scores
  assert "p01p02" in str(new.DATA_DIR)
  assert not hasattr(driver, "run_ppo")
  assert driver.TERMINAL == {
    "validation_candidate_selected",
    "training_complete_no_score_candidate",
  }
