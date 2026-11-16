from __future__ import annotations

import numpy as np
from scripts.render_g1_shoulder_ik_review import select_records
from scripts.run_g1_shoulder_ik_data import Cfg, amplitude_distribution, specs_sha256
from scripts.smoke_g1_shoulder_ik_v2 import Cfg as SmokeCfg
from scripts.smoke_g1_shoulder_ik_v2 import make_boundary_specs

from smp.data.shoulder_ik import FPS, make_specs


def _record(spec):
  target = float(np.degrees(spec.yaw_amplitude))
  return {
    "status": "accepted",
    "accepted": {
      "clip_id": spec.clip_id,
      "family_id": spec.family_id,
      "split": spec.split,
      "condition_id": spec.condition_id,
      "target_yaw_amplitude_deg": target,
      "actual_waist_amplitude_deg": target,
      "target_mean_yaw_speed_deg_s": spec.target_mean_yaw_speed_deg_s,
      "entry_duration_s": spec.entry_duration_s,
    },
  }


def test_distribution_gate_requires_every_five_degree_bin():
  specs = make_specs("pilot", seed=42)
  records = [_record(spec) for spec in specs]
  report = amplitude_distribution(records, Cfg())
  assert report["passed"]
  assert all(group["passed"] for group in report["groups"].values())
  missing = [
    record
    for record in records
    if not (
      record["accepted"]["condition_id"] == 1
      and 30.0 <= record["accepted"]["actual_waist_amplitude_deg"] < 35.0
    )
  ]
  failed = amplitude_distribution(missing, Cfg())
  assert not failed["passed"]
  assert not failed["groups"]["pilot/left_shoulder"]["passed"]


def test_specs_digest_is_deterministic_and_seed_sensitive():
  assert specs_sha256(make_specs("pilot", 42)) == specs_sha256(make_specs("pilot", 42))
  assert specs_sha256(make_specs("pilot", 42)) != specs_sha256(make_specs("pilot", 43))


def test_visual_review_is_balanced_across_amplitude_quartiles():
  records = [_record(spec) for spec in make_specs("pilot", seed=42)]
  manifest = {
    "identity": {"stage": "pilot"},
    "slots": {str(index): record for index, record in enumerate(records)},
  }
  selected = select_records(manifest, 60)
  assert [sum(item["condition_id"] == c for item in selected) for c in range(3)] == [
    12,
    24,
    24,
  ]
  for condition in (1, 2):
    amplitudes = [
      item["target_yaw_amplitude_deg"]
      for item in selected
      if item["condition_id"] == condition
    ]
    quartiles = [min(3, int((value - 15.0) / 13.75)) for value in amplitudes]
    assert np.bincount(quartiles, minlength=4).tolist() == [6, 6, 6, 6]


def test_boundary_smoke_has_exact_angles_sides_and_frame_aligned_time():
  specs = make_boundary_specs(SmokeCfg())
  assert len(specs) == 6
  for amplitude in (15.0, 42.5, 70.0):
    pair = [
      spec for spec in specs if np.isclose(np.degrees(spec.yaw_amplitude), amplitude)
    ]
    assert sorted(spec.condition_id for spec in pair) == [1, 2]
    for spec in pair:
      assert spec.entry_duration_s == np.ceil(amplitude / 15.0 * FPS) / FPS
