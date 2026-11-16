from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scripts.csv_to_npz import _compute_windows

from smp.data.shoulder_ik import (
  FPS,
  JOINT_NAMES,
  MEAN_YAW_SPEED_MAX_DEG_S,
  MEAN_YAW_SPEED_MIN_DEG_S,
  YAW_MAX_DEG,
  YAW_MIN_DEG,
  ShoulderIkGenerator,
  atomic_json,
  classify_condition_windows,
  make_specs,
  minimum_jerk,
  retry_spec,
  split_counts,
)
from smp.data.shoulder_ik_dataset import _angular_velocity, compute_window_features


def test_minimum_jerk_endpoints_and_monotonicity():
  x = np.linspace(0.0, 1.0, 101)
  y = minimum_jerk(x)
  assert y[0] == 0.0
  assert y[-1] == 1.0
  assert np.all(np.diff(y) >= 0.0)
  assert abs(y[1] - y[0]) < 1e-4
  assert abs(y[-1] - y[-2]) < 1e-4


def test_atomic_json_normalizes_numpy_metrics(tmp_path):
  path = tmp_path / "metrics.json"
  atomic_json(path, {"passed": np.bool_(True), "count": np.int64(4)})
  assert json.loads(path.read_text()) == {"passed": True, "count": 4}


def test_exact_pilot_and_formal_quotas():
  pilot = make_specs("pilot")
  formal = make_specs("formal")
  assert len(pilot) == 240
  assert split_counts(pilot) == {"pilot": [48, 96, 96]}
  assert len(formal) == 2400
  assert split_counts(formal) == {
    "test": [72, 144, 144],
    "train": [336, 672, 672],
    "validation": [72, 144, 144],
  }


def test_continuous_latin_hypercube_covers_every_posture_and_split():
  shoulder = [spec for spec in make_specs("formal", seed=42) if spec.condition_id == 1]
  expected = {"train": 112, "validation": 24, "test": 24}
  for cell in range(6):
    for split, count in expected.items():
      group = [
        spec for spec in shoulder if spec.grid_index == cell and spec.split == split
      ]
      assert len(group) == count
      amplitudes = np.asarray([np.degrees(spec.yaw_amplitude) for spec in group])
      speeds = np.asarray([spec.target_mean_yaw_speed_deg_s for spec in group])
      assert np.all((amplitudes >= YAW_MIN_DEG) & (amplitudes <= YAW_MAX_DEG))
      assert np.all(
        (speeds >= MEAN_YAW_SPEED_MIN_DEG_S) & (speeds <= MEAN_YAW_SPEED_MAX_DEG_S)
      )
      amplitude_strata = np.floor(
        (amplitudes - YAW_MIN_DEG) / (YAW_MAX_DEG - YAW_MIN_DEG) * count
      ).astype(int)
      speed_strata = np.floor(
        (speeds - MEAN_YAW_SPEED_MIN_DEG_S)
        / (MEAN_YAW_SPEED_MAX_DEG_S - MEAN_YAW_SPEED_MIN_DEG_S)
        * count
      ).astype(int)
      assert sorted(amplitude_strata.tolist()) == list(range(count))
      assert sorted(speed_strata.tolist()) == list(range(count))
      assert not np.array_equal(amplitude_strata, speed_strata)


def test_continuous_sampling_is_reproducible_seeded_and_frame_aligned():
  first = [spec for spec in make_specs("pilot", seed=42) if spec.condition_id == 1]
  repeated = [spec for spec in make_specs("pilot", seed=42) if spec.condition_id == 1]
  changed = [spec for spec in make_specs("pilot", seed=43) if spec.condition_id == 1]
  assert [asdict(spec) for spec in first] == [asdict(spec) for spec in repeated]
  assert [spec.yaw_amplitude for spec in first] != [
    spec.yaw_amplitude for spec in changed
  ]
  assert len({spec.yaw_amplitude for spec in first}) == len(first)
  for spec in first:
    target_deg = np.degrees(spec.yaw_amplitude)
    expected_frames = int(np.ceil(target_deg / spec.target_mean_yaw_speed_deg_s * FPS))
    assert spec.entry_duration_s == expected_frames / FPS
    assert round(spec.entry_duration_s * FPS) == expected_frames


def test_side_pairs_share_family_and_split_without_copying_identity():
  shoulder = [spec for spec in make_specs("formal") if spec.condition_id]
  families = {}
  for spec in shoulder:
    families.setdefault((spec.family_id, spec.slot_index), []).append(spec)
  assert len(families) == 960
  for pair in families.values():
    assert sorted(spec.condition_id for spec in pair) == [1, 2]
    assert len({spec.split for spec in pair}) == 1
    assert len({spec.clip_id for spec in pair}) == 2


def test_retry_is_deterministic_and_preserves_slot_identity():
  original = make_specs("pilot", seed=42)[100]
  first = retry_spec(original)
  second = retry_spec(original)
  assert asdict(first) == asdict(second)
  assert first.family_id == original.family_id
  assert first.condition_id == original.condition_id
  assert first.split == original.split
  assert first.attempt == 1
  assert first.clip_id != original.clip_id
  assert first.yaw_amplitude == original.yaw_amplitude
  assert first.target_mean_yaw_speed_deg_s == original.target_mean_yaw_speed_deg_s
  assert first.entry_duration_s == original.entry_duration_s


def _motion(sign: float) -> np.ndarray:
  frames = 160
  motion = np.zeros((frames, 36), dtype=np.float64)
  motion[:, 2] = 0.76
  motion[:, 6] = 1.0  # xyzw identity quaternion.
  entry = 50
  duration = 50
  progress = np.zeros(frames)
  progress[entry : entry + duration] = minimum_jerk(
    np.arange(1, duration + 1) / duration
  )
  progress[entry + duration :] = 1.0
  waist_column = 7 + JOINT_NAMES.index("waist_yaw_joint")
  motion[:, waist_column] = sign * 0.20 * progress
  return motion


def test_condition_classifier_uses_clip_timing_not_motion_phases():
  starts, left = classify_condition_windows(_motion(1.0), 1)
  _, right = classify_condition_windows(_motion(-1.0), 2)
  assert len(starts) == len(left) == len(right)
  assert set(left) == {-1, 0, 1}
  assert set(right) == {-1, 0, 2}
  assert np.all(left[starts + 10 <= 50] == 0)
  assert np.all(left[starts >= 50] == 1)
  assert np.all(right[starts >= 50] == 2)


def test_stand_condition_accepts_all_finite_windows():
  static = _motion(0.0)
  starts, labels = classify_condition_windows(static, 0)
  assert len(starts) == (len(static) - 10) // 5 + 1
  assert np.all(labels == 0)


def test_joint_quaternion_and_59_feature_contract_matches_existing_converter():
  assert len(JOINT_NAMES) == 29
  assert JOINT_NAMES.index("waist_yaw_joint") == 12
  frames = 40
  time = np.arange(frames) / 50.0
  root = np.column_stack(
    (0.02 * time, 0.01 * np.sin(time), 0.76 + 0.001 * np.cos(time))
  )
  quaternion = Rotation.from_rotvec(
    np.column_stack((np.zeros(frames), np.zeros(frames), 0.1 * time))
  ).as_quat()
  joints = np.tile(np.linspace(-0.2, 0.2, 29), (frames, 1))
  joints += 0.01 * np.sin(time[:, None] * np.arange(1, 30))
  offsets = np.asarray(
    [
      [0.0, 0.12, -0.7],
      [0.0, -0.12, -0.7],
      [0.0, 0.0, 0.25],
      [0.1, 0.3, 0.0],
      [0.1, -0.3, 0.0],
    ]
  )
  rotation = Rotation.from_quat(quaternion).as_matrix()
  ee = root[:, None] + np.einsum("tij,ej->tei", rotation, offsets)
  starts, actual = compute_window_features(root, quaternion, joints, ee)
  velocity = np.gradient(root, 1 / 50, axis=0)
  angular = _angular_velocity(quaternion)
  established = _compute_windows(
    *[
      torch.tensor(value, dtype=torch.float32)
      for value in (
        root,
        quaternion[:, [3, 0, 1, 2]],
        velocity,
        angular,
        ee,
        joints,
      )
    ],
    10,
    5,
  ).numpy()
  assert np.array_equal(starts, np.arange(0, frames - 9, 5))
  assert actual.shape == (len(starts), 10, 59)
  np.testing.assert_allclose(actual, established, atol=3e-7, rtol=1e-6)


def test_whole_body_ik_and_free_root_physics_smoke():
  specs = make_specs("pilot")
  stand = next(spec for spec in specs if spec.condition_id == 0)
  shoulder = min(
    (spec for spec in specs if spec.condition_id == 1),
    key=lambda spec: spec.yaw_amplitude,
  )
  generator = ShoulderIkGenerator()
  for spec in (stand, shoulder):
    passed, motion, execution, metrics = generator.produce(spec)
    assert passed, metrics["failed_checks"]
    assert motion.qpos.shape[1] == 36
    assert execution.qpos.shape == motion.qpos.shape
    assert "phase_windows" not in metrics["checks"]
    if spec.condition_id:
      assert metrics["endpoint_position_reached"]
      assert metrics["completion_time_passed"]
      assert metrics["hold_position_passed"]
      assert metrics["checks"]["waist_amplitude_range"]
      assert metrics["checks"]["waist_amplitude_target"]
      assert 15.0 <= metrics["actual_waist_amplitude_deg"] <= 70.0
    assert metrics["left_contact_fraction"] >= 0.95
    assert metrics["right_contact_fraction"] >= 0.95
    assert metrics["root_height_min_m"] >= 0.60
    assert not metrics["self_collision"]
    assert not metrics["nonfoot_ground"]
  assert shoulder.condition_id == 1
  assert metrics["shoulder_retreat_final_m"] >= 0.015
  assert metrics["shoulder_completion_time_s"] <= shoulder.entry_duration_s
  assert metrics["hold_shoulder_position_max_deviation_m"] <= 0.002
