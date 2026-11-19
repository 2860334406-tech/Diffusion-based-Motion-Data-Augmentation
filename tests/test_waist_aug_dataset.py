"""Tests for the trajectory-level waist augmentation dataset."""

from __future__ import annotations

import numpy as np

from smp.pretrain.waist_aug_dataset import (
  COND_DIM,
  FEATURE_DIM,
  Clip,
  WaistAugDataset,
  compute_conditions,
  from_38d,
  locate_onset,
  matrix_to_quat_xyzw,
  quat_xyzw_to_matrix,
  quat_xyzw_to_tan_norm,
  resample_to_fps,
  slice_segments,
  tan_norm_to_quat_xyzw,
  to_38d,
  yaw_of_quat_xyzw,
)


def _random_quats(n: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  q = rng.normal(size=(n, 4))
  return q / np.linalg.norm(q, axis=1, keepdims=True)


def _identity_quat(t: float) -> np.ndarray:
  return np.stack(
    [np.zeros(t), np.zeros(t), np.sin(t * 0) , np.cos(t * 0)], axis=1
  )


def _clip(n: int = 200, seed: int = 0) -> Clip:
  rng = np.random.default_rng(seed)
  quat = _random_quats(n, seed)
  return Clip(
    root_pos=rng.normal(scale=0.01, size=(n, 3)) + np.array([0.0, 0.0, 0.78]),
    root_quat=quat,
    joints=rng.normal(scale=0.05, size=(n, 29)),
    fps=30.0,
    source="synthetic",
  )


def test_quat_matrix_roundtrip():
  q = _random_quats(500, 3)
  q2 = matrix_to_quat_xyzw(quat_xyzw_to_matrix(q))
  err = np.minimum(np.abs(q2 - q).max(1), np.abs(q2 + q).max(1))
  assert err.max() < 1e-6


def test_tan_norm_roundtrip():
  q = _random_quats(500, 4)
  q2 = tan_norm_to_quat_xyzw(quat_xyzw_to_tan_norm(q))
  err = np.minimum(np.abs(q2 - q).max(1), np.abs(q2 + q).max(1))
  assert err.max() < 1e-6


def test_tan_norm_is_orthonormal_columns():
  v6 = quat_xyzw_to_tan_norm(_random_quats(100, 5))
  a, b = v6[:, :3], v6[:, 3:]
  assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
  assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-6)
  assert np.allclose(np.sum(a * b, axis=1), 0.0, atol=1e-6)


def test_to_38d_shape_and_anchoring():
  clip = _clip(120)
  x = to_38d(clip)
  assert x.shape == (120, FEATURE_DIM)
  # First frame sits at the local origin in xy.
  assert np.allclose(x[0, :2], 0.0, atol=1e-6)
  # z keeps world height.
  assert np.isclose(x[0, 2], clip.root_pos[0, 2])


def test_from_38d_recovers_joints():
  clip = _clip(80)
  x = to_38d(clip)
  _, _, joints = from_38d(x)
  assert np.allclose(joints, clip.joints, atol=1e-5)


def test_from_38d_recovers_relative_rotation():
  clip = _clip(60)
  x = to_38d(clip)
  _, quat, _ = from_38d(x)
  # The anchor is yaw-only: the first frame keeps its pitch/roll but has zero yaw.
  assert abs(float(yaw_of_quat_xyzw(quat[:1])[0])) < 1e-5
  # Round-trip through 38D twice must be stable.
  x2 = to_38d(Clip(clip.root_pos, quat, from_38d(x)[2], clip.fps))
  assert np.allclose(x[:, 3:9], x2[:, 3:9], atol=1e-5)


def test_locate_onset_on_step():
  t = np.arange(200)
  yaw = np.where(t < 100, 0.0, 0.3)
  onset, baseline = locate_onset(yaw)
  assert abs(baseline) < 1e-9
  assert 95 <= onset <= 102


def test_locate_onset_falls_back():
  yaw = np.zeros(50)
  onset, _ = locate_onset(yaw)
  assert 0 <= onset < len(yaw)


def test_slice_segments_count():
  x = np.zeros((400, FEATURE_DIM), dtype=np.float32)
  segs = slice_segments(x, window=160, stride=32, onset=100, n_onset=2)
  assert segs.shape[1:] == (160, FEATURE_DIM)
  assert segs.shape[0] >= len(range(0, 400 - 160 + 1, 32))


def test_slice_pads_short_clip():
  x = np.ones((40, FEATURE_DIM), dtype=np.float32)
  segs = slice_segments(x, window=160, stride=32, onset=10)
  assert segs.shape == (1, 160, FEATURE_DIM)


def test_compute_conditions_dim_and_range():
  clip = _clip(200, seed=7)
  x = to_38d(clip)
  c = compute_conditions(x)
  assert c.shape == (COND_DIM,)
  assert np.all(np.isfinite(c))
  assert np.all(np.abs(c) <= 3.0)


def test_compute_conditions_tracks_waist_amplitude():
  clip = _clip(200, seed=8)
  joints = clip.joints.copy()
  waist = np.zeros(200)
  waist[80:] = 0.6  # 34 degrees
  joints[:, 12] = waist
  c = compute_conditions(to_38d(Clip(clip.root_pos, clip.root_quat, joints)))
  assert c[0] > 0.3          # amp / 1.22
  assert c[2] == 1.0         # positive sign


def test_resample_preserves_duration():
  clip = _clip(101, seed=9)
  clip.fps = 50.0
  out = resample_to_fps(clip, 30.0)
  assert abs(out.joints.shape[0] - 61) <= 1
  assert out.fps == 30.0
  # Endpoints are preserved.
  assert np.allclose(out.joints[0], clip.joints[0], atol=1e-5)
  assert np.allclose(out.joints[-1], clip.joints[-1], atol=1e-5)


def test_dataset_shapes():
  ds = WaistAugDataset([_clip(300, seed=10), _clip(300, seed=11)], window=96, stride=32)
  x, c = ds[0]
  assert tuple(x.shape) == (96, FEATURE_DIM)
  assert tuple(c.shape) == (COND_DIM,)
  assert len(ds) == ds.windows.shape[0]


def test_norm_stats_span_nonzero():
  ds = WaistAugDataset([_clip(200, seed=12)], window=96, stride=48)
  lo, hi = ds.compute_norm_stats()
  assert np.all(hi - lo > 0)