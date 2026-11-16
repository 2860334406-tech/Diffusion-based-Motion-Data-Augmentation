from __future__ import annotations

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smp.data.personal_dataset import publication_gate, window_labels
from smp.data.personal_retarget import (
  Config,
  PersonalRetargeter,
  SimulationFailure,
  SupportLimit,
  collision_limit,
  config_dict,
  coverage,
  detect_onset,
  heading_retreat,
  resample,
  stable_terminal_target,
  verify_attempt,
)
from smp.data.shoulder_ik import _build_model, _qpos_ids, atomic_json, sha256
from smp.data.shoulder_ik_dataset import ExecutedFk, compute_window_features


def test_heading_retreat_keeps_pelvis_rotation_and_removes_root_translation():
  # Rotating a left shoulder around a stationary pelvis is valid even when the
  # waist itself is neutral. Translating the root cannot counterfeit it.
  shoulder = np.array([[0.0, 0.2, 0.3], [-0.1, 0.1732, 0.3]])
  pelvis = np.zeros((2, 3))
  world, compensated = heading_retreat(shoulder, pelvis, 0)
  assert np.isclose(world[-1], 0.1) and np.isclose(compensated[-1], 0.1)
  shift = np.array([[0.0, 0, 0], [-0.1, 0, 0]])
  world, compensated = heading_retreat(shoulder[:1].repeat(2, 0) + shift, shift, 0)
  assert np.isclose(world[-1], 0.1) and np.isclose(compensated[-1], 0)
  yaw = 0.7
  rotation = Rotation.from_rotvec([0, 0, yaw])
  w, c = heading_retreat(rotation.apply(shoulder), rotation.apply(pelvis), yaw)
  np.testing.assert_allclose(w, [0.0, 0.1], atol=1e-7)
  np.testing.assert_allclose(c, w)


def test_arm_motion_cannot_move_torso_attached_shoulder_landmark():
  generator = PersonalRetargeter()
  model = generator.sim_model
  q = model.key_qpos[model.key("init_state").id].copy()
  poses = np.repeat(q[None], 3, axis=0)
  poses[1, model.jnt_qposadr[model.joint("left_shoulder_pitch_joint").id]] += 0.8
  poses[2, model.jnt_qposadr[model.joint("left_shoulder_roll_joint").id]] += 0.4
  _, shoulder, _ = generator.landmarks(poses)
  np.testing.assert_allclose(shoulder, np.repeat(shoulder[:1], 3, axis=0), atol=1e-10)


def test_full_collision_avoidance_includes_hands_and_hips():
  model = _build_model(soft_limits=False)
  limit = collision_limit(model, 0.003)
  actual = {tuple(sorted(pair)) for pair in limit.geom_id_pairs}
  for hand in ("left_hand_collision", "right_hand_collision"):
    for hip in ("left_hip_collision", "right_hip_collision"):
      pair = tuple(sorted((model.geom(hand).id, model.geom(hip).id)))
      assert pair in actual


def test_support_limit_linearization_enforces_inset():
  model = _build_model(soft_limits=False)
  import mink

  configuration = mink.Configuration(model)
  configuration.update_from_keyframe("init_state")
  normals = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
  bounds = np.full(4, 0.2)
  limit = SupportLimit(normals, bounds, 0.01)
  result = limit.compute_qp_inequalities(configuration, 0.005)
  assert result.G.shape == (4, model.nv)
  np.testing.assert_allclose(
    result.h, bounds - 0.01 - normals @ configuration.data.subtree_com[1, :2]
  )


def test_pelvis_has_no_extra_amplitude_limit_and_subjects_stay_separate():
  cfg = config_dict(Config())
  assert cfg["pelvis_yaw_limit"] is None
  assert cfg["minimum_waist_amplitude"] is None
  assert cfg["subject_splits"] == {"P01": "train", "P02": "validation", "P03": "test"}
  assert len({tuple(x) for x in cfg["stances"]}) == 6
  assert config_dict(Config()) == config_dict(Config())


def test_source_resampling_preserves_rotation_and_never_extrapolates():
  t = np.arange(91) / 30
  root = np.column_stack((0.01 * t, np.zeros_like(t), 0.75 + 0.001 * t))
  quat = Rotation.from_rotvec(np.column_stack((t * 0, t * 0, 0.6 * t))).as_quat()
  joints = np.repeat((0.1 * t)[:, None], 29, axis=1)
  p, q, j = resample(root, quat, joints, 30)
  assert p.shape == (151, 3) and j.shape == (151, 29)
  np.testing.assert_allclose(p[[0, -1]], root[[0, -1]])
  np.testing.assert_allclose(
    Rotation.from_quat(q[-1]).as_matrix(), Rotation.from_quat(quat[-1]).as_matrix()
  )
  assert np.all(np.sum(q[1:] * q[:-1], axis=1) >= 0)


def test_onset_ignores_small_pre_turn_jitter():
  yaw = np.zeros(500)
  yaw[80:100] = 0.021
  yaw[200:350] = np.linspace(0, 1, 150)
  yaw[350:] = 1
  onset = detect_onset(yaw, Config())
  assert 200 < onset < 220
  with pytest.raises(ValueError):
    detect_onset(np.zeros(200), Config())


def test_terminal_target_holds_observed_pose_without_modifying_source_or_timing():
  values = np.zeros((400, 3))
  values[100:250, 2] = np.linspace(0, 0.8, 150)
  values[250:350, 2] = 0.8
  values[350:, 2] = np.linspace(0.8, 0.6, 50)
  original = values.copy()
  result = stable_terminal_target(values, Config())
  np.testing.assert_array_equal(values, original)
  np.testing.assert_array_equal(result[:300], original[:300])
  np.testing.assert_allclose(result[-50:, 2], 0.8)
  assert result.shape == values.shape
  assert np.max(result[:, 2]) <= np.max(original[:, 2]) + 1e-14


def test_window_labels_do_not_mix_boundaries_or_call_non_neutral_pose_stand():
  root = np.zeros((160, 3))
  root[:, 2] = 0.75
  quat = np.zeros((160, 4))
  quat[:, 3] = 1
  joints = np.zeros((160, 29))
  joints[50:, 12] = 0.2
  starts, labels = window_labels(root, quat, joints, 1, 50)
  assert np.all(labels[starts + 10 <= 50] == 0)
  assert np.all(labels[starts >= 50] == 1)
  assert np.all(labels[(starts < 50) & (starts + 10 > 50)] == -1)
  joints[:50, 12] = 0.2
  _, labels = window_labels(root, quat, joints, 1, 50)
  assert np.all(labels[starts < 50] == -1)


def test_coverage_requires_every_subject_and_condition_and_review():
  records, reviews = {}, {"clips": {}}
  for subject in ("P01", "P02", "P03"):
    for condition in range(3):
      clip = f"{subject}_{condition}"
      records[clip] = {
        "subject_id": subject,
        "condition_id": condition,
        "status": "accepted",
      }
      reviews["clips"][clip] = {"decision": "pass"}
  assert coverage(records)["passed"]
  assert not publication_gate(records, reviews)[
    "passed"
  ]  # 27 inputs must be processed.
  for index in range(18):
    records[f"rejected{index}"] = {
      "subject_id": "P01",
      "condition_id": 0,
      "status": "rejected",
    }
  assert publication_gate(records, reviews)["passed"]
  reviews["clips"]["P03_2"]["decision"] = "pending"
  assert not publication_gate(records, reviews)["passed"]
  records["P03_2"]["status"] = "rejected"
  assert not coverage(records)["passed"]


def test_resume_rejects_artifact_tampering(tmp_path):
  p = tmp_path / "execution.npz"
  np.savez(p, data=np.arange(3))
  metadata = tmp_path / "metadata.json"
  atomic_json(
    metadata, {"files": {"execution.npz": {"path": p.name, "sha256": sha256(p)}}}
  )
  record = {"metadata": metadata.name, "metadata_sha256": sha256(metadata)}
  assert verify_attempt(tmp_path, record)["files"]
  np.savez(p, data=np.arange(4))
  with pytest.raises(ValueError, match="artifact changed"):
    verify_attempt(tmp_path, record)


@pytest.mark.parametrize("mode", ["warning", "nonfinite"])
def test_simulation_failure_does_not_accept_finite_auto_reset(monkeypatch, mode):
  generator = PersonalRetargeter()
  model = generator.sim_model
  reference = model.key_qpos[model.key("init_state").id][None].copy()

  def broken_step(data, target, maps):
    if mode == "warning":
      data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1
    else:
      data.qvel[0] = np.nan
    return np.zeros(model.nu)

  monkeypatch.setattr(generator, "_track_step", broken_step)
  with pytest.raises(SimulationFailure):
    generator.execute_checked(reference)


def test_personal_59d_features_keep_measured_pelvis_rotation():
  model = _build_model(soft_limits=False)
  q = np.repeat(model.key_qpos[model.key("init_state").id][None], 30, axis=0)
  yaw = np.linspace(0, 0.3, len(q))
  q[:, 3:7] = Rotation.from_rotvec(np.column_stack((yaw * 0, yaw * 0, yaw))).as_quat()[
    :, [3, 0, 1, 2]
  ]
  root = q[:, :3]
  quat = q[:, [4, 5, 6, 3]]
  joints = q[:, _qpos_ids(model)]
  fk = ExecutedFk()
  _, windows = compute_window_features(
    root, quat, joints, fk.positions(root, quat, joints)
  )
  assert windows.shape == (5, 10, 59) and np.isfinite(windows).all()
  assert np.abs(windows[:, :, 58]).mean() > 0.1
