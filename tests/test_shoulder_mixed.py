"""EXP-035 source mass, normalization, recovery, score and response regressions."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts.evaluate_shoulder_mixed import _gate, safety_failure_union
from scripts.probe_shoulder_mixed_online import reward_gate

from smp.pretrain.shoulder_mixed import (
  CONDITION_SCHEMA,
  MIN_SPAN,
  QUOTAS,
  ShoulderMixedSampler,
  condition_weights,
  normalization,
)
from smp.pretrain.shoulder_mixed_score import _metrics
from smp.rl.tasks.force_unloading import conditional_shoulder_mixed as task


def toy():
  rows = []
  for condition in range(3):
    for source in QUOTAS[condition]:
      bins = range(11) if source == "ik" and condition else range(1)
      for b in bins:
        for clip in range(2):
          for window in range(3):
            rows.append(
              {
                "condition_id": condition,
                "dataset": source,
                "clip_id": f"{condition}/{source}/{b}/{clip}",
                "source_group": f"{source}/{clip}",
                "target_yaw_amplitude_deg": 17.5 + 5 * b,
                "frame50_start": 5 * window,
              }
            )
  d = SimpleNamespace(rows=rows, labels=torch.tensor([r["condition_id"] for r in rows]))
  d.clip_equal_weights = lambda c: torch.from_numpy(condition_weights(rows, c))
  return d


def test_exact_source_condition_and_bin_quotas():
  d = toy()
  sampler = ShoulderMixedSampler(d)
  ids = list(sampler)
  observed = Counter((int(d.labels[i]), d.rows[i]["dataset"]) for i in ids)
  assert observed == {
    (c, s): n for c, sources in QUOTAS.items() for s, n in sources.items()
  }
  for c in (1, 2):
    counts = Counter(
      int((d.rows[i]["target_yaw_amplitude_deg"] - 15) // 5)
      for i in ids
      if int(d.labels[i]) == c and d.rows[i]["dataset"] == "ik"
    )
    assert sorted(counts.values()) == [61] * 10 + [62]


def test_sampler_restore_reproduces_next_epoch():
  d = toy()
  a, b = ShoulderMixedSampler(d), ShoulderMixedSampler(d)
  assert list(a) == list(b)
  saved = a.state_dict()
  expected = list(a)
  recovered = ShoulderMixedSampler(d)
  recovered.load_state_dict(saved)
  assert list(recovered) == expected
  assert list(ShoulderMixedSampler(d, seed=43)) != list(ShoulderMixedSampler(d))
  with pytest.raises(ValueError, match="identity"):
    recovered.load_state_dict({**saved, "seed": 8})


def test_conditional_source_mass_and_personal_clip_balance():
  d = toy()
  for c in range(3):
    w = condition_weights(d.rows, c)
    assert w.sum() == pytest.approx(1)
    for source, expected in (
      ("ik", 0.25 if c == 0 else 0.5),
      ("personal", 0.25 if c == 0 else 0.5),
    ):
      mask = [r["dataset"] == source for r in d.rows]
      assert w[mask].sum() == pytest.approx(expected)
  # Repeating one clip's windows must not increase that clip's mass.
  d.rows.extend([dict(d.rows[-1])] * 20)
  w = condition_weights(d.rows, 2)
  personal = {}
  for row, value in zip(d.rows, w, strict=True):
    if row["dataset"] == "personal" and row["condition_id"] == 2:
      personal[row["clip_id"]] = personal.get(row["clip_id"], 0) + value
  assert list(personal.values()) == pytest.approx([0.25, 0.25])


def test_no_legacy_shoulders_or_missing_source_fallback():
  d = toy()
  d.rows.append({**d.rows[0], "condition_id": 1, "dataset": "lafan"})
  with pytest.raises(ValueError, match="Legacy shoulder"):
    condition_weights(d.rows, 1)
  d = toy()
  d.rows = [r for r in d.rows if r["dataset"] != "g1_moves"]
  with pytest.raises(ValueError, match="required training pool"):
    ShoulderMixedSampler(d)


def test_minimum_spans_constant_features_and_inverse_roundtrip():
  raw = np.ones((6, 10, 59), dtype=np.float32)
  raw[3:] += 0.001
  low, high, _ = normalization(raw, np.ones(6))
  assert np.all(high - low >= MIN_SPAN - 1e-6)
  normalized = 2 * (raw - low) / (high - low) - 1
  assert np.isfinite(normalized).all()
  restored = (normalized + 1) / 2 * (high - low) + low
  np.testing.assert_allclose(restored, raw, atol=1e-6)


def test_quantiles_respect_source_mass_instead_of_window_count():
  raw = np.zeros((101, 10, 59), dtype=np.float32)
  raw[-1] = 1
  weights = np.array([0.5 / 100] * 100 + [0.5])
  _, _, quantiles = normalization(raw, weights)
  assert np.all(quantiles[0] == 0)
  assert np.all(quantiles[1] == 1)


def test_source_weighted_score_cannot_hide_bad_personal_actions():
  d = toy()
  ids = (d.labels == 1).nonzero().flatten()
  good = torch.tensor([d.rows[i]["dataset"] == "ik" for i in ids])
  correct = {
    "reward": good.float()[:, None],
    "contrast_margin": torch.ones(len(ids), 1),
  }
  stand = {"reward": torch.full((len(ids), 1), 0.5)}
  m = _metrics(d, 1, ids, correct, stand, stand)
  assert m["correct_gt_stand"] == pytest.approx(0.5)
  assert m["by_source"]["personal"]["correct_gt_stand"] == 0
  assert m["by_source"]["ik"]["correct_gt_stand"] == pytest.approx(1)


def test_failure_union_includes_collision_and_joint_limit():
  failure, timeout = safety_failure_union(
    {
      "self_collision": torch.tensor([1, 0, 0, 0], dtype=torch.bool),
      "joint_hard_limit": torch.tensor([0, 1, 0, 0], dtype=torch.bool),
      "time_out": torch.tensor([0, 0, 1, 0], dtype=torch.bool),
      "emergency_force": torch.tensor([0, 0, 0, 1], dtype=torch.bool),
    }
  )
  assert failure.tolist() == [True, True, False, False]
  assert timeout.tolist() == [False, False, True, False]


def test_reward_gate_detects_zero_underflow_and_tiny_positive_scores():
  assert reward_gate(torch.ones(1024))["passed"]
  assert not reward_gate(torch.zeros(2048))["passed"]
  assert not reward_gate(torch.full((2048,), 1e-20))["passed"]
  assert not reward_gate(torch.ones(1023))["passed"]


def test_a_gate_rejects_short_episodes_even_with_small_displacement():
  metrics = {
    "precontact_root_planar_speed_m_s": {"p95": 0.01},
    "failure_nonmechanical_union": {"mean": 0},
    "precontact_root_displacement_m": {"p50": 0.001},
    "timeout_completed": {"mean": 0.2},
  }
  assert not _gate("a", metrics)["passed"]


def _response_env():
  state = {
    "started": torch.zeros(1, dtype=torch.bool),
    "root": torch.zeros(1, 3),
    "shoulder": torch.zeros(1, 3),
    "heading": torch.zeros(1),
    "torso_yaw": torch.zeros(1),
    "world_retreat": torch.zeros(1),
    "compensated_retreat": torch.zeros(1),
    "torso_signed_peak": torch.zeros(1),
    "direction": torch.zeros(1),
  }
  data = SimpleNamespace(
    root_link_pos_w=torch.zeros(1, 3),
    heading_w=torch.zeros(1),
    body_link_pos_w=torch.zeros(1, 1, 3),
    body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
  )
  c = SimpleNamespace(
    robot=SimpleNamespace(data=data),
    mixed_response=state,
    mixed_offsets=torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]),
    torso_body_id=0,
    response_condition=torch.tensor([1]),
    response_steps=torch.tensor([1]),
  )
  return SimpleNamespace(_rail_controller=c, num_envs=1, step_dt=0.02)


def test_shoulder_retreat_allows_pelvis_yaw_and_rejects_translation_only():
  env = _response_env()
  task.mixed_response_metrics(env)
  d = env._rail_controller.robot.data
  angle = torch.tensor(0.35)
  d.body_link_quat_w[0, 0] = torch.tensor(
    [torch.cos(angle / 2), 0, 0, torch.sin(angle / 2)]
  )
  task.mixed_response_metrics(env)
  assert env._rail_controller.mixed_response["direction"].item() == 1
  env = _response_env()
  task.mixed_response_metrics(env)
  d = env._rail_controller.robot.data
  d.root_link_pos_w[:, 0] = -0.1
  d.body_link_pos_w[:, :, 0] = -0.1
  task.mixed_response_metrics(env)
  assert env._rail_controller.mixed_response["direction"].item() == 0


def test_zero_guard_stops_at_twenty_rollouts(monkeypatch):
  env = SimpleNamespace(
    _smp_unexpected_condition=torch.zeros(2), _smp_reward=torch.zeros(2)
  )
  monkeypatch.setattr(task, "mixed_response_metrics", lambda env: None)
  monkeypatch.setattr(
    task, "task_conditional_contrastive_smp_product", lambda *a: torch.zeros(2)
  )
  for _ in range(24 * 20 - 1):
    task.mixed_task_product(env, ())
  with pytest.raises(RuntimeError, match="20 rollouts"):
    task.mixed_task_product(env, ())


def test_schema_and_legacy_task_remain_separate():
  from smp.pretrain.schema import resolve_condition, validate_schema
  from smp.rl.tasks.force_unloading.conditional_shoulder_ik_v2 import TASK_ID

  assert validate_schema(CONDITION_SCHEMA, 3) == CONDITION_SCHEMA
  assert resolve_condition("null", CONDITION_SCHEMA) == 3
  with pytest.raises(ValueError):
    resolve_condition("chest", CONDITION_SCHEMA)

  assert CONDITION_SCHEMA["null_id"] == 3
  assert CONDITION_SCHEMA["names"][0] == "stand"
  assert TASK_ID != task.TASK_ID
  cfg = task.shoulder_mixed_env_cfg()
  assert cfg.events["init_smp_state"].params["gsi_max_root_planar_speed"] == 0.15
  assert cfg.rewards["task_smp_product"].func is task.mixed_task_product


def test_new_shoulder_gate_does_not_reapply_old_pelvis_cap():
  metrics = {
    "failure_emergency_force": {"mean": 0.02},
    "event_success": {"mean": 0.3},
    "contact_coverage": {"mean": 0.9},
    "contact_condition_classification_correct": {"mean": 0.9},
    "response_direction_correct_given_target_contact": {"mean": 0.85},
    "signed_waist_response_peak_rad_given_target_contact": {"p50": 0.02},
    "pelvis_yaw_abs_peak_rad_given_target_contact": {"p95": 0.6},
  }
  assert _gate("b1", metrics, shoulder_gate=True)["passed"]


def test_checkpoint_keeps_all_ppo_tensors_and_checks_prior_identity(
  tmp_path, monkeypatch
):
  monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
  learning = {
    "actor_state_dict": {"weight": torch.randn(2, 3)},
    "critic_state_dict": {"weight": torch.randn(2, 4)},
    "optimizer_state_dict": {"momentum": torch.randn(3)},
    "normalizer": {"mean": torch.randn(3)},
  }
  env = SimpleNamespace(
    common_step_counter=123,
    _smp_prior_identity={"sha": "prior"},
    _smp_gsi_prior_identity={"sha": "prior"},
    _smp_score_identity={"sha": "calibration"},
  )
  runner = task.ShoulderMixedRunner.__new__(task.ShoulderMixedRunner)
  runner.env = SimpleNamespace(unwrapped=env)
  runner.alg = SimpleNamespace(save=lambda: dict(learning), learning_rate=7e-5)
  runner.cfg = {"upload_model": False}
  runner.current_learning_iteration = 9
  path = tmp_path / "model_9.pt"
  runner.save(path)
  stored = torch.load(path, weights_only=True)
  for field, values in learning.items():
    for key, tensor in values.items():
      assert torch.equal(tensor, stored[field][key])
  assert stored["iter"] == 9
  assert stored["infos"]["env_state"]["common_step_counter"] == 123
  monkeypatch.setattr(task.MjlabOnPolicyRunner, "load", lambda *a, **k: {})
  monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
  runner.alg.learning_rate = 1e-4
  runner.load(path)
  assert runner.current_learning_iteration == 10
  assert runner.alg.learning_rate == pytest.approx(7e-5)
  env._smp_prior_identity = {"sha": "wrong"}
  with pytest.raises(ValueError, match="Prior/schema mismatch"):
    runner.load(path)


def test_mixed_iteration_budget_excludes_already_completed_update():
  from scripts.run_g1_shoulder_mixed_prior_ppo_suite import (
    expected_terminal_iteration,
    remaining_iterations,
  )

  assert expected_terminal_iteration(SimpleNamespace(iterations=3000), None) == 2999
  assert expected_terminal_iteration(SimpleNamespace(iterations=4000), 2999) == 6999
  assert remaining_iterations(6999, 4999) == 2000
  with pytest.raises(ValueError):
    remaining_iterations(6999, 6999)


def test_evaluation_wrapper_adds_back_slot_and_new_script(monkeypatch):
  from scripts import run_g1_shoulder_mixed_prior_ppo_suite as suite

  captured = {}

  def original(*args, **kwargs):
    captured.update(kwargs)
    return ["python", "scripts/evaluate_force_unloading.py"]

  monkeypatch.setattr(suite.curriculum, "build_evaluation_command", original)
  monkeypatch.setattr(suite, "build_evaluation_command", original)
  suite.install_shoulder_gate_command_wrapper()
  command = suite.build_evaluation_command(
    None, SimpleNamespace(key="b1"), None, None, target_probabilities=(0.0, 1.0, 0.0)
  )
  assert captured["target_probabilities"] == (0.0, 1.0, 0.0, 0.0)
  assert command[1] == "scripts/evaluate_shoulder_mixed.py"
  assert "--shoulder-gate" in command
