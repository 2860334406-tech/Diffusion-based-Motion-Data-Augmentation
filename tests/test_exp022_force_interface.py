"""CPU contract checks for the EXP-022 force estimator interface."""

from types import SimpleNamespace

import pytest
import torch

from smp.force_estimation.exp022_interface import (
  G1_JOINT_NAMES,
  Exp022ContactAdapter,
  add_force_estimator_observations,
  joint_space_actuator_force,
  validate_force_estimator_input,
)


def test_input_groups_and_joint_force_contract():
  cfg = SimpleNamespace(observations={})
  add_force_estimator_observations(cfg)
  x_cfg = cfg.observations["force_estimator_x"]
  tau_cfg = cfg.observations["force_estimator_tau"]
  assert tuple(x_cfg.terms) == (
    "joint_pos",
    "joint_vel",
    "base_ang_vel",
    "projected_gravity",
    "actions",
  )
  assert x_cfg.terms["joint_pos"].params == {"biased": True}
  assert x_cfg.history_length == 10 and x_cfg.flatten_history_dim
  assert tau_cfg.history_length == 10 and not tau_cfg.flatten_history_dim
  with pytest.raises(ValueError, match="already exists"):
    add_force_estimator_observations(cfg)

  torque = torch.arange(58, dtype=torch.float32).reshape(2, 29)
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(data=SimpleNamespace(qfrc_actuator=torque))}
  )
  assert joint_space_actuator_force(env) is torque


def test_input_validation_rejects_action_permutation():
  action_names = list(G1_JOINT_NAMES)
  term = SimpleNamespace(target_names=action_names)
  env = SimpleNamespace(
    num_envs=2,
    scene={"robot": SimpleNamespace(joint_names=G1_JOINT_NAMES)},
    action_manager=SimpleNamespace(get_term=lambda _: term),
  )
  obs = {
    "force_estimator_x": torch.zeros(2, 930),
    "force_estimator_tau": torch.zeros(2, 10, 29),
  }
  validate_force_estimator_input(env, obs)
  action_names[0], action_names[1] = action_names[1], action_names[0]
  with pytest.raises(ValueError, match="Action order"):
    validate_force_estimator_input(env, obs)


def test_contact_mapping_filter_heading_and_history():
  controller = SimpleNamespace(
    num_envs=1,
    device="cpu",
    weight=100,
    cfg=SimpleNamespace(ema_alpha=0.8, force_on_ratio=0.0075, force_obs_clip=0.4),
  )
  adapter = Exp022ContactAdapter.from_controller(controller)
  yaw_90 = torch.tensor([[2**-0.5, 0, 0, 2**-0.5]])
  zeros = torch.zeros(1, 3)
  frame = adapter.step(
    torch.tensor([[10.0, 0, 0]]),
    torch.tensor([[0.2, 0, 0]]),
    torso_quat_w=yaw_90,
    torso_pos_w=torch.tensor([[1.0, 2.0, 1.0]]),
    root_quat_w=yaw_90,
    root_pos_w=torch.tensor([[1.0, 2.0, 0.0]]),
  )
  torch.testing.assert_close(frame[0], torch.tensor([1, 0.02, 1, 0, 0, 0.2, 0, 1]))
  torch.testing.assert_close(adapter.history[0, :24], torch.zeros(24))
  torch.testing.assert_close(adapter.history[0, 24:], frame[0])

  # With zero instantaneous force, the EMA and last observed direction/point persist.
  next_frame = adapter.step(
    zeros,
    zeros,
    torso_quat_w=yaw_90,
    torso_pos_w=torch.tensor([[1.0, 2.0, 1.0]]),
    root_quat_w=yaw_90,
    root_pos_w=torch.tensor([[1.0, 2.0, 0.0]]),
  )
  torch.testing.assert_close(next_frame[0, 1], torch.tensor(0.016))
  torch.testing.assert_close(next_frame[0, 2:8], frame[0, 2:8])
  adapter.reset()
  torch.testing.assert_close(adapter.history, torch.zeros(1, 32))


def test_contact_history_resets_only_selected_environment():
  adapter = Exp022ContactAdapter(2, "cpu", robot_weight_n=100)
  identity = torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1)
  adapter.step(
    torch.tensor([[10.0, 0, 0], [10.0, 0, 0]]),
    torch.zeros(2, 3),
    torso_quat_w=identity,
    torso_pos_w=torch.zeros(2, 3),
    root_quat_w=identity,
    root_pos_w=torch.zeros(2, 3),
  )
  adapter.reset(torch.tensor([0]))
  torch.testing.assert_close(adapter.history[0], torch.zeros(32))
  assert adapter.history[1, -8] == 1
