"""Collect the 2026-11-11 estimator inputs under the frozen rail pusher force."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

from smp.rl.tasks.force_unloading.conditional import conditional_env_cfg
from smp.rl.tasks.force_unloading.mdp.controller import RailPusherCfg


def build_930(critic_obs):
  """Reorder the native critic history into the saved estimator contract."""
  if critic_obs.shape != (critic_obs.shape[0], 992):
    raise ValueError(f"Unexpected critic observation shape: {tuple(critic_obs.shape)}")
  return torch.cat(
    (
      critic_obs[:, 90:380],
      critic_obs[:, 380:670],
      critic_obs[:, 30:60],
      critic_obs[:, 60:90],
      critic_obs[:, 670:960],
    ),
    dim=-1,
  )


def newest(history):
  return np.concatenate(
    (
      history[..., 261:290],
      history[..., 551:580],
      history[..., 607:610],
      history[..., 637:640],
    ),
    axis=-1,
  )


@torch.no_grad()
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--split", choices=("train", "val", "dev", "test"), required=True)
  parser.add_argument("--batch-index", type=int, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--seed", type=int, default=20261201)
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.out.exists():
    raise FileExistsError(args.out)

  configure_torch_backends()
  rail = RailPusherCfg(
    conditional=True,
    sphere_radius=0.10,
    sphere_mass=5.0,
    friction=0.2,
    start_surface_gap=0.02,
    reference_speed=0.05,
    stiffness=700.0,
    damping=118.32159566199232,
    max_force_weight_ratio=0.15,
    max_travel=0.375,
    ramp_time=0.25,
    retract_speed=0.15,
    wait_range_s=(1.0, 3.0),
    target_probabilities=(0.25, 0.25, 0.25, 0.25),
    push_direction_mode="front_to_back",
    stationary_speed_scale=10.0,
  )
  cfg = conditional_env_cfg(rail_cfg=rail)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed + args.batch_index
  for name in ("init_smp_state", "gsi_reset", "gsi_refresh", "push_robot"):
    cfg.events.pop(name, None)
  cfg.rewards = {}
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    robot = env.scene["robot"]
    controller = env._rail_controller
    contact_sensor = env.scene["rail_pusher_contact"]
    torso_id = robot.body_names.index("torso_link")
    obs, _ = env.reset()
    root_lock = robot.data.default_root_state.clone()
    root_lock[:, :3] = robot.data.root_link_pos_w
    root_lock[:, 3:7] = robot.data.root_link_quat_w
    root_lock[:, 7:] = 0.0
    actions = torch.zeros(
      args.num_envs, env.action_manager.total_action_dim, device=args.device
    )
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
    xs, taus, forces, points, contacts, phases, targets, roots = [], [], [], [], [], [], [], []
    valids, terminations, timeouts = [], [], []

    for _ in range(args.steps):
      robot.write_root_state_to_sim(root_lock)
      obs, _, term, timeout, _ = env.step(actions)
      controller.refresh_contact()
      data = contact_sensor.data
      found = data.found > 0
      normal = data.force[..., 0].abs() * found
      normal_sum = normal.sum(dim=1)
      weighted_vec = (normal[..., None] * data.normal).sum(dim=1)
      weighted_pos = (normal[..., None] * data.pos).sum(dim=1) / normal_sum.clamp_min(
        1.0e-8
      )[:, None]
      hit = normal_sum >= 1.0
      torso_pos = robot.data.body_link_pos_w[:, torso_id]
      torso_quat = robot.data.body_link_quat_w[:, torso_id]
      force_b = quat_apply_inverse(torso_quat, weighted_vec)
      point_b = quat_apply_inverse(torso_quat, weighted_pos - torso_pos)
      force_b = torch.where(hit[:, None], force_b, 0.0)
      point_b = torch.where(hit[:, None], point_b, 0.0)

      xs.append(build_930(obs["critic"]).clone())
      taus.append(robot.data.qfrc_actuator.clone())
      forces.append(force_b)
      points.append(point_b)
      contacts.append(hit)
      phases.append(controller.phase.clone())
      targets.append(controller.target.clone())
      roots.append(
        torch.cat(
          (
            robot.data.root_link_pos_w,
            robot.data.root_link_lin_vel_w,
            robot.data.root_link_quat_w,
          ),
          dim=-1,
        )
      )
      valids.append(alive & ~(term | timeout))
      terminations.append(term.clone())
      timeouts.append(timeout.clone())
      alive &= ~(term | timeout)
      if not bool(alive.any()):
        break

    x = torch.stack(xs, dim=1).cpu().numpy()
    tau = torch.stack(taus, dim=1).cpu().numpy()
    force = torch.stack(forces, dim=1).cpu().numpy()
    point = torch.stack(points, dim=1).cpu().numpy()
    contact = torch.stack(contacts, dim=1).cpu().numpy().astype(np.uint8)
    phase = torch.stack(phases, dim=1).cpu().numpy().astype(np.uint8)
    target = torch.stack(targets, dim=1).cpu().numpy().astype(np.int16)
    root = torch.stack(roots, dim=1).cpu().numpy()
    valid = torch.stack(valids, dim=1).cpu().numpy().astype(bool)
    terminated = torch.stack(terminations, dim=1).cpu().numpy().astype(bool)
    timed_out = torch.stack(timeouts, dim=1).cpu().numpy().astype(bool)
    pair = valid[:, :-1] & valid[:, 1:]
    selected = np.nonzero(pair)
    run_ids = np.asarray(
      [f"{args.split}_b{args.batch_index:03d}_e{i:04d}" for i in range(args.num_envs)]
    )
    payload = {
      "x": x[:, :-1][selected].astype(np.float32),
      "next_proprio64": newest(x[:, 1:])[selected].astype(np.float32),
      "tau_est": tau[:, :-1][selected].astype(np.float32),
      "force_body_n": force[:, :-1][selected].astype(np.float32),
      "point_body_m": point[:, :-1][selected].astype(np.float32),
      "contact": contact[:, :-1][selected],
      "phase": phase[:, :-1][selected],
      "site_id": target[:, :-1][selected],
      "root_state": root[:, :-1][selected].astype(np.float32),
      "run_id": run_ids[selected[0]],
      "step": selected[1].astype(np.int16),
    }
    if payload["x"].shape != (len(selected[0]), 930):
      raise ValueError(f"Unexpected input shape: {payload['x'].shape}")
    if not all(np.isfinite(payload[name]).all() for name in ("x", "tau_est", "force_body_n")):
      raise ValueError("Nonfinite estimator sample")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    active = payload["contact"] > 0
    magnitudes = np.linalg.norm(payload["force_body_n"][active], axis=1)
    manifest = {
      "split": args.split,
      "batch_index": args.batch_index,
      "seed": args.seed,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "rows": len(payload["x"]),
      "contact_rows": int(active.sum()),
      "contact_fraction": float(active.mean()),
      "force_contact_quantiles_n": np.quantile(
        magnitudes, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
      ).tolist(),
      "terminated_envs": int(terminated[:, 0].sum()),
      "timed_out_envs": int(timed_out[:, -1].sum()),
      "input_contract": (
        "930D: joint_pos/joint_vel/angular_velocity/projected_gravity/actions, "
        "10-frame history; no pusher, target, phase, contact, or future state"
      ),
      "tau_contract": "post-step robot.data.qfrc_actuator, 29 joints",
      "label_contract": (
        "post-step contact-sensor normal force and normal-weighted contact point, "
        "both transformed into torso_link coordinates"
      ),
      "device": str(env.device),
      "rail": {
        "sphere_radius_m": rail.sphere_radius,
        "sphere_mass_kg": rail.sphere_mass,
        "reference_speed_mps": rail.reference_speed,
        "stiffness_n_m": rail.stiffness,
        "damping_ns_m": rail.damping,
        "max_force_weight_ratio": rail.max_force_weight_ratio,
        "max_travel_m": rail.max_travel,
        "ramp_time_s": rail.ramp_time,
        "start_surface_gap_m": rail.start_surface_gap,
        "push_direction_mode": rail.push_direction_mode,
        "wait_range_s": rail.wait_range_s,
      },
      "data_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
