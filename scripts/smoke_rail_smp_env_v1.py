"""Validate the frozen rail pusher in its native SMP environment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv

from smp.rl.tasks.force_unloading.conditional import conditional_env_cfg
from smp.rl.tasks.force_unloading.mdp.controller import RailPusherCfg


@dataclass
class Cfg:
  output: str = (
    "/personal_fast_storage/Route_B_SuperMimic/evaluations/"
    "rail_smooth_force_v1/native_mechanical_smoke.json"
  )
  num_envs: int = 64
  steps: int = 600
  device: str = "cuda:0"


def main(cfg: Cfg):
  rail = RailPusherCfg(
    conditional=True,
    sphere_radius=0.10,
    sphere_mass=5.0,
    start_surface_gap=0.02,
    reference_speed=0.05,
    max_force_weight_ratio=0.15,
    ramp_time=0.25,
    wait_range_s=(0.0, 0.0),
    target_probabilities=(0.25, 0.25, 0.25, 0.25),
    push_direction_mode="front_to_back",
    stationary_speed_scale=10.0,
  )
  config = conditional_env_cfg(rail_cfg=rail)
  config.scene.num_envs = cfg.num_envs
  config.seed = 42
  for name in ("init_smp_state", "gsi_reset", "gsi_refresh"):
    config.events.pop(name, None)
  config.rewards = {}
  config.terminations = {}
  env = ManagerBasedRlEnv(config, device=cfg.device)
  try:
    obs, _ = env.reset()
    robot = env.scene["robot"]
    roots = robot.data.default_root_state.clone()
    roots[:, :3] = robot.data.root_link_pos_w
    roots[:, 3:7] = robot.data.root_link_quat_w
    roots[:, 7:] = 0.0
    joints = robot.data.joint_pos.clone()
    actions = torch.zeros(
      cfg.num_envs, env.action_manager.total_action_dim, device=cfg.device
    )
    controller = env._rail_controller
    contacts = set()
    first_times = {}
    peak_force = torch.zeros(4, device=cfg.device)
    max_cross = 0.0
    max_drive_ratio = 0.0
    travel_limit = False

    for step in range(cfg.steps):
      robot.write_root_state_to_sim(roots)
      robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
      env.sim.forward()
      obs, *_ = env.step(actions)
      controller.refresh_contact()
      for target in range(4):
        selected = controller.target == target
        if selected.any():
          peak_force[target] = torch.maximum(
            peak_force[target],
            controller.raw_force_peak[selected].max(),
          )
          active = selected & (controller.raw_force_peak > controller.force_on)
          if active.any():
            contacts.add(target)
            first_times.setdefault(str(target), (step + 1) * float(env.step_dt))
      max_cross = max(max_cross, float(controller.rail_cross_axis_error.max()))
      max_drive_ratio = max(
        max_drive_ratio,
        float(controller.env.sim.data.qfrc_applied[:, controller.pusher.indexing.joint_v_adr[0]].abs().max())
        / controller.max_drive_force,
      )
      travel_limit |= bool(controller.travel_limit_reached.any())
      if not torch.isfinite(obs["actor"]).all() or not torch.isfinite(obs["critic"]).all():
        raise FloatingPointError("Nonfinite native smoke observation")

    report = {
      "physical_device": torch.cuda.get_device_name(0),
      "robot_mass_kg": controller.robot_mass,
      "weight_n": controller.weight,
      "max_drive_force_n": controller.max_drive_force,
      "contact_targets": sorted(contacts),
      "first_contact_time_s": first_times,
      "peak_normal_force_n_by_target": peak_force.cpu().tolist(),
      "max_cross_axis_error_m": max_cross,
      "max_drive_fraction_of_cap": max_drive_ratio,
      "travel_limit_reached": travel_limit,
      "steps": cfg.steps,
      "num_envs": cfg.num_envs,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if contacts != {0, 1, 2, 3}:
      raise RuntimeError(f"Native four-target smoke failed: {report}")
    if max_cross > 1e-5 or max_drive_ratio > 1.0 + 1e-5:
      raise RuntimeError(f"Native mechanism check failed: {report}")
    print(json.dumps(report, indent=2), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg))
