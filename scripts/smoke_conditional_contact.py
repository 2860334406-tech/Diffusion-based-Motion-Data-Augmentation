"""GPU1 B1 mechanical smoke; holds robot pose to isolate four contact geometries."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv

from smp.rl.tasks.force_unloading.conditional import conditional_env_cfg
from smp.rl.tasks.force_unloading.mdp.controller import RailPusherCfg


@dataclass
class Cfg:
  output: str = "logs/reproduction/conditional_contact_v1/contact_smoke.json"
  num_envs: int = 64
  steps: int = 600
  device: str = "cuda:0"


def main(cfg):
  if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise RuntimeError("GPU1 isolation required")
  rail = RailPusherCfg(
    conditional=True, sphere_radius=.1, start_surface_gap=.02,
    reference_speed=.01, max_force_weight_ratio=.05, ramp_time=1.,
    wait_range_s=(0., 0.), target_probabilities=(.25, .25, .25, .25),
    push_direction_mode="front_to_back", stationary_speed_scale=10.,
  )
  config = conditional_env_cfg(rail_cfg=rail)
  config.scene.num_envs, config.seed = cfg.num_envs, 42
  for name in ("init_smp_state", "gsi_reset", "gsi_refresh"):
    config.events.pop(name, None)
  config.rewards, config.terminations = {}, {}
  env = ManagerBasedRlEnv(config, device=cfg.device)
  try:
    obs, _ = env.reset()
    robot = env.scene["robot"]
    roots = robot.data.default_root_state.clone()
    roots[:, :3] = robot.data.root_link_pos_w
    roots[:, 3:7] = robot.data.root_link_quat_w
    roots[:, 7:] = 0
    joints = robot.data.joint_pos.clone()
    actions = torch.zeros(cfg.num_envs, env.action_manager.total_action_dim, device=cfg.device)
    controller = env._rail_controller
    contacts, ids_seen, first_times = set(), set(), {}
    cross = 0.
    collision_geoms = {}
    region_peaks = torch.zeros((4, 4), device=cfg.device)
    force_peaks = torch.zeros(4, device=cfg.device)
    for step in range(cfg.steps):
      robot.write_root_state_to_sim(roots)
      robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
      env.sim.forward()
      obs, *_ = env.step(actions)
      condition = controller.condition_state.ids
      for c in condition.unique().tolist():
        ids_seen.add(c)
        if c > 0:
          first_times.setdefault(str(c), (step+1)*env.step_dt)
      for target in range(4):
        selected = controller.target == target
        if selected.any():
          region_peaks[target] = torch.maximum(region_peaks[target], controller.region_ratios[selected].max(0).values)
          force_peaks[target] = torch.maximum(force_peaks[target], controller.raw_force_peak[selected].max())
        contact = (controller.target == target) & (controller.raw_force_peak > controller.force_on)
        if contact.any():
          contacts.add(target)
          if str(target) not in collision_geoms:
            world = int(contact.nonzero()[0, 0])
            cpu_data = mujoco.MjData(env.sim.mj_model)
            cpu_data.qpos[:] = env.sim.data.qpos[world].cpu().numpy()
            cpu_data.mocap_pos[:] = env.sim.data.mocap_pos[world].cpu().numpy()
            cpu_data.mocap_quat[:] = env.sim.data.mocap_quat[world].cpu().numpy()
            mujoco.mj_forward(env.sim.mj_model, cpu_data)
            collision_geoms[str(target)] = {
              "time": (step+1)*env.step_dt,
              "pairs": [[env.sim.mj_model.geom(int(c.geom[0])).name,
                         env.sim.mj_model.geom(int(c.geom[1])).name, float(c.dist)]
                        for c in cpu_data.contact
                        if "rail_pusher/sphere" in (env.sim.mj_model.geom(int(c.geom[0])).name,
                                                   env.sim.mj_model.geom(int(c.geom[1])).name)]}

      cross = max(cross, float(controller.rail_cross_axis_error.max()))
      if not torch.isfinite(obs["actor"]).all() or not torch.isfinite(obs["critic"]).all():
        raise FloatingPointError("Nonfinite mechanical smoke observation")
    sphere_id = env.sim.mj_model.geom("rail_pusher/sphere").id
    radius = float(env.sim.mj_model.geom_size[sphere_id, 0])
    report = {
      "physical_gpu": 1, "robot_kinematically_held": True,
      "reference_speed": .01, "ramp": 1., "initial_surface_gap": .02,
      "actual_sphere_radius": radius, "actor_shape": list(obs["actor"].shape),
      "critic_shape": list(obs["critic"].shape), "contact_targets": sorted(contacts),
      "conditions_seen": sorted(ids_seen), "first_condition_time_s": first_times,
      "max_cross_axis_error_m": cross, "finite": True,
      "region_names": controller.region_names, "region_peak_ratios_by_target": region_peaks.tolist(),
      "total_force_peaks_by_target": force_peaks.tolist(), "collision_geoms": collision_geoms,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    if contacts != {0, 1, 2, 3} or not {1, 2, 3, 4}.issubset(ids_seen):
      raise RuntimeError(f"Four-region contact smoke failed: {report}")
    if abs(radius-.1) > 1e-6 or cross > 1e-5:
      raise RuntimeError(f"Physical geometry mismatch: {report}")
    assert obs["actor"].shape[-1] == 128 and obs["critic"].shape[-1] == 992
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg))
