"""CPU-only shape/order smoke for the EXP-022 force-estimator interface.

This removes GSI startup/reset events to avoid running a policy episode. It
checks observation layout and one finite frozen-model inference; it is not an
accuracy or control evaluation.
"""

from __future__ import annotations

import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import smp.rl.tasks  # noqa: F401
from smp.force_estimation import load_force_estimator
from smp.force_estimation.exp022_interface import (
  add_force_estimator_observations,
  validate_force_estimator_input,
)


def main() -> None:
  cfg = load_env_cfg("Smp-Force-Unloading-G1", play=True)
  cfg.scene.num_envs = 1
  for name in ("init_smp_state", "gsi_reset", "gsi_refresh"):
    cfg.events.pop(name, None)
  add_force_estimator_observations(cfg)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    observations, _ = env.reset()
    validate_force_estimator_input(env, observations)
    estimator = load_force_estimator(version="v392", device="cpu")
    output = estimator.step(
      observations["force_estimator_x"][0].cpu().numpy(),
      observations["force_estimator_tau"][0].cpu().numpy(),
      target_site_id=0,
    )
    if (
      not np.isfinite(output["force_body_n"]).all()
      or not np.isfinite(output["point_body_m"]).all()
    ):
      raise ValueError("Estimator returned non-finite force or point")
    print(
      "EXP-022 interface smoke OK:",
      {name: tuple(value.shape) for name, value in observations.items()},
      "force_body_n:",
      output["force_body_n"].shape,
      "point_body_m:",
      output["point_body_m"].shape,
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
