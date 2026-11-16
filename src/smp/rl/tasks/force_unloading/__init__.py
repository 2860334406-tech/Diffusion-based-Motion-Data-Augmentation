"""Register the G1 rail-pusher SMP force-unloading tasks."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.force_unloading.force_unloading_env_cfg import (
  g1_force_unloading_smp_env_cfg,
)

_rl_cfg = unitree_g1_smp_ppo_runner_cfg()
_rl_cfg.experiment_name = "smp_force_unloading_g1"
_rl_cfg.run_name = "smp_force_unloading_g1"

register_mjlab_task(
  task_id="Smp-Force-Unloading-G1",
  env_cfg=g1_force_unloading_smp_env_cfg(play=False),
  play_env_cfg=g1_force_unloading_smp_env_cfg(play=True),
  rl_cfg=_rl_cfg,
)


def _register_conditional_task():
  # Defer until the base package is initialized to avoid its mdp import cycle.
  from smp.rl.tasks.force_unloading.conditional import register_conditional_task

  register_conditional_task()


_register_conditional_task()


def _register_shoulder_score_v3_task():
  from smp.rl.tasks.force_unloading.conditional_shoulder_score_v3 import (
    register_shoulder_score_v3_task,
  )

  register_shoulder_score_v3_task()


_register_shoulder_score_v3_task()


def _register_shoulder80_ppo_task():
  from smp.rl.tasks.force_unloading.conditional_shoulder80_ppo import (
    register_shoulder80_ppo_task,
  )

  register_shoulder80_ppo_task()


_register_shoulder80_ppo_task()


def _register_shoulder_ik_v2_task():
  from smp.rl.tasks.force_unloading.conditional_shoulder_ik_v2 import (
    register_shoulder_ik_v2_task,
  )

  register_shoulder_ik_v2_task()


_register_shoulder_ik_v2_task()


def _register_shoulder_mixed_task():
  from smp.rl.tasks.force_unloading.conditional_shoulder_mixed import (
    register_shoulder_mixed_task,
  )

  register_shoulder_mixed_task()


_register_shoulder_mixed_task()

__all__ = ["g1_force_unloading_smp_env_cfg"]
