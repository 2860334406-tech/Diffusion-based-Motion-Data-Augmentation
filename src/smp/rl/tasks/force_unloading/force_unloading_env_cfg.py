"""G1 SMP task for unloading a rail-mounted spherical pusher."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.force_unloading import mdp


def g1_force_unloading_smp_env_cfg(
  play: bool = False,
  rail_cfg: mdp.RailPusherCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the stage-1 chest-push force-unloading baseline."""
  cfg = g1_smp_env_cfg(play=play)
  rail_cfg = rail_cfg or mdp.RailPusherCfg()

  cfg.scene.entities["rail_pusher"] = mdp.rail_pusher_entity_cfg(rail_cfg)
  pusher_contact = ContactSensorCfg(
    name="rail_pusher_contact",
    primary=ContactMatch(mode="geom", pattern="sphere", entity="rail_pusher"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force", "pos", "normal"),
    reduce="none",
    num_slots=8,
    history_length=cfg.decimation,
  )
  nonfoot_ground = ContactSensorCfg(
    name="nonfoot_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision",
      entity="robot",
      exclude=(r"^(left|right)_foot[1-7]_collision$",),
    ),
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  foot_ground = ContactSensorCfg(
    name="foot_ground_contact",
    primary=ContactMatch(
      mode="geom", pattern=r"^(left|right)_foot[1-7]_collision$", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    pusher_contact,
    nonfoot_ground,
    foot_ground,
  )
  cfg.scene.extent = 3.0
  cfg.sim.nconmax = 64

  cfg.events.pop("push_robot", None)
  cfg.events["rail_pusher_controller"] = EventTermCfg(
    func=mdp.RailPusherController,
    mode="step",
    params={"rail_cfg": rail_cfg},
  )
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_loco.pt"
  )
  cfg.events["init_smp_state"].params.update(
    {
      "gsi_buffer_size": 1024,
      "gsi_batch_size": 1024,
      "gsi_reject_unsafe": True,
      "gsi_joint_limit_margin": 1.0e-4,
      "gsi_max_root_planar_speed": 1.2,
      "gsi_self_collision_sensor": "self_collision",
      "gsi_max_sampling_multiplier": 20,
    }
  )
  if "gsi_refresh" in cfg.events:
    cfg.events["gsi_refresh"].params["num_samples"] = 256

  # Base cfg shares term objects between groups; split them before adding
  # critic-only histories.
  cfg.observations["critic"].terms = deepcopy(cfg.observations["critic"].terms)
  cfg.observations["critic"].history_length = None
  for term in cfg.observations["critic"].terms.values():
    term.history_length = 10
  cfg.observations["actor"].terms["rail_contact"] = ObservationTermCfg(
    func=mdp.rail_contact_observation,
    noise=mdp.RailContactNoiseCfg(),
    history_length=4,
  )
  cfg.observations["critic"].terms["rail_contact"] = ObservationTermCfg(
    func=mdp.rail_contact_observation,
    history_length=4,
  )
  cfg.observations["actor"].nan_policy = "error"
  cfg.observations["critic"].nan_policy = "error"
  if play:
    cfg.observations["actor"].enable_corruption = False

  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": ((mdp.force_unloading_task_reward, 1.0, {}),),
      "ws": 6.0,
    },
  )

  cfg.episode_length_s = 12.0
  cfg.terminations.update(
    {
      "nonfoot_ground": TerminationTermCfg(func=mdp.nonfoot_ground_contact),
      "root_height": TerminationTermCfg(func=mdp.root_height_failure),
      "torso_tilt": TerminationTermCfg(func=mdp.torso_tilt_failure),
      "workspace": TerminationTermCfg(func=mdp.workspace_failure),
      "emergency_force": TerminationTermCfg(func=mdp.emergency_force),
      "joint_hard_limit": TerminationTermCfg(func=mdp.joint_hard_limit_failure),
      "nan_or_inf": TerminationTermCfg(func=mdp.nan_failure),
    }
  )

  cfg.metrics = {
    "peak_force_weight_ratio": MetricsTermCfg(func=mdp.peak_force_ratio, reduce="max"),
    "normalized_force_impulse_s": MetricsTermCfg(
      func=mdp.normalized_force_impulse, reduce="sum"
    ),
    "unload_time_s": MetricsTermCfg(func=mdp.unload_time, reduce="last"),
    "event_success": MetricsTermCfg(func=mdp.event_success, reduce="max"),
    "max_x_displacement_m": MetricsTermCfg(func=mdp.max_x_displacement, reduce="max"),
    "max_y_displacement_m": MetricsTermCfg(func=mdp.max_y_displacement, reduce="max"),
    "precontact_root_planar_speed_m_s": MetricsTermCfg(
      func=mdp.precontact_root_planar_speed, reduce="max"
    ),
    "precontact_root_displacement_m": MetricsTermCfg(
      func=mdp.precontact_root_displacement, reduce="max"
    ),
    "capture_point_overflow_m": MetricsTermCfg(
      func=mdp.capture_point_overflow, reduce="max"
    ),
    "max_contact_foot_slip_m_s": MetricsTermCfg(
      func=mdp.max_contact_foot_slip, reduce="max"
    ),
    "failure_nonfoot_ground": MetricsTermCfg(
      func=mdp.failure_metric, params={"reason": "nonfoot_ground"}, reduce="max"
    ),
    "failure_height": MetricsTermCfg(
      func=mdp.failure_metric, params={"reason": "height"}, reduce="max"
    ),
    "failure_tilt": MetricsTermCfg(
      func=mdp.failure_metric, params={"reason": "tilt"}, reduce="max"
    ),
    "failure_workspace": MetricsTermCfg(
      func=mdp.failure_metric, params={"reason": "workspace"}, reduce="max"
    ),
    "failure_emergency_force": MetricsTermCfg(
      func=mdp.emergency_failure_metric, reduce="max"
    ),
    "travel_limit_reached": MetricsTermCfg(func=mdp.travel_limit_reached, reduce="max"),
    "failure_joint": MetricsTermCfg(
      func=mdp.failure_metric, params={"reason": "joint"}, reduce="max"
    ),
    "failure_nan": MetricsTermCfg(
      func=mdp.failure_metric, params={"reason": "nan"}, reduce="max"
    ),
    "smp_reward_mean": MetricsTermCfg(func=mdp.smp_reward_mean, reduce="mean"),
    "smp_reward_variance": MetricsTermCfg(func=mdp.smp_reward_variance, reduce="mean"),
  }
  return cfg
