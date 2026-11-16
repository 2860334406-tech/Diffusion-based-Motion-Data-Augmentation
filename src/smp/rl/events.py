"""Startup + reset events for SMP RL.

Run from mjlab's event manager so the task stays a plain ``ManagerBasedRlEnv``.
Motion features carry no absolute root pose, so GSI writes a default root frame
(each env's origin, identity yaw) to sim and primes the feature buffer in an
env-origin-relative frame, so the SMP reward is invariant to env placement.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser
from smp.sampling.feature_to_state import (
  EE_BODY_NAMES,
  NUM_EE,
  rot6d_to_quat,
  slice_features,
)

NUM_JOINTS = 29


def _maybe_compile(model, compile_model: bool, compile_mode: str | None):
  """``torch.compile`` ``model`` (no-op if ``compile_model`` false), working
  around the Inductor ``pad_mm`` TF32 crash by disabling shape padding."""
  if not compile_model:
    return model
  torch.set_float32_matmul_precision("high")
  try:
    import torch._inductor.config as _ic

    _ic.shape_padding = False
  except ImportError:
    pass
  if compile_mode is not None:
    return torch.compile(model, fullgraph=True, mode=compile_mode)
  return torch.compile(model, fullgraph=True)


def init_smp_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  ckpt_path: str = "",
  gsi_ckpt_path: str = "",
  conditional_score_calibration_path: str = "",
  gsi_buffer_size: int = 4096,
  gsi_batch_size: int = 256,
  gsi_reject_unsafe: bool = False,
  gsi_joint_limit_margin: float = 0.0,
  gsi_max_root_planar_speed: float | None = None,
  gsi_self_collision_sensor: str | None = None,
  gsi_max_sampling_multiplier: int = 10,
  compile_model: bool = True,
  compile_mode: str | None = None,
) -> None:
  """Startup-mode event: load the frozen denoiser, allocate the feature buffer +
  ``DiffNormalizer`` (stashed on the env), and pre-generate the GSI pool of
  ``gsi_buffer_size`` windows that ``gsi_reset`` samples from (amortizes the DDPM
  cost).  If ``compile_model``, the denoiser is ``torch.compile``-d and pre-warmed
  so Inductor compiles here, not on the first sim step."""
  del env_ids
  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. Set it on the EventTermCfg: "
      "EventTermCfg(func=init_smp_state, mode='startup', "
      "params={'ckpt_path': '/path/to/pretrained.pt'})."
    )
    raise RuntimeError(msg)
  model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(
    ckpt_path, env.device
  )
  env._smp_num_conditions = model.num_conditions  # type: ignore[attr-defined]
  if model.num_conditions:
    from smp.pretrain.conditioning import sha256
    from smp.pretrain.schema import validate_schema
    from smp.rl.conditional_normalizer import ConditionalDiffNormalizer

    schema = validate_schema(model.condition_schema, model.num_conditions)
    env._smp_prior_identity = {"sha256": sha256(ckpt_path), "schema": schema}
  if conditional_score_calibration_path:
    if not model.num_conditions:
      raise ValueError("Conditional score calibration requires a conditional Prior")
    from smp.rl.conditional_score import load_score_calibration

    calibration, identity = load_score_calibration(
      conditional_score_calibration_path,
      env.device,
      expected_prior_sha256=env._smp_prior_identity["sha256"],
      expected_num_conditions=model.num_conditions,
      expected_condition_schema=model.condition_schema,
    )
    env._smp_score_calibration = calibration  # type: ignore[attr-defined]
    env._smp_score_identity = identity  # type: ignore[attr-defined]
  model = _maybe_compile(model, compile_model, compile_mode)
  env._smp_bundle = (  # type: ignore[attr-defined]
    model,
    scheduler,
    q_low,
    q_high,
    feature_dim,
    window_size,
  )
  if gsi_ckpt_path:
    (
      gsi_model,
      gsi_scheduler,
      gsi_q_low,
      gsi_q_high,
      gsi_feature_dim,
      gsi_window_size,
    ) = load_denoiser(gsi_ckpt_path, env.device)
    if (gsi_feature_dim, gsi_window_size) != (feature_dim, window_size):
      raise ValueError(
        "GSI and reward Priors must use the same feature/window shape: "
        f"reward={(feature_dim, window_size)}, "
        f"gsi={(gsi_feature_dim, gsi_window_size)}."
      )
    from smp.pretrain.conditioning import sha256

    gsi_schema = None
    if gsi_model.num_conditions:
      from smp.pretrain.schema import validate_schema

      gsi_schema = validate_schema(gsi_model.condition_schema, gsi_model.num_conditions)
    env._smp_gsi_prior_identity = {  # type: ignore[attr-defined]
      "sha256": sha256(gsi_ckpt_path),
      "schema": gsi_schema,
    }
    env._smp_gsi_num_conditions = gsi_model.num_conditions  # type: ignore[attr-defined]
    gsi_model = _maybe_compile(gsi_model, compile_model, compile_mode)
    env._smp_gsi_bundle = (  # type: ignore[attr-defined]
      gsi_model,
      gsi_scheduler,
      gsi_q_low,
      gsi_q_high,
      gsi_feature_dim,
      gsi_window_size,
    )
  else:
    env._smp_gsi_bundle = env._smp_bundle  # type: ignore[attr-defined]
    env._smp_gsi_num_conditions = env._smp_num_conditions  # type: ignore[attr-defined]
    if hasattr(env, "_smp_prior_identity"):
      env._smp_gsi_prior_identity = env._smp_prior_identity  # type: ignore[attr-defined]
  robot = env.scene["robot"]
  env._smp_ee_indexes = torch.tensor(  # type: ignore[attr-defined]
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=env.device,
  )
  if conditional_score_calibration_path:
    env._smp_normalizer = None  # type: ignore[attr-defined]
  else:
    env._smp_normalizer = (
      ConditionalDiffNormalizer(scheduler.num_timesteps, env.device)
      if env._smp_num_conditions
      else DiffNormalizer(scheduler.num_timesteps, env.device)
    )  # type: ignore[attr-defined]

  safety_cfg = {
    "enabled": gsi_reject_unsafe,
    "joint_limit_margin": gsi_joint_limit_margin,
    "max_root_planar_speed": gsi_max_root_planar_speed,
    "self_collision_sensor": gsi_self_collision_sensor,
    "max_sampling_multiplier": gsi_max_sampling_multiplier,
    "batch_size": gsi_batch_size,
  }
  env._smp_gsi_safety_cfg = safety_cfg  # type: ignore[attr-defined]
  pool, stats = _generate_gsi_windows(
    env,
    num_samples=gsi_buffer_size,
    **safety_cfg,
  )
  env._smp_gsi_pool = pool  # type: ignore[attr-defined]
  env._smp_gsi_safety_stats = stats  # type: ignore[attr-defined]
  if gsi_reject_unsafe:
    _print_gsi_safety_stats("initial pool", stats)

  if compile_model and env.num_envs != gsi_batch_size:
    # Warm the reward-path shape so its Inductor compile happens here.
    score_pair_factor = 2 if conditional_score_calibration_path else 1
    reward_batch_size = env.num_envs * score_pair_factor
    with torch.no_grad():
      dummy_x = torch.randn(
        reward_batch_size, window_size, feature_dim, device=env.device
      )
      dummy_t = torch.zeros(reward_batch_size, dtype=torch.long, device=env.device)
      if env._smp_num_conditions:
        _ = model(dummy_x, dummy_t, torch.zeros_like(dummy_t))
      else:
        _ = model(dummy_x, dummy_t)

  gsi_reset(env)


def _window_safety_masks(
  windows: torch.Tensor,
  joint_pos_limits: torch.Tensor,
  joint_limit_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return finite and last-frame joint-limit masks for GSI windows.

  The simulator only receives the last frame during GSI reset. If limits are
  per-environment, use their intersection so a retained window is safe in any
  environment it may later be sampled into.
  """
  if joint_limit_margin < 0.0:
    raise ValueError(
      f"gsi_joint_limit_margin must be non-negative, got {joint_limit_margin}."
    )
  if joint_pos_limits.ndim == 3:
    lower = joint_pos_limits[..., 0].amax(dim=0)
    upper = joint_pos_limits[..., 1].amin(dim=0)
  elif joint_pos_limits.ndim == 2:
    lower = joint_pos_limits[..., 0]
    upper = joint_pos_limits[..., 1]
  else:
    raise ValueError(
      "joint_pos_limits must have shape (J, 2) or (num_envs, J, 2); "
      f"got {tuple(joint_pos_limits.shape)}."
    )

  lower = lower + joint_limit_margin
  upper = upper - joint_limit_margin
  if torch.any(lower > upper):
    raise ValueError(
      "gsi_joint_limit_margin leaves an empty interval for at least one joint."
    )

  finite = torch.isfinite(windows).all(dim=-1).all(dim=-1)
  joint_pos = slice_features(windows)["joint_pos"][:, -1]
  within_limits = ((joint_pos >= lower) & (joint_pos <= upper)).all(dim=-1)
  return finite, within_limits


def _root_planar_speed_mask(
  windows: torch.Tensor, max_root_planar_speed: float | None
) -> torch.Tensor:
  """Keep GSI windows whose last root frame is within the task speed bound."""
  if max_root_planar_speed is None:
    return torch.ones(windows.shape[0], device=windows.device, dtype=torch.bool)
  if max_root_planar_speed <= 0.0:
    raise ValueError(
      "gsi_max_root_planar_speed must be positive when set, "
      f"got {max_root_planar_speed}."
    )
  root_lin_vel = slice_features(windows)["root_lin_vel"][:, -1, :2]
  return torch.linalg.vector_norm(root_lin_vel, dim=-1) <= max_root_planar_speed


@torch.no_grad()
def _reject_self_collisions(
  env: ManagerBasedRlEnv,
  windows: torch.Tensor,
  candidate_mask: torch.Tensor,
  sensor_name: str | None,
) -> torch.Tensor:
  """Reject candidate windows whose last-frame pose self-collides.

  Candidates are checked in chunks no larger than ``env.num_envs``. A
  kinematic ``forward`` refreshes MuJoCo contacts without advancing time.
  """
  if sensor_name is None:
    return candidate_mask
  if sensor_name not in env.scene.sensors:
    available = ", ".join(sorted(env.scene.sensors))
    raise ValueError(
      f"GSI self-collision sensor '{sensor_name}' not found. Available: {available}"
    )

  sensor = env.scene[sensor_name]
  safe = candidate_mask.clone()
  candidate_ids = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
  if candidate_ids.numel() == 0:
    return safe

  # Screening reuses the first few simulation worlds. Preserve both physical
  # state and the SMP history so a periodic gsi_refresh is observationally
  # neutral to environments that are currently mid-episode.
  max_test_envs = min(int(candidate_ids.numel()), env.num_envs)
  test_env_ids = torch.arange(max_test_envs, device=env.device)
  qpos = env.sim.data.qpos[test_env_ids].clone()
  qvel = env.sim.data.qvel[test_env_ids].clone()
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buffer_fields = (
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "ee_pos_w",
    "joint_pos",
    "joint_vel",
  )
  buffer_state = {
    name: getattr(buffer, name)[test_env_ids].clone() for name in buffer_fields
  }

  try:
    for start in range(0, int(candidate_ids.numel()), env.num_envs):
      window_ids = candidate_ids[start : start + env.num_envs]
      count = int(window_ids.numel())
      env_ids = test_env_ids[:count]
      _prime_sim_and_buffer(env, env_ids, windows[window_ids])
      env.sim.forward()
      # ``forward`` updates sensordata; reset only invalidates the Python cache.
      sensor.reset(env_ids)
      found = sensor.data.found
      if found is None:
        raise RuntimeError(
          f"GSI self-collision sensor '{sensor_name}' does not expose 'found'."
        )
      colliding = found[env_ids].bool().any(dim=-1)
      safe[window_ids[colliding]] = False
  finally:
    env.sim.data.qpos[test_env_ids] = qpos
    env.sim.data.qvel[test_env_ids] = qvel
    for name, value in buffer_state.items():
      getattr(buffer, name)[test_env_ids] = value
    env.sim.forward()
    sensor.reset()
  return safe


@torch.no_grad()
def _generate_gsi_windows(
  env: ManagerBasedRlEnv,
  num_samples: int,
  batch_size: int,
  enabled: bool,
  joint_limit_margin: float,
  max_root_planar_speed: float | None,
  self_collision_sensor: str | None,
  max_sampling_multiplier: int,
) -> tuple[torch.Tensor, dict[str, int | float]]:
  """Generate a GSI pool, optionally retaining only safe last-frame states."""
  if num_samples <= 0:
    raise ValueError(f"GSI sample count must be positive, got {num_samples}.")
  if batch_size <= 0:
    raise ValueError(f"GSI batch size must be positive, got {batch_size}.")
  if max_sampling_multiplier <= 0:
    raise ValueError(
      f"gsi_max_sampling_multiplier must be positive, got {max_sampling_multiplier}."
    )

  if not enabled:
    chunks: list[torch.Tensor] = []
    for start in range(0, num_samples, batch_size):
      chunks.append(_ddpm_sample(env, min(batch_size, num_samples - start)))
    stats: dict[str, int | float] = {
      "requested": num_samples,
      "sampled": num_samples,
      "safe_candidates": num_samples,
      "retained": num_samples,
      "nonfinite_rejected": 0,
      "joint_limit_rejected": 0,
      "root_speed_rejected": 0,
      "self_collision_rejected": 0,
      "acceptance_rate": 1.0,
    }
    return torch.cat(chunks, dim=0), stats

  max_batches = math.ceil(num_samples / batch_size) * max_sampling_multiplier
  retained_chunks: list[torch.Tensor] = []
  retained = 0
  sampled = 0
  safe_candidates = 0
  nonfinite_rejected = 0
  joint_limit_rejected = 0
  root_speed_rejected = 0
  self_collision_rejected = 0
  robot = env.scene["robot"]

  for _ in range(max_batches):
    windows = _ddpm_sample(env, batch_size)
    sampled += batch_size
    finite, within_limits = _window_safety_masks(
      windows, robot.data.joint_pos_limits, joint_limit_margin
    )
    nonfinite_rejected += int((~finite).sum().item())
    joint_limit_rejected += int((finite & ~within_limits).sum().item())
    within_speed = _root_planar_speed_mask(windows, max_root_planar_speed)
    root_speed_rejected += int((finite & within_limits & ~within_speed).sum().item())
    pre_collision = finite & within_limits & within_speed
    safe = _reject_self_collisions(env, windows, pre_collision, self_collision_sensor)
    self_collision_rejected += int((pre_collision & ~safe).sum().item())
    safe_candidates += int(safe.sum().item())

    remaining = num_samples - retained
    accepted = windows[safe][:remaining]
    if accepted.numel() > 0:
      retained_chunks.append(accepted)
      retained += int(accepted.shape[0])
    if retained == num_samples:
      break

  stats = {
    "requested": num_samples,
    "sampled": sampled,
    "safe_candidates": safe_candidates,
    "retained": retained,
    "nonfinite_rejected": nonfinite_rejected,
    "joint_limit_rejected": joint_limit_rejected,
    "root_speed_rejected": root_speed_rejected,
    "self_collision_rejected": self_collision_rejected,
    "acceptance_rate": safe_candidates / sampled if sampled else 0.0,
  }
  if retained != num_samples:
    raise RuntimeError(
      "Unable to fill the safe GSI pool within the sampling budget: "
      f"{stats}. Increase gsi_max_sampling_multiplier only after checking "
      "the prior, joint limits, and self-collision sensor."
    )
  return torch.cat(retained_chunks, dim=0), stats


def _print_gsi_safety_stats(label: str, stats: dict[str, int | float]) -> None:
  print(
    f"[INFO] Safe GSI {label}: retained={stats['retained']}, "
    f"safe_candidates={stats['safe_candidates']}/{stats['sampled']} "
    f"(acceptance={100.0 * float(stats['acceptance_rate']):.2f}%, "
    f"nonfinite={stats['nonfinite_rejected']}, "
    f"joint_limit={stats['joint_limit_rejected']}, "
    f"root_speed={stats['root_speed_rejected']}, "
    f"self_collision={stats['self_collision_rejected']})"
  )


def _prime_sim_and_buffer(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  window: torch.Tensor,
) -> None:
  """Common GSI tail: write the window's last frame to sim, fill the feature
  buffer.  The buffer is env-origin-RELATIVE (placement-invariant features) while
  the sim write adds each env's origin so robots spread across the grid.
  ``joint_vel`` is finite-differenced from ``joint_pos`` (not in the window)."""
  n, W, _ = window.shape
  E = NUM_EE
  parts = slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  joint_pos = parts["joint_pos"]
  ee_pos_local = parts["ee_pos"].reshape(n, W, E, 3)
  root_lin_vel_local = parts["root_lin_vel"]
  root_ang_vel_local = parts["root_ang_vel"]

  control_dt = float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation)
  if W > 1:
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, :-1] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / control_dt
    joint_vel[:, -1] = joint_vel[:, -2]
  else:
    joint_vel = torch.zeros_like(joint_pos)

  robot = env.scene["robot"]
  default_root = robot.data.default_root_state[env_ids].clone()
  default_pos = default_root[:, 0:3]
  default_quat = default_root[:, 3:7]
  yaw_T = yaw_quat(default_quat)
  yaw_T_W = yaw_T[:, None, :].expand(n, W, 4).reshape(-1, 4)

  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0
  world_offset_xy = quat_apply(yaw_T_W, local_xy.reshape(-1, 3)).reshape(n, W, 3)
  pelvis_pos_w = world_offset_xy.clone()
  pelvis_pos_w[..., 0] += default_pos[:, None, 0]
  pelvis_pos_w[..., 1] += default_pos[:, None, 1]
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  root_rot_local_quat = rot6d_to_quat(root_rot_6d.reshape(-1, 6)).reshape(n, W, 4)
  pelvis_quat_w = quat_mul(yaw_T_W, root_rot_local_quat.reshape(-1, 4)).reshape(n, W, 4)

  lin_vel_w = quat_apply(yaw_T_W, root_lin_vel_local.reshape(-1, 3)).reshape(n, W, 3)
  ang_vel_w = quat_apply(yaw_T_W, root_ang_vel_local.reshape(-1, 3)).reshape(n, W, 3)

  yaw_T_E = yaw_T[:, None, None, :].expand(n, W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(n, W, E, 3)
  ee_pos_w = ee_offset_w + pelvis_pos_w[:, :, None, :]

  # Buffer stays env-relative; the sim write is offset to each env's origin.
  origins = env.scene.env_origins[env_ids]
  last_root_state = torch.cat(
    [
      pelvis_pos_w[:, -1] + origins,
      pelvis_quat_w[:, -1],
      lin_vel_w[:, -1],
      ang_vel_w[:, -1],
    ],
    dim=-1,
  )
  robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos[:, -1], joint_vel[:, -1], env_ids=env_ids)

  buf: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buf.reset(
    env_ids,
    pelvis_pos_w,
    pelvis_quat_w,
    lin_vel_w,
    ang_vel_w,
    ee_pos_w,
    joint_pos,
    joint_vel,
  )


@torch.no_grad()
def _ddpm_sample(env: ManagerBasedRlEnv, n: int) -> torch.Tensor:
  """Run DDPM ancestral sampling and return ``n`` denormalized windows."""
  bundle = getattr(env, "_smp_gsi_bundle", env._smp_bundle)
  if getattr(env, "_smp_gsi_num_conditions", env._smp_num_conditions):
    from smp.pretrain.conditioning import sample_condition

    return sample_condition(bundle, "stand", n)
  model, scheduler, q_low, q_high, feature_dim, window_size = bundle
  x_t = torch.randn(n, window_size, feature_dim, device=env.device)
  for t_int in reversed(range(scheduler.num_timesteps)):
    t = torch.full((n,), t_int, dtype=torch.long, device=env.device)
    eps = model(x_t, t)
    x_t = scheduler.step(eps, x_t, t_int)
  return (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def gsi_refresh(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  num_samples: int = 1024,
  step_interval: int = 2400,
) -> None:
  """Step-mode event: every ``step_interval`` steps, FIFO-replace ``num_samples``
  GSI-pool windows with fresh DDPM samples so the init distribution stays fresh."""
  del env_ids
  cur = int(env.common_step_counter)
  if cur == 0 or (cur % step_interval) != 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  pool_size = pool.shape[0]
  if num_samples > pool_size:
    msg = f"num_samples ({num_samples}) cannot exceed pool size ({pool_size})"
    raise ValueError(msg)

  safety_cfg = getattr(env, "_smp_gsi_safety_cfg", None)
  if safety_cfg is None:
    new_windows = _ddpm_sample(env, num_samples)
  else:
    new_windows, stats = _generate_gsi_windows(
      env,
      num_samples=num_samples,
      **safety_cfg,
    )
    env._smp_gsi_refresh_safety_stats = stats  # type: ignore[attr-defined]
    if bool(safety_cfg["enabled"]):
      _print_gsi_safety_stats("refresh", stats)
  head = int(getattr(env, "_smp_gsi_head", 0))
  end = head + num_samples
  if end <= pool_size:
    pool[head:end] = new_windows
  else:
    first = pool_size - head
    pool[head:] = new_windows[:first]
    pool[: end - pool_size] = new_windows[first:]
  env._smp_gsi_head = end % pool_size  # type: ignore[attr-defined]


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Generative State Initialization: sample ``n`` windows from the GSI pool and
  prime sim + feature buffer from them.  Must run AFTER mjlab's ``reset_base``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  idx = torch.randint(0, pool.shape[0], (n,), device=env.device)
  window = pool[idx]
  _prime_sim_and_buffer(env, env_ids, window)
