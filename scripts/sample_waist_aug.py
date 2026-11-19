"""Sample, physically validate and export augmented waist trajectories.

Each generated segment is described by a ``ClipSpec`` (its intended target), so
the existing ``ShoulderIkGenerator.execute`` + ``validate`` gates can be reused
verbatim: the spec defines what "pass" means for that sample.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import tyro

from smp.data.personal_retarget import (
  Config as RetargetConfig,
  PersonalRetargeter,
  load_source,
)
from smp.data.shoulder_ik import (
  FPS,
  ClipSpec,
  Motion,
  ShoulderIkGenerator,
  contract_to_qpos,
  qpos_to_csv,
  write_result,
)
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.pretrain.trajectory_sampling import (
  condition_from_targets,
  constraint_report,
  sample_trajectories,
)
from smp.pretrain.waist_aug_dataset import (
  COND_DIM,
  COND_SCALE,
  FEATURE_DIM,
  J_WAIST_YAW,
  WaistAugDataset,
  compute_conditions,
  from_38d,
  load_csv_clip,
  load_npz_clip,
  to_38d,
)


def load_anchor_clips(anchor_dir: str) -> list:
  root = Path(anchor_dir) / "clips" / "accepted"
  paths = [p for p in sorted(root.rglob("*.npz")) if not p.name.endswith(".reference.npz")]
  if not paths:
    raise FileNotFoundError(f"No accepted anchor npz under {root}")
  return [load_npz_clip(p) for p in paths]


@dataclass
class Target:
  amp_deg: float
  speed_deg_s: float
  sign: float
  abduction_deg: float
  knee_deg: float


@dataclass
class Cfg:
  ckpt: str = ""
  anchor_dir: str = "logs/reproduction/waist_aug_v1/anchors"
  out_dir: str = "logs/reproduction/waist_aug_v1/generated"
  seed_clip: str = ""
  start_timestep: int = 15
  num_steps: int = 25
  eta: float = 0.3
  guidance_scale: float = 2.0
  guidance_warmup: int = 8
  constraint_weight: float = 3.0
  constraint_stop_frac: float = 0.3
  pin_frames: int = 0
  lock_root: bool = True
  samples_per_target: int = 1
  amp_grid: str = "15,25,35,45,55,70"
  speed_grid: str = "9,13,17"
  abduction_grid: str = "15,32,55"
  knee_grid: str = "18,33"
  seed: int = 42
  device: str = ""
  skip_validation: bool = False
  smoothing_frames: int = 11
  # Add real captures to the seed candidate pool. Off by default: real clips carry
  # pelvic compensation that often fails the physical gates. Enable to bias the
  # generated style toward real motion.
  include_real_seeds: bool = False
  # When real seeds are enabled, force this fraction of samples to draw their
  # seed from the real pool. Distance matching alone never picks them: the
  # anchors cover the parameter grid better, so real windows always lose.
  real_seed_prob: float = 0.5
  real_csv_dir: str = "data/real"


def _now() -> str:
  return datetime.now(timezone.utc).isoformat()


# Checks that describe "does this trajectory track the requested target exactly".
# They are informative for coverage statistics but must not gate augmentation:
# the augmented sample is a valid motion even if its amplitude lands 40 deg
# instead of the requested 35 deg.
SEMANTIC_CHECKS = frozenset({
  "completion_time",
  "endpoint_shoulder_position",
  "hold_shoulder_position",
  "waist_amplitude_direction",
  "waist_amplitude_range",
  "waist_amplitude_target",
})


def physical_passed(metrics: dict) -> tuple[bool, list[str]]:
  fails = [
    f for f in (metrics.get("failed_checks") or [])
    if f not in SEMANTIC_CHECKS
  ]
  return len(fails) == 0, fails


def load_model(ckpt_path: str, device):
  ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  cfg = ck["cfg"]
  model = DiffusionDenoiser(
    feature_dim=cfg.get("feature_dim", FEATURE_DIM),
    window_size=cfg["window"],
    d_model=cfg["d_model"],
    nhead=cfg["nhead"],
    num_layers=cfg["num_layers"],
    dropout=0.0,
    condition_dim=cfg.get("condition_dim", COND_DIM),
  )
  model.load_state_dict(ck["model"])
  model.to(device).eval()
  return model, cfg, ck


def build_targets(cfg: Cfg) -> list[Target]:
  amps = [float(v) for v in cfg.amp_grid.split(",")]
  speeds = [float(v) for v in cfg.speed_grid.split(",")]
  abds = [float(v) for v in cfg.abduction_grid.split(",")]
  knees = [float(v) for v in cfg.knee_grid.split(",")]
  out: list[Target] = []
  i = 0
  for amp in amps:
    for speed in speeds:
      for abd in abds:
        for knee in knees:
          sign = 1.0 if i % 2 == 0 else -1.0
          out.append(Target(amp, speed, sign, abd, knee))
          i += 1
  return out


def trajectory_to_qpos(x_raw: np.ndarray, model):
  """(H,38) -> MuJoCo qpos (H, nq). Root pose at the standing origin."""
  from smp.data.shoulder_ik import JOINT_NAMES

  H = x_raw.shape[0]
  root_local, quat, joints = from_38d(x_raw)
  qpos = np.zeros((H, model.nq), dtype=np.float64)
  qpos[:, :3] = root_local
  qpos[:, 3:7] = quat[:, [3, 0, 1, 2]]  # xyzw -> wxyz
  dof_ids = [int(model.jnt_qposadr[model.joint(n).id]) for n in JOINT_NAMES]
  qpos[:, dof_ids] = joints
  return qpos, joints


def main(cfg: Cfg) -> None:
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

  model, mcfg, ck = load_model(cfg.ckpt, device)
  q_low = torch.from_numpy(ck["q_low"]).to(device)
  q_high = torch.from_numpy(ck["q_high"]).to(device)
  window = int(mcfg["window"])

  gen = PersonalRetargeter(RetargetConfig(
    smoothing_frames=cfg.smoothing_frames, allow_onset_fallback=True))
  anchor_clips = load_anchor_clips(cfg.anchor_dir)
  anchor_ds = WaistAugDataset(anchor_clips, window=window, stride=32)
  pool_cond, pool_win = anchor_ds.conditions, anchor_ds.windows
  real_cond = real_win = None
  if cfg.include_real_seeds:
    real_paths = sorted(Path(cfg.real_csv_dir).glob("P0*.csv"))
    if real_paths:
      real_ds = WaistAugDataset(
        [load_csv_clip(p, fps=30.0) for p in real_paths], window=window, stride=32
      )
      real_cond, real_win = real_ds.conditions, real_ds.windows
  print(json.dumps({"anchor_windows": len(anchor_ds),
                    "seed_pool": len(pool_cond),
                    "real_pool": 0 if real_cond is None else len(real_cond)}), flush=True)

  out = Path(cfg.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
  out.mkdir(parents=True, exist_ok=True)
  records: list[dict] = []
  targets = build_targets(cfg)

  for ti, target in enumerate(targets):
    entry = min(target.amp_deg / target.speed_deg_s, window / 30.0 - 1.2)
    cond = condition_from_targets(
      waist_amp_deg=target.amp_deg,
      waist_speed_deg_s=target.amp_deg / entry,
      sign=target.sign,
      entry_duration_s=entry,
      knee_deg=target.knee_deg,
      l_sh_roll_deg=target.abduction_deg,
      r_sh_roll_deg=target.abduction_deg,
      device=device,
    )
    cond = cond.repeat(cfg.samples_per_target, 1)
    # Weight the parameters we actually want to cover. With uniform weights the
    # 18-dim L2 lets waist/arm noise dominate and the seed's abduction collapses
    # to the population mode (~13 deg) regardless of the target.
    from smp.pretrain.waist_aug_dataset import COND_DIM as _CD
    key_w = np.ones(_CD, dtype=np.float32)
    key_w[[0, 1, 4, 6, 12]] = 8.0  # amp, speed, knee, L/R shoulder roll
    use_real = (
      real_cond is not None and float(np.random.uniform()) < cfg.real_seed_prob
    )
    src_cond = real_cond if use_real else pool_cond
    src_win = real_win if use_real else pool_win
    dist = (((src_cond - cond[0].cpu())
             * torch.from_numpy(key_w)) ** 2).sum(1)
    seed = src_win[int(dist.argmin())][None]
    x = sample_trajectories(
      model, scheduler=DDPMScheduler(num_timesteps=mcfg.get("num_timesteps", 100)).to(device),
      condition=cond, num_steps=cfg.num_steps, eta=cfg.eta,
      guidance_scale=cfg.guidance_scale, guidance_warmup=cfg.guidance_warmup,
      constraint_weight=cfg.constraint_weight, constraint_stop_frac=cfg.constraint_stop_frac,
      q_low=q_low, q_high=q_high, window=window,
      init_clean=seed, start_timestep=cfg.start_timestep,
      pin_frames=cfg.pin_frames,
    )
    # A waist twist does not need the root to move. Locking the generated root
    # pose to the seed's removes drift that would otherwise topple the controller;
    # only the 29 joints remain generated.
    if seed is not None and cfg.lock_root:
      x = x.clone()
      x[:, :, :9] = seed[:, :, :9].to(x.device)
    report = constraint_report(x)
    for si in range(x.shape[0]):
      x_raw = x[si].cpu().numpy()
      # Validate against the trajectory's MEASURED attributes, not the requested
      # target: augmentation needs coverage, not exact hits on a grid point.
      cm = compute_conditions(x_raw)
      amp_deg = float(cm[0] * COND_SCALE[0]) * 57.2957795
      speed_deg_s = float(cm[1] * COND_SCALE[1])
      meas_sign = 1.0 if cm[2] >= 0 else -1.0
      meas_entry = float(np.clip(amp_deg / max(speed_deg_s, 1e-3), 0.4, window / 30.0 - 1.0))
      meas_abduct = abs(float(cm[6] * COND_SCALE[6])) * 57.2957795
      meas_knee = float(cm[4] * COND_SCALE[4]) * 57.2957795
      condition_id = 1 if meas_sign > 0 else 2
      clip_id = (f"aug-a{int(round(amp_deg)):02d}-v{int(round(speed_deg_s)):02d}"
                 f"-abd{int(round(meas_abduct)):02d}-k{int(round(meas_knee)):02d}"
                 f"-{'L' if meas_sign > 0 else 'R'}-{si:02d}")
      spec = ClipSpec(
        clip_id=clip_id, family_id="aug", stage="formal", split="train",
        condition_id=condition_id, grid_index=0, slot_index=ti, attempt=0,
        seed=cfg.seed, yaw_amplitude=math.radians(amp_deg),
        entry_duration_s=meas_entry, hold_duration_s=1.5,
        stance_width_m=0.24, arm_variant=(0.0, 0.0, 0.0, 0.0),
        com_offset_y_m=0.0, target_mean_yaw_speed_deg_s=speed_deg_s,
        shoulder_roll_rad=math.radians(meas_abduct),
        knee_flexion_rad=math.radians(meas_knee),
      )
      if cfg.skip_validation:
        records.append({"clip_id": clip_id, "passed": None, "constraints": report})
        continue
      try:
        # IK post-process: the generated joints become the soft upper-body
        # reference; the legs are re-solved against the CoM/foot constraints so
        # the result is dynamically feasible. See docs on retargeting.
        root_p, quat_p, joints_p = from_38d(x_raw)
        joints_raw = joints_p.copy()
        # PersonalRetargeter's onset detector needs >=1 s of neutral prefix and a
        # >=1 s terminal hold. Pad with the MEDIAN first-second pose rather than
        # the first frame: seeds taken from real clips often start mid-motion, so
        # repeating frame 0 would not read as "neutral".
        pad = int(round(1.5 * 30.0))
        rest_r = np.median(root_p[:30], axis=0, keepdims=True)
        rest_q = np.median(quat_p[:30], axis=0, keepdims=True)
        rest_q /= np.linalg.norm(rest_q, axis=-1, keepdims=True)
        rest_j = np.median(joints_p[:30], axis=0, keepdims=True)
        root_p = np.concatenate([np.repeat(rest_r, pad, 0), root_p,
                                 np.repeat(root_p[-1:], pad, 0)])
        quat_p = np.concatenate([np.repeat(rest_q, pad, 0), quat_p,
                                 np.repeat(quat_p[-1:], pad, 0)])
        joints_p = np.concatenate([np.repeat(rest_j, pad, 0), joints_p,
                                   np.repeat(joints_p[-1:], pad, 0)])
        src_dir = out / "sources"
        src_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
          src_dir / f"{clip_id}.npz", fps=np.float32(30.0),
          root_pos=root_p, root_rot=quat_p, dof_pos=joints_p,
        )
        item = {
          "metadata": {"clip_id": clip_id, "condition_id": condition_id,
                       "subject_id": "P01"},
          "identity": {},
        }
        source = load_source(src_dir, item, gen.ik_model, gen.cfg)
        # Preserve the generated posture: by default the personal-retarget IK uses
        # a fixed stance table and ~11 deg abduction, which would discard the
        # arm/knee intent the augmentation is supposed to add.
        from smp.pretrain.waist_aug_dataset import (
          J_L_KNEE, J_R_KNEE, L_ARM_JOINTS,
        )
        jr = joints_raw
        posture_overrides = {
          "knee_flexion_rad": float(np.median(
            0.5 * (jr[:, J_L_KNEE] + jr[:, J_R_KNEE]))),
          "shoulder_roll_rad": float(np.median(np.abs(jr[:, L_ARM_JOINTS[0]]))),
          "shoulder_yaw_rad": float(np.median(np.abs(jr[:, L_ARM_JOINTS[1]]))),
          "shoulder_pitch_rad": float(np.median(jr[:, L_ARM_JOINTS[2]])),
          "elbow_rad": float(np.median(jr[:, L_ARM_JOINTS[3]])),
          "wrist_roll_rad": float(np.median(np.abs(jr[:, L_ARM_JOINTS[4]]))),
        }
        qpos_ref, ik = gen.reference(source, 0, posture_overrides)
        execution, contacts = gen.execute_checked(qpos_ref)
        metrics = gen.validate(source, qpos_ref, execution, ik, contacts)
        passed = bool(metrics.get("passed"))
        # write_result() reports this key; PersonalRetargeter's validator does not
        # compute it, so derive it from the generated joints.
        metrics.setdefault(
          "actual_waist_amplitude_deg",
          float(np.degrees(np.abs(
            joints_p[:, J_WAIST_YAW]
            - np.median(joints_p[:30, J_WAIST_YAW])).max())),
        )
        frames = qpos_ref.shape[0]
        motion = Motion(
          qpos=qpos_ref,
          phase_target=np.zeros(frames, dtype=np.int8),
          reference_progress=np.ones(frames, dtype=np.float32),
        )
        write_result(out, spec, passed, motion, execution, metrics, gen.sim_model)
        records.append({
          "clip_id": clip_id, "passed": passed,
          "failed_checks": metrics.get("failed_checks"),
          "waist_amp_deg": float(np.degrees(np.abs(
            joints_p[:, J_WAIST_YAW]
            - np.median(joints_p[:30, J_WAIST_YAW])).max())),
          "ik_com_margin_m": ik.get("minimum_ik_com_margin_m"),
          "constraints": report,
        })
      except Exception as error:  # noqa: BLE001
        records.append({"clip_id": clip_id, "passed": False,
                        "exception": f"{type(error).__name__}: {error}"})
    print(json.dumps({"target": asdict(target), "constraints": report}), flush=True)

  ok = [r for r in records if r.get("passed")]
  summary = {
    "finished_at_utc": _now(), "targets": len(targets),
    "samples": len(records), "passed": len(ok),
    "pass_rate": round(len(ok) / max(1, len(records)), 3),
    "records": records,
  }
  (out / "summary.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))


if __name__ == "__main__":
  tyro.cli(main)