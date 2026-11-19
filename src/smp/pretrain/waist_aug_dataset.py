"""Trajectory-level dataset for waist-twist data augmentation (GTA-style).

Per-frame 38-dim representation:
  [0:3]   root_pos   xy relative to the segment's first frame, z keeps world height
  [3:9]   root_rot   6D tan-normal (rotation-matrix col0 + col2), expressed in
                     the segment's first-frame yaw-only frame
  [9:38]  29 joint angles (rad)

The 6D form matches ``scripts/csv_to_npz.py::_tan_norm_from_quat`` so features stay
interchangeable with the existing prior stack, but the anchor here is the segment's
FIRST frame (causal) instead of the last frame.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Must match ``src/smp/data/shoulder_ik.py::JOINT_NAMES`` (29 joints, G1).
JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
  "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
NUM_JOINTS = 29
FEATURE_DIM = 3 + 6 + NUM_JOINTS  # 38

# Joint indices used by the condition vector.
J_WAIST_YAW = JOINT_NAMES.index("waist_yaw_joint")
J_WAIST_ROLL = JOINT_NAMES.index("waist_roll_joint")
J_WAIST_PITCH = JOINT_NAMES.index("waist_pitch_joint")
J_L_KNEE = JOINT_NAMES.index("left_knee_joint")
J_R_KNEE = JOINT_NAMES.index("right_knee_joint")
J_L_HIP_YAW = JOINT_NAMES.index("left_hip_yaw_joint")
J_R_HIP_YAW = JOINT_NAMES.index("right_hip_yaw_joint")
L_ARM_JOINTS = (
  JOINT_NAMES.index("left_shoulder_roll_joint"),
  JOINT_NAMES.index("left_shoulder_yaw_joint"),
  JOINT_NAMES.index("left_shoulder_pitch_joint"),
  JOINT_NAMES.index("left_elbow_joint"),
  JOINT_NAMES.index("left_wrist_roll_joint"),
)
R_ARM_JOINTS = (
  JOINT_NAMES.index("right_shoulder_roll_joint"),
  JOINT_NAMES.index("right_shoulder_yaw_joint"),
  JOINT_NAMES.index("right_shoulder_pitch_joint"),
  JOINT_NAMES.index("right_elbow_joint"),
  JOINT_NAMES.index("right_wrist_roll_joint"),
)

# Condition layout (18 scalars, already divided by an empirical scale).
COND_NAMES: tuple[str, ...] = (
  "waist_amp", "waist_speed", "waist_sign", "entry_duration",
  "knee", "hip_yaw",
  "l_sh_roll", "l_sh_yaw", "l_sh_pitch", "l_elbow", "l_wr_roll",
  "r_sh_roll", "r_sh_yaw", "r_sh_pitch", "r_elbow", "r_wr_roll",
  "init_waist_pitch", "init_waist_roll",
)
COND_DIM = len(COND_NAMES)
COND_SCALE = np.array(
  [1.22, 40.0, 1.0, 6.5,
   0.70, 0.30,
   1.20, 0.50, 0.40, 1.30, 0.50,
   1.20, 0.50, 0.40, 1.30, 0.50,
   0.05, 0.05],
  dtype=np.float32,
)
COND_CLIP = 3.0

ONSET_THRESHOLD_RAD = 0.05
ONSET_BASELINE_FRAMES = 30
ONSET_CONFIRM_FRAMES = 5


# --------------------------------------------------------------------------- #
# quaternion / 6D helpers (pure NumPy, xyzw everywhere)
# --------------------------------------------------------------------------- #
def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
  """(...,4) xyzw unit quaternion -> (...,3,3) rotation matrix."""
  q = np.asarray(q, dtype=np.float64)
  x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  xx, yy, zz = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = w * x, w * y, w * z
  m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
  m[..., 0, 0] = 1 - 2 * (yy + zz)
  m[..., 0, 1] = 2 * (xy - wz)
  m[..., 0, 2] = 2 * (xz + wy)
  m[..., 1, 0] = 2 * (xy + wz)
  m[..., 1, 1] = 1 - 2 * (xx + zz)
  m[..., 1, 2] = 2 * (yz - wx)
  m[..., 2, 0] = 2 * (xz - wy)
  m[..., 2, 1] = 2 * (yz + wx)
  m[..., 2, 2] = 1 - 2 * (xx + yy)
  return m


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
  """(...,3,3) rotation matrix -> (...,4) xyzw unit quaternion."""
  m = np.asarray(m, dtype=np.float64)
  tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
  q = np.zeros(m.shape[:-2] + (4,), dtype=np.float64)
  # Four candidate solutions; pick the numerically dominant one per element.
  big = tr > 0
  s = np.sqrt(np.clip(tr + 1.0, 1e-12, None)) * 2.0
  q[..., 3] = np.where(big, 0.25 * s, q[..., 3])
  q[..., 0] = np.where(big, (m[..., 2, 1] - m[..., 1, 2]) / s, q[..., 0])
  q[..., 1] = np.where(big, (m[..., 0, 2] - m[..., 2, 0]) / s, q[..., 1])
  q[..., 2] = np.where(big, (m[..., 1, 0] - m[..., 0, 1]) / s, q[..., 2])

  c0 = (~big) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
  s0 = np.sqrt(np.clip(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-12, None)) * 2.0
  q[..., 0] = np.where(c0, 0.25 * s0, q[..., 0])
  q[..., 1] = np.where(c0, (m[..., 0, 1] + m[..., 1, 0]) / s0, q[..., 1])
  q[..., 2] = np.where(c0, (m[..., 0, 2] + m[..., 2, 0]) / s0, q[..., 2])
  q[..., 3] = np.where(c0, (m[..., 2, 1] - m[..., 1, 2]) / s0, q[..., 3])

  c1 = (~big) & (~c0) & (m[..., 1, 1] > m[..., 2, 2])
  s1 = np.sqrt(np.clip(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 1e-12, None)) * 2.0
  q[..., 0] = np.where(c1, (m[..., 0, 1] + m[..., 1, 0]) / s1, q[..., 0])
  q[..., 1] = np.where(c1, 0.25 * s1, q[..., 1])
  q[..., 2] = np.where(c1, (m[..., 1, 2] + m[..., 2, 1]) / s1, q[..., 2])
  q[..., 3] = np.where(c1, (m[..., 0, 2] - m[..., 2, 0]) / s1, q[..., 3])

  c2 = (~big) & (~c0) & (~c1)
  s2 = np.sqrt(np.clip(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 1e-12, None)) * 2.0
  q[..., 0] = np.where(c2, (m[..., 0, 2] + m[..., 2, 0]) / s2, q[..., 0])
  q[..., 1] = np.where(c2, (m[..., 1, 2] + m[..., 2, 1]) / s2, q[..., 1])
  q[..., 2] = np.where(c2, 0.25 * s2, q[..., 2])
  q[..., 3] = np.where(c2, (m[..., 1, 0] - m[..., 0, 1]) / s2, q[..., 3])

  q /= np.linalg.norm(q, axis=-1, keepdims=True)
  # Canonical sign: w >= 0 keeps consecutive frames continuous.
  q = np.where(q[..., 3:4] < 0, -q, q)
  return q.astype(np.float32)


def quat_xyzw_to_tan_norm(q: np.ndarray) -> np.ndarray:
  """(...,4) xyzw -> (...,6): rotation-matrix col0 then col2."""
  m = quat_xyzw_to_matrix(q)
  return np.concatenate([m[..., :, 0], m[..., :, 2]], axis=-1).astype(np.float32)


def tan_norm_to_quat_xyzw(v6: np.ndarray) -> np.ndarray:
  """(...,6) -> (...,4) xyzw via Gram-Schmidt on the two 3-vectors."""
  v6 = np.asarray(v6, dtype=np.float64)
  a = v6[..., 0:3]
  b = v6[..., 3:6]
  a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9, None)
  b = b - np.sum(a * b, axis=-1, keepdims=True) * a
  b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9, None)
  # Columns must satisfy col0 x col1 = col2, so col1 = col2 x col0.
  c = np.cross(b, a)
  m = np.stack([a, c, b], axis=-1)  # [col0, col1, col2]
  return matrix_to_quat_xyzw(m)


def yaw_of_quat_xyzw(q: np.ndarray) -> np.ndarray:
  x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat_xyzw(q: np.ndarray) -> np.ndarray:
  yaw = yaw_of_quat_xyzw(q)
  out = np.zeros(q.shape[:-1] + (4,), dtype=np.float64)
  out[..., 2] = np.sin(yaw / 2.0)
  out[..., 3] = np.cos(yaw / 2.0)
  return out.astype(np.float32)


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ],
    axis=-1,
  ).astype(np.float32)


def quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
  return np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1)


def rotate_vec_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  m = quat_xyzw_to_matrix(q)
  return np.einsum("...ij,...j->...i", m, v)


# --------------------------------------------------------------------------- #
# clip I/O and 38-dim packing
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
  root_pos: np.ndarray      # (T,3) world
  root_quat: np.ndarray     # (T,4) xyzw
  joints: np.ndarray        # (T,29)
  fps: float = 30.0
  source: str = ""


def load_csv_clip(path: str | Path, fps: float = 30.0) -> Clip:
  """Load a header-less 36-column G1 CSV (root xyz + quat xyzw + 29 joints)."""
  arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
  if arr.ndim != 2 or arr.shape[1] != 36:
    raise ValueError(f"{path}: expected (T,36), got {arr.shape}")
  quat = arr[:, 3:7].copy()
  quat /= np.linalg.norm(quat, axis=1, keepdims=True)
  return Clip(root_pos=arr[:, :3], root_quat=quat, joints=arr[:, 7:36],
              fps=fps, source=str(path))


def load_npz_clip(path: str | Path) -> Clip:
  with np.load(path) as z:
    return Clip(
      root_pos=z["root_pos"].astype(np.float64),
      root_quat=z["root_rot"].astype(np.float64),
      joints=z["dof_pos"].astype(np.float64),
      fps=float(z["fps"]) if "fps" in z else 30.0,
      source=str(path),
    )


def to_38d(clip: Clip) -> np.ndarray:
  """Pack a whole clip into (T,38); anchor = the clip's first frame."""
  T = clip.joints.shape[0]
  yaw0 = yaw_quat_xyzw(clip.root_quat[:1])[0]
  yaw0_inv = quat_conj_xyzw(yaw0[None, :])[0]

  # root_pos: xy relative to frame 0, then rotated into the yaw-only frame;
  # z keeps world height (gravity direction is already the world -z).
  dyaw = rotate_vec_xyzw(np.repeat(yaw0_inv[None, :], T, axis=0),
                         clip.root_pos - clip.root_pos[:1])
  root_pos = np.concatenate([dyaw[:, :2], clip.root_pos[:, 2:3]], axis=1)

  # root_rot: heading-invariant orientation, then 6D.
  rel = quat_mul_xyzw(np.repeat(yaw0_inv[None, :], T, axis=0), clip.root_quat)
  root_rot = quat_xyzw_to_tan_norm(rel)

  return np.concatenate([root_pos, root_rot, clip.joints], axis=1).astype(np.float32)


def from_38d(x38: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Unpack (...,38) -> world root_pos (...,3), quat xyzw (...,4), joints (...,29).

  The anchor frame is unknown after generation, so the result is expressed in the
  segment's own first-frame yaw frame. Downstream callers that need a world pose
  should compose it with a chosen first-frame pose.
  """
  x38 = np.asarray(x38, dtype=np.float64)
  root_pos_local = x38[..., 0:3]
  root_rot = tan_norm_to_quat_xyzw(x38[..., 3:9])
  joints = x38[..., 9:38]
  return root_pos_local.astype(np.float32), root_rot, joints.astype(np.float32)


# --------------------------------------------------------------------------- #
# onset detection / slicing / conditions
# --------------------------------------------------------------------------- #
def locate_onset(waist_yaw: np.ndarray) -> tuple[int, float]:
  """Return (t0, baseline). Falls back to the peak angular-speed frame, then 60."""
  n = min(ONSET_BASELINE_FRAMES, len(waist_yaw))
  baseline = float(np.median(waist_yaw[:n]))
  dev = np.abs(waist_yaw - baseline)
  above = dev > ONSET_THRESHOLD_RAD
  run = 0
  for i, flag in enumerate(above):
    run = run + 1 if flag else 0
    if run >= ONSET_CONFIRM_FRAMES:
      return i - ONSET_CONFIRM_FRAMES + 1, baseline
  # Fallback: frame of maximum smoothed angular speed.
  if len(waist_yaw) > 5:
    speed = np.abs(np.gradient(np.convolve(
      waist_yaw, np.ones(5) / 5.0, mode="same")))
    return int(np.argmax(speed)), baseline
  return min(60, len(waist_yaw) - 1), baseline


def compute_conditions(seg: np.ndarray, fps: float = 30.0) -> np.ndarray:
  """18-dim condition vector for one (H,38) segment, pre-normalised."""
  waist_yaw = seg[:, 9 + J_WAIST_YAW].astype(np.float64)
  t0, baseline = locate_onset(waist_yaw)
  dev = waist_yaw - baseline
  amp = float(dev[np.argmax(np.abs(dev))])
  abs_amp = abs(amp)
  sign = 1.0 if amp >= 0 else -1.0

  thr = 0.9 * abs_amp
  rise = int(np.argmax(np.abs(dev) >= thr)) if abs_amp > 1e-6 else 0
  entry_s = max(rise / fps, 1.0 / fps)
  speed_deg_s = np.degrees(abs_amp) / entry_s

  n0 = min(ONSET_BASELINE_FRAMES, len(seg))
  j = seg[:, 9:38]
  c = np.array(
    [
      abs_amp,
      speed_deg_s,
      sign,
      entry_s,
      float(np.median(0.5 * (j[:, J_L_KNEE] + j[:, J_R_KNEE]))),
      float(np.median(0.5 * (j[:, J_L_HIP_YAW] + j[:, J_R_HIP_YAW]))),
      *[float(np.median(j[:n0, k])) for k in L_ARM_JOINTS],
      *[float(np.median(j[:n0, k])) for k in R_ARM_JOINTS],
      float(np.median(j[:n0, J_WAIST_PITCH])),
      float(np.median(j[:n0, J_WAIST_ROLL])),
    ],
    dtype=np.float32,
  )
  c = c / COND_SCALE
  return np.clip(c, -COND_CLIP, COND_CLIP).astype(np.float32)


def slice_segments(
  x38: np.ndarray, window: int, stride: int, onset: int, n_onset: int = 2,
) -> np.ndarray:
  """Cut (T,38) into (N,window,38); half the windows are anchored on the onset."""
  T = x38.shape[0]
  if T < window:
    pad = np.repeat(x38[-1:], window - T, axis=0)
    return np.concatenate([x38, pad], axis=0)[None]

  starts = list(range(0, T - window + 1, stride))
  onset_starts = [
    onset - window // 2 + k * (window // 4)
    for k in range(n_onset)
  ]
  for s in onset_starts:
    s = int(np.clip(s, 0, T - window))
    if s not in starts:
      starts.append(s)
  starts = sorted(set(starts))
  return np.stack([x38[s : s + window] for s in starts]).astype(np.float32)


def resample_to_fps(clip: Clip, target_fps: float = 30.0) -> Clip:
  """Linear/slerp resample; anchors are produced at 50 Hz and stored at 30 Hz."""
  if abs(clip.fps - target_fps) < 1e-6:
    return clip
  T = clip.joints.shape[0]
  new_T = max(2, int(round((T - 1) * target_fps / clip.fps)) + 1)
  src = np.linspace(0.0, T - 1.0, T)
  dst = np.linspace(0.0, T - 1.0, new_T)
  joints = np.stack([np.interp(dst, src, clip.joints[:, k]) for k in range(NUM_JOINTS)], axis=1)
  root_pos = np.stack([np.interp(dst, src, clip.root_pos[:, k]) for k in range(3)], axis=1)
  # Slerp for the root orientation.
  idx0 = np.clip(np.floor(dst).astype(int), 0, T - 1)
  idx1 = np.clip(idx0 + 1, 0, T - 1)
  frac = (dst - idx0)[:, None]
  q0, q1 = clip.root_quat[idx0], clip.root_quat[idx1]
  dot = np.clip(np.sum(q0 * q1, axis=1, keepdims=True), -1.0, 1.0)
  q1 = np.where(dot < 0, -q1, q1)
  dot = np.abs(dot)
  theta = np.arccos(dot)
  sin_theta = np.sin(theta)
  safe = np.where(sin_theta > 1e-6, sin_theta, 1.0)
  w0 = np.where(sin_theta > 1e-6, np.sin((1 - frac) * theta) / safe, 1 - frac)
  w1 = np.where(sin_theta > 1e-6, np.sin(frac * theta) / safe, frac)
  quat = w0 * q0 + w1 * q1
  quat /= np.linalg.norm(quat, axis=1, keepdims=True)
  return Clip(root_pos=root_pos, root_quat=quat, joints=joints,
              fps=target_fps, source=clip.source)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WaistAugDataset(Dataset):
  """Returns ``(x:(window,38), c:(18,))`` for trajectory-level diffusion."""

  def __init__(
    self,
    clips: list[Clip],
    window: int = 160,
    stride: int = 32,
    n_onset_windows: int = 2,
    with_conditions: bool = True,
  ) -> None:
    self.window = window
    self.with_conditions = with_conditions
    xs: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    self.sources: list[str] = []
    for clip in clips:
      clip = resample_to_fps(clip, 30.0)
      x38 = to_38d(clip)
      waist_yaw = x38[:, 9 + J_WAIST_YAW]
      onset, _ = locate_onset(waist_yaw)
      segs = slice_segments(x38, window, stride, onset, n_onset_windows)
      for seg in segs:
        xs.append(seg)
        cs.append(compute_conditions(seg) if with_conditions else np.zeros(COND_DIM, np.float32))
        self.sources.append(clip.source)

    self.windows = torch.from_numpy(np.stack(xs))
    self.conditions = torch.from_numpy(np.stack(cs))
    self.feature_dim = FEATURE_DIM

  def __len__(self) -> int:
    return self.windows.shape[0]

  def __getitem__(self, index: int):
    if self.with_conditions:
      return self.windows[index], self.conditions[index]
    return self.windows[index]

  # ---- normalisation (q01/q99 over the training split) ----
  def compute_norm_stats(self, q_low: float = 0.01, q_high: float = 0.99):
    flat = self.windows.reshape(-1, self.feature_dim).numpy()
    lo = np.quantile(flat, q_low, axis=0).astype(np.float32)
    hi = np.quantile(flat, q_high, axis=0).astype(np.float32)
    span = hi - lo
    tiny = span < 1e-6
    hi[tiny] = lo[tiny] + 1.0
    return lo, hi


def normalise(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
  return 2.0 * (x - lo) / (hi - lo) - 1.0


def denormalise(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
  return (x + 1.0) * 0.5 * (hi - lo) + lo


def load_real_clips(csv_dir: str | Path) -> list[Clip]:
  paths = sorted(Path(csv_dir).glob("P0*.csv"))
  if not paths:
    raise FileNotFoundError(f"No P0*.csv under {csv_dir}")
  return [load_csv_clip(p, fps=30.0) for p in paths]