"""Re-retarget the real captures with a pelvic-yaw cap.

Real video retargeting lets the pelvis turn ~30 deg, which makes the motion
physically invalid for a feet-planted G1 (and, after the augmentation pipeline,
drag every generated sample back toward the procedural-anchor style). Capping
pelvic yaw transfers that rotation into the relative waist angle and produces
real-motion-style trajectories that can pass the same physical gates.

Output CSVs keep the 36-column layout, so they drop straight into
``train_waist_aug.py --cfg.real-csv-dir``.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from smp.data.personal_retarget import Config, PersonalRetargeter, load_source


@dataclass
class Cfg:
  real_csv_dir: str = "data/real"
  out_dir: str = "data/real_pelvis_capped"
  pelvis_yaw_limit: float = 0.0524  # 3 deg
  smoothing_frames: int = 11


def main(cfg: Cfg) -> None:
  paths = sorted(glob.glob(str(Path(cfg.real_csv_dir) / "P0*.csv")))
  if not paths:
    raise FileNotFoundError(cfg.real_csv_dir)
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  tmp = out / "_sources"
  tmp.mkdir(exist_ok=True)

  rt = PersonalRetargeter(
    Config(pelvis_yaw_limit=cfg.pelvis_yaw_limit,
           smoothing_frames=cfg.smoothing_frames)
  )

  ok = 0
  for p in paths:
    a = np.loadtxt(p, delimiter=",", dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 36:
      print(f"skip {p}: shape {a.shape}")
      continue
    cid = Path(p).stem
    # The onset detector requires >=1 s of neutral prefix and a terminal hold.
    # Real takes often start mid-motion or end abruptly, so pin both ends with
    # repeated boundary frames (physically a standstill).
    pad = int(round(1.2 * 30.0))
    root_p = np.concatenate([np.repeat(a[:1, :3], pad, 0), a[:, :3],
                             np.repeat(a[-1:, :3], pad, 0)])
    quat_p = np.concatenate([np.repeat(a[:1, 3:7], pad, 0), a[:, 3:7],
                             np.repeat(a[-1:, 3:7], pad, 0)])
    dof_p = np.concatenate([np.repeat(a[:1, 7:36], pad, 0), a[:, 7:36],
                            np.repeat(a[-1:, 7:36], pad, 0)])
    np.savez(tmp / f"{cid}.npz", fps=np.float32(30.0), root_pos=root_p,
             root_rot=quat_p, dof_pos=dof_p)
    cond_id = 0 if "stand" in cid else (1 if "left" in cid else 2)
    item = {"metadata": {"clip_id": cid, "condition_id": cond_id,
                         "subject_id": cid.split("_")[0]}, "identity": {}}
    try:
      source = load_source(tmp, item, rt.ik_model, rt.cfg)
      ref, _ik = rt.reference(source, 0, {})
      joints = np.asarray([q[rt.ik_joint_qpos] for q in ref])
      arr = np.concatenate([ref[:, :3], ref[:, 3:7], joints], axis=1)
      np.savetxt(out / f"{cid}.csv", arr, delimiter=",", fmt="%.9g")
      ok += 1
      print(f"ok {cid}: frames={len(arr)}")
    except Exception as error:  # noqa: BLE001
      print(f"fail {cid}: {type(error).__name__}: {error}")
  print(f"done {ok}/{len(paths)} -> {out}")


if __name__ == "__main__":
  tyro.cli(main)