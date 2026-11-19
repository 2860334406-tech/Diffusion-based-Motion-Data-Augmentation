"""Export generated 36-column CSVs into the repository's (N,10,59) window format.

Uses ``ExecutedFk`` + ``compute_window_features`` (pure MuJoCo forward kinematics,
no renderer), so it works on headless servers where ``csv_to_npz.py`` fails on EGL.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from smp.data.shoulder_ik_dataset import ExecutedFk, compute_window_features


@dataclass
class Cfg:
  input_glob: str = "logs/reproduction/waist_aug_v1/full_grid2/*/clips/accepted/**/*.csv"
  out_dir: str = "logs/reproduction/waist_aug_v1/windows"


def _load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  a = np.loadtxt(path, delimiter=",", dtype=np.float64)
  if a.ndim != 2 or a.shape[1] != 36:
    raise ValueError(f"{path}: expected (T,36), got {a.shape}")
  return a[:, :3], a[:, 3:7], a[:, 7:36]


def main(cfg: Cfg) -> None:
  paths = sorted(glob.glob(cfg.input_glob, recursive=True))
  if not paths:
    raise FileNotFoundError(cfg.input_glob)
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fk = ExecutedFk()
  total_windows = 0
  for p in paths:
    root, quat, dof = _load(p)
    ee = fk.positions(root, quat, dof)
    starts, windows = compute_window_features(root, quat, dof, ee)
    name = Path(p).stem
    np.savez_compressed(
      out / f"{name}.npz", windows=windows.astype(np.float32),
      fps=np.float32(50.0), window_size=np.int32(windows.shape[1]),
      stride=np.int32(5),
    )
    total_windows += windows.shape[0]
  print(f"files={len(paths)} windows={total_windows} out={out}")


if __name__ == "__main__":
  tyro.cli(main)