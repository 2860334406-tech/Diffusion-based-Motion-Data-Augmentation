"""EXP-022 chest-only force estimator and policy contact mapping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from smp.force_estimation.exp022_proprio import Exp022ProprioModel

MODEL_DIR = Path(__file__).resolve().parents[3] / "force/exp022_chest_v1"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class Exp022ChestEstimator:
  """Stream a 3D resultant and a chest-specific normal/point for the actor.

  This package is limited to the EXP-022 front-chest sphere protocol. The
  policy normal and point are calibrated constants, not general point or
  multi-contact estimates. Call reset at every episode boundary.
  """

  def __init__(self, model_dir: str | Path = MODEL_DIR) -> None:
    root = Path(model_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["name"] != "EXP022 chest force and actor mapping v1":
      raise ValueError("Wrong estimator package")
    model_file = root / "candidate.joblib"
    mapping_file = root / "policy_mapping_calibration.json"
    if _sha256(model_file) != manifest["model_sha256"]:
      raise ValueError("EXP022 force model checksum mismatch")
    if _sha256(mapping_file) != manifest["mapping_sha256"]:
      raise ValueError("EXP022 policy mapping checksum mismatch")
    if (
      _sha256(Path(__file__).with_name("exp022_proprio.py"))
      != manifest["feature_code_sha256"]
    ):
      raise ValueError("EXP022 causal feature code checksum mismatch")
    mapping = json.loads(mapping_file.read_text())
    self.model = Exp022ProprioModel.load(model_file)
    self.normal_body = np.asarray(mapping["normal_direction_body"], np.float32)
    self.point_body = np.asarray(mapping["point_body_m"], np.float32)
    if (
      self.normal_body.shape != (3,)
      or self.point_body.shape != (3,)
      or not np.isfinite(self.normal_body).all()
      or not np.isfinite(self.point_body).all()
      or not np.isclose(np.linalg.norm(self.normal_body), 1.0, atol=1e-4)
    ):
      raise ValueError("Invalid chest policy mapping")

  def reset(self) -> None:
    self.model.reset()

  def step(self, x: np.ndarray, tau_hist: np.ndarray) -> dict:
    force, probability = self.model.step(x, tau_hist)
    magnitude = float(np.linalg.norm(force))
    return {
      "force_body_n": force,
      "contact_probability": probability,
      "policy_normal_force_body_n": (magnitude * self.normal_body).astype(np.float32),
      "policy_point_body_m": self.point_body.copy(),
      "frame": "torso_link",
      "version": "exp022_chest_v1",
    }


def load_exp022_chest_estimator(
  model_dir: str | Path = MODEL_DIR,
) -> Exp022ChestEstimator:
  """Verify and load the front-chest domain adapter from its repository bundle."""
  return Exp022ChestEstimator(model_dir)
