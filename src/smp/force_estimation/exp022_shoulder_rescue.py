"""Experimental EXP-022 shoulder migration rescue, separate from frozen V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from smp.force_estimation.exp022_proprio import Exp022ProprioModel, episode_features


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class Exp022ShoulderRescueEstimator:
  """Return a 3D resultant and policy contact term from causal 50 Hz inputs."""

  def __init__(self, bundle_path: str | Path, *, expected_sha256: str) -> None:
    source = Path(bundle_path)
    if sha256(source) != expected_sha256:
      raise ValueError("Shoulder rescue candidate checksum mismatch")
    bundle = joblib.load(source)
    if bundle.get("name") != "EXP022 shoulder rescue pilot":
      raise ValueError("Wrong shoulder rescue bundle")
    self.bundle = bundle
    self.baseline = Exp022ProprioModel(bundle["base"])
    self.point_shoulder = np.asarray(bundle["shoulder_point_body_m"], np.float32)
    self.normal_shoulder = np.asarray(bundle["shoulder_normal_body"], np.float32)
    root = Path(__file__).resolve().parents[3] / "force/exp022_chest_v1"
    v1_manifest = json.loads((root / "manifest.json").read_text())
    mapping_path = root / "policy_mapping_calibration.json"
    if sha256(mapping_path) != v1_manifest["mapping_sha256"]:
      raise ValueError("V1 chest mapping checksum mismatch")
    mapping = json.loads(mapping_path.read_text())
    self.point_chest = np.asarray(mapping["point_body_m"], np.float32)
    self.normal_chest = np.asarray(mapping["normal_direction_body"], np.float32)
    self.frame_index = 0

  def reset(self) -> None:
    self.baseline.reset()
    self.frame_index = 0

  def step(self, x: np.ndarray, tau_hist: np.ndarray) -> dict:
    force, probability = self.baseline.step(x, tau_hist)
    features = episode_features(
      np.asarray(x, np.float32)[None],
      np.asarray(tau_hist, np.float32)[None],
      variant=self.bundle["base"]["variant"],
    )
    region_features = np.clip(
      self.bundle["region_scaler"].transform(features), -10, 10
    )
    shoulder_probability = float(
      self.bundle["region_classifier"].predict_proba(region_features)[0, 1]
    )
    proposed_features = np.clip(
      self.bundle["force_scaler"].transform(features), -10, 10
    )
    proposed_force = self.bundle["force_regressor"].predict(proposed_features)[0].astype(
      np.float32
    )
    rescued = bool(
      shoulder_probability >= self.bundle["region_threshold"]
      and np.linalg.norm(force) < 10
      and np.linalg.norm(proposed_force) >= 10
    )
    if rescued:
      force = proposed_force
      probability = max(probability, shoulder_probability)
    magnitude = float(np.linalg.norm(force))
    shoulder_mapping = bool(
      shoulder_probability >= self.bundle["region_threshold"] and magnitude > 0
    )
    if shoulder_mapping:
      normal_regressor = self.bundle["shoulder_normal_regressor"]
      if normal_regressor is None:
        normal = self.normal_shoulder
      else:
        raw_normal = normal_regressor.predict(proposed_features)[0].astype(np.float32)
        normal = raw_normal / max(1e-8, float(np.linalg.norm(raw_normal)))
      point = self.point_shoulder
    else:
      normal = self.normal_chest
      point = self.point_chest
    self.frame_index += 1
    return {
      "force_body_n": force.astype(np.float32, copy=False),
      "contact_probability": float(probability),
      "policy_normal_force_body_n": (magnitude * normal).astype(np.float32),
      "policy_point_body_m": point.copy(),
      "shoulder_probability": shoulder_probability,
      "shoulder_force_rescued": rescued,
      "shoulder_mapping_active": shoulder_mapping,
      "frame": "torso_link",
      "version": "exp022_shoulder_rescue_pilot",
    }
