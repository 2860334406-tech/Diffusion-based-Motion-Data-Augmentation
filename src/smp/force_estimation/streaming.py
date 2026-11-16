"""Causal loading and inference for the frozen V3.57/V3.82/V3.92 force chain."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

from smp.force_estimation._components import (
  HighProbabilityResidual,
  StateResidual,
  TorsoCorrection,
  low_features,
)
from smp.force_estimation._dual_vae import load_dual_vae_force

MODEL_DIR = Path(__file__).resolve().parents[3] / "force/v392"
VERSIONS = ("v357", "v382", "v392")


def _verify_artifacts(directory: Path) -> dict:
  manifest_path = directory / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  for name, expected in manifest["sha256"].items():
    path = directory / name
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
      for chunk in iter(lambda: stream.read(1 << 20), b""):
        checksum.update(chunk)
    digest = checksum.hexdigest()
    if not hmac.compare_digest(digest, expected):
      raise ValueError(f"Checksum mismatch: {path}")
  return manifest


class StreamingForceEstimator:
  """One 50 Hz episode per instance; call ``reset`` at every episode boundary.

  ``x`` is the original SuperMimic term-major 930D observation, and
  ``tau_hist`` is ten aligned 29D actuator-force samples. No contact labels are
  accepted by this API. V3.92 additionally needs the known target site ID.
  """

  def __init__(
    self,
    model_dir: str | Path = MODEL_DIR,
    *,
    version: str = "v392",
    device: str = "cpu",
  ) -> None:
    if version not in VERSIONS:
      raise ValueError(f"version must be one of {VERSIONS}")
    if sklearn.__version__ != "1.9.0":
      raise RuntimeError(
        "The frozen joblib models require scikit-learn 1.9.0; "
        "install the force-estimator extra"
      )
    self.model_dir = Path(model_dir)
    self.manifest = _verify_artifacts(self.model_dir)
    self.version = version
    self.device = torch.device(device)
    self.base = load_dual_vae_force(self.model_dir / "base.pt", device=device)
    self.calibrator = joblib.load(self.model_dir / "calibrator.joblib")

    saved = torch.load(
      self.model_dir / "residual_v338.pt", map_location="cpu", weights_only=True
    )
    state = saved["model"]
    self.state_residual = StateResidual(
      state["x_mean"],
      state["x_std"],
      state["tau_mean"],
      state["tau_std"],
      saved.get("correction_cap_n", 30.0),
      saved.get("probability_floor", 0.0),
    )
    self.state_residual.load_state_dict(state, strict=True)
    self.state_residual = (
      self.state_residual.to(self.device).eval().requires_grad_(False)
    )
    self.lowp_classifier = joblib.load(self.model_dir / "lowp_classifier.joblib")
    self.lowp_force_head = joblib.load(self.model_dir / "lowp_force_head.joblib")

    saved = torch.load(
      self.model_dir / "residual_v370.pt", map_location="cpu", weights_only=True
    )
    if saved["epoch"] != 10 or saved["use_highrate"]:
      raise ValueError("Unexpected V3.70 checkpoint configuration")
    state = saved["model"]
    self.highp_residual = HighProbabilityResidual(
      *(
        state[key]
        for key in (
          "x_mean",
          "x_std",
          "tau_mean",
          "tau_std",
          "highrate_mean",
          "highrate_std",
        )
      )
    )
    self.highp_residual.load_state_dict(state, strict=True)
    self.highp_residual = (
      self.highp_residual.to(self.device).eval().requires_grad_(False)
    )
    self.highp_norms = {
      key: state[key].cpu().numpy()
      for key in ("x_mean", "x_std", "tau_mean", "tau_std")
    }
    self.benefit_gate = joblib.load(self.model_dir / "benefit_gate.joblib")

    with np.load(self.model_dir / "region_norms.npz", allow_pickle=False) as data:
      self.region_norms = {key: data[key] for key in data.files}
    self.region_classifier = joblib.load(self.model_dir / "region_classifier.joblib")
    if not np.array_equal(self.region_classifier.classes_, np.arange(5)):
      raise ValueError("Unexpected V3.92 region classes")
    saved = torch.load(
      self.model_dir / "force_control_v391.pt", map_location="cpu", weights_only=True
    )
    if saved["epoch"] != 3 or saved["use_region"]:
      raise ValueError("Unexpected V3.91 force checkpoint configuration")
    self.torso_correction = TorsoCorrection().to(self.device)
    self.torso_correction.load_state_dict(saved["model"], strict=True)
    self.torso_correction.eval().requires_grad_(False)
    self.previous_base: deque[np.ndarray] = deque(maxlen=2)

  def reset(self) -> None:
    self.previous_base.clear()

  @torch.inference_mode()
  def step(
    self,
    x: np.ndarray,
    tau_hist: np.ndarray,
    *,
    target_site_id: int | None = None,
  ) -> dict[str, np.ndarray | float | str]:
    x = np.asarray(x, dtype=np.float32)
    tau_hist = np.asarray(tau_hist, dtype=np.float32)
    if x.shape != (930,) or tau_hist.shape != (10, 29):
      raise ValueError(
        f"Expected x (930,), tau_hist (10,29); got {x.shape}, {tau_hist.shape}"
      )
    if not np.isfinite(x).all() or not np.isfinite(tau_hist).all():
      raise ValueError("Inputs must be finite")
    if self.version == "v392" and target_site_id not in (0, 1, 2, 3):
      raise ValueError(
        "V3.92 requires target_site_id: 0 chest, 1 left shoulder, 2 right shoulder, 3 back"
      )

    tx = torch.as_tensor(x[None], device=self.device)
    tt = torch.as_tensor(tau_hist[None], device=self.device)
    raw = self.base.predict(tx, tt)
    force = raw["force_n"][0].cpu().numpy().astype(np.float64)
    point = raw["point_m"][0].cpu().numpy().astype(np.float64)
    probability = float(raw["contact_prob"][0].cpu())
    current = np.concatenate((force / 100, point / 0.2))
    first = self.previous_base[0] if len(self.previous_base) >= 1 else np.zeros(6)
    second = self.previous_base[1] if len(self.previous_base) >= 2 else np.zeros(6)
    features = np.concatenate((current, first, second))[None]
    transformed = self.calibrator["polynomial"].transform(
      self.calibrator["scaler"].transform(features)
    )
    calibrated = (
      force + probability * 100 * self.calibrator["ridge"].predict(transformed)[0]
    ).astype(np.float32)
    self.previous_base.appendleft(current)

    v338 = (
      self.state_residual(
        tx,
        tt,
        torch.as_tensor(calibrated[None], device=self.device),
        torch.as_tensor(point[None].astype(np.float32), device=self.device),
        torch.as_tensor([probability], dtype=torch.float32, device=self.device),
      )[0]
      .cpu()
      .numpy()
      .astype(np.float32)
    )
    v357 = v338.copy()
    if probability < 0.5:
      clipped = np.clip(np.float64(probability), 1e-8, 1 - 1e-8)
      logit = np.log(clipped / (1 - clipped))
      lowp = np.concatenate(
        (
          raw["mu_s"][0].float().cpu().numpy(),
          calibrated / 100,
          point / 0.2,
          np.array([logit]),
          (tau_hist[-1] - tau_hist[-2]) / 50,
          tau_hist[-1] / 50,
        )
      ).astype(np.float32)[None]
      if self.lowp_classifier.predict_proba(lowp)[0, 1] >= 0.9999:
        proposed = (100 * self.lowp_force_head.predict(lowp)[0]).astype(np.float32)
        if np.linalg.norm(proposed) >= 15:
          v357 = proposed
    if self.version == "v357":
      return self._output(v357, point, probability)

    residual_output = (
      self.highp_residual(tx, tt, torch.as_tensor(v357[None], device=self.device))[0]
      .cpu()
      .numpy()
      .astype(np.float32)
    )
    raw_residual = residual_output - v357
    low = low_features(x[None], tau_hist[None], v357[None], self.highp_norms)
    gate_input = np.concatenate((low, (0.5 * raw_residual / 100)[None]), axis=1)
    if probability >= 0.5 and self.benefit_gate.predict_proba(gate_input)[0, 1] >= 0.5:
      v382 = v357 + 0.25 * raw_residual
    else:
      v382 = v357
    if self.version == "v382":
      return self._output(v382, point, probability)

    region_low = low_features(x[None], tau_hist[None], v357[None], self.region_norms)
    region = int(self.region_classifier.predict_proba(region_low)[0].argmax())
    if probability >= 0.5 and target_site_id in (0, 3) and region not in (0, 4):
      force_low = low_features(x[None], tau_hist[None], v382[None], self.region_norms)
      correction = (
        100
        * self.torso_correction(torch.as_tensor(force_low, device=self.device))[0]
        .cpu()
        .numpy()
      )
      v392 = v382 + correction
    else:
      v392 = v382
    return self._output(v392, point, probability)

  def _output(
    self, force: np.ndarray, point: np.ndarray, probability: float
  ) -> dict[str, np.ndarray | float | str]:
    return {
      "force_body_n": force.astype(np.float32),
      "point_body_m": point.astype(np.float32),
      "contact_probability": probability,
      "frame": "torso_link",
      "version": self.version,
    }


def load_force_estimator(
  model_dir: str | Path = MODEL_DIR,
  *,
  version: str = "v392",
  device: str = "cpu",
) -> StreamingForceEstimator:
  """Verify and load the frozen model chain from the repository bundle."""
  return StreamingForceEstimator(model_dir, version=version, device=device)
