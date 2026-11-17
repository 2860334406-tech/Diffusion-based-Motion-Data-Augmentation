"""Observable-only force features for EXP-022 domain-adaptation experiments."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def episode_features(
  x: np.ndarray, tau_hist: np.ndarray, *, variant: str = "raw"
) -> np.ndarray:
  """Build causal features; the optional reference is the episode's first frame."""
  x = np.asarray(x, dtype=np.float32)
  tau_hist = np.asarray(tau_hist, dtype=np.float32)
  if x.ndim != 2 or x.shape[1] != 930 or tau_hist.shape != (len(x), 10, 29):
    raise ValueError("Expected aligned x [T,930] and tau_hist [T,10,29]")
  if not np.isfinite(x).all() or not np.isfinite(tau_hist).all():
    raise ValueError("Non-finite proprioceptive input")
  if variant not in (
    "raw",
    "raw_plus_initial_delta",
    "raw_plus_acc",
    "raw_plus_acc_and_initial_delta",
    "full_history",
  ):
    raise ValueError(f"Unknown feature variant: {variant}")
  if variant == "full_history":
    return np.concatenate((x, tau_hist.reshape(len(x), -1)), axis=1)
  current = np.concatenate(
    (
      x[:, 261:290],
      x[:, 551:580],
      x[:, 607:610],
      x[:, 637:640],
      x[:, 901:930],
      x[:, 261:290] - x[:, 232:261],
      x[:, 901:930] - x[:, 872:901],
      tau_hist[:, -1],
      tau_hist[:, -1] - tau_hist[:, -2],
    ),
    axis=1,
  )
  if "acc" in variant:
    accelerations = np.concatenate(
      (
        x[:, 551:580] - x[:, 522:551],
        x[:, 607:610] - x[:, 604:607],
        x[:, 637:640] - x[:, 634:637],
        tau_hist[:, -1] - tau_hist[:, -3],
      ),
      axis=1,
    )
    current = np.concatenate((current, accelerations), axis=1)
  if "initial_delta" not in variant:
    return current.astype(np.float32, copy=False)
  reference_indices = np.r_[np.arange(261, 290), np.arange(901, 930)]
  initial_delta = np.concatenate(
    (
      x[:, reference_indices] - x[0, reference_indices],
      tau_hist[:, -1] - tau_hist[0, -1],
    ),
    axis=1,
  )
  return np.concatenate((current, initial_delta), axis=1).astype(np.float32)


class Exp022ProprioModel:
  """A candidate force head with no contact-label or future-state inputs."""

  def __init__(self, bundle: dict):
    self.bundle = bundle
    self.first_x: np.ndarray | None = None
    self.first_tau: np.ndarray | None = None
    self.frame_index = 0

  @classmethod
  def load(cls, path: str | Path) -> Exp022ProprioModel:
    return cls(joblib.load(path))

  def reset(self) -> None:
    self.first_x = None
    self.first_tau = None
    self.frame_index = 0

  def _predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = np.clip(self.bundle["scaler"].transform(features), -10, 10)
    probability = self.bundle["classifier"].predict_proba(scaled)[:, 1]
    force = self.bundle["regressor"].predict(scaled).astype(np.float32)
    force[probability < self.bundle["threshold"]] = 0
    return force, probability.astype(np.float32)

  def predict_episode(
    self, x: np.ndarray, tau_hist: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    features = episode_features(x, tau_hist, variant=self.bundle["variant"])
    force, probability = self._predict(features)
    warmup = min(len(force), self.bundle.get("warmup_frames", 0))
    force[:warmup] = 0
    probability[:warmup] = 0
    return force, probability

  def step(self, x: np.ndarray, tau_hist: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    tau_hist = np.asarray(tau_hist, dtype=np.float32)
    if x.shape != (930,) or tau_hist.shape != (10, 29):
      raise ValueError("Expected x (930,) and tau_hist (10,29)")
    if self.first_x is None:
      self.first_x = x.copy()
      self.first_tau = tau_hist.copy()
    assert self.first_tau is not None
    if "initial_delta" in self.bundle["variant"]:
      features = episode_features(
        np.stack((self.first_x, x)),
        np.stack((self.first_tau, tau_hist)),
        variant=self.bundle["variant"],
      )[-1:]
    else:
      features = episode_features(
        x[None], tau_hist[None], variant=self.bundle["variant"]
      )
    force, probability = self._predict(features)
    if self.frame_index < self.bundle.get("warmup_frames", 0):
      force[0] = 0
      probability[0] = 0
    self.frame_index += 1
    return force[0], float(probability[0])
