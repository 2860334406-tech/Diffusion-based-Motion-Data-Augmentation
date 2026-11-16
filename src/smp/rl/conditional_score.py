"""Frozen calibration and pure helpers for contrastive conditional SMP scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from smp.pretrain.conditioning import sha256

SCORE_SCHEMA_VERSION = 1
SCORE_MODE = "shoulder_mirror_contrast_v1"
FIXED_TIMESTEPS = (8, 15, 22)
FEATURE_DIM = 59
WAIST_YAW_FEATURE_INDEX = 21
WAIST_YAW_FEATURE_WEIGHT = 10.0
CONTRAST_TEMPERATURE = 4.0
MARGIN_EPSILON = 1.0e-6


def score_feature_weights(
  device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """Return the frozen 59-D score weights (waist yaw is the sole 10x term)."""
  weights = torch.ones(FEATURE_DIM, device=device, dtype=dtype)
  weights[WAIST_YAW_FEATURE_INDEX] = WAIST_YAW_FEATURE_WEIGHT
  return weights


def shoulder_score_condition_ids(
  condition_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Map unsupported incidental contacts to stand and return a support mask.

  The shoulder-only Prior has no chest/back embeddings. Those contacts must not
  crash training or receive a shoulder reward, so they are evaluated safely as
  stand and masked out of the final reward.
  """
  supported = (condition_ids >= 0) & (condition_ids <= 2)
  return torch.where(
    supported, condition_ids, torch.zeros_like(condition_ids)
  ), supported


def opposite_shoulder_conditions(condition_ids: torch.Tensor) -> torch.Tensor:
  """Swap left/right shoulder IDs while leaving all other conditions unchanged."""
  return torch.where(
    condition_ids == 1,
    torch.full_like(condition_ids, 2),
    torch.where(condition_ids == 2, torch.ones_like(condition_ids), condition_ids),
  )


def paired_condition_predictions(
  model,
  x_t: torch.Tensor,
  timesteps: torch.Tensor,
  correct_condition: torch.Tensor,
  opposite_condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Evaluate correct/opposite conditions on byte-identical noisy inputs."""
  batch = x_t.shape[0]
  if not (
    timesteps.shape == correct_condition.shape == opposite_condition.shape == (batch,)
  ):
    raise ValueError("Paired score IDs and timesteps must have shape (B,)")
  prediction = model(
    torch.cat((x_t, x_t), dim=0),
    torch.cat((timesteps, timesteps), dim=0),
    torch.cat((correct_condition, opposite_condition), dim=0),
  )
  if prediction.shape != (2 * batch, *x_t.shape[1:]):
    raise ValueError("Conditional denoiser returned an unexpected paired shape")
  return prediction.chunk(2, dim=0)


def weighted_denoising_mse(
  prediction: torch.Tensor,
  target: torch.Tensor,
  feature_weights: torch.Tensor,
  time_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Per-sample weighted MSE with an optional per-frame validity mask."""
  if prediction.shape != target.shape or prediction.ndim != 3:
    raise ValueError("prediction and target must have matching (B, W, F) shapes")
  if feature_weights.shape != (prediction.shape[-1],):
    raise ValueError("feature_weights must match the feature dimension")
  if time_mask is None:
    time_mask = torch.ones(
      prediction.shape[:2], device=prediction.device, dtype=prediction.dtype
    )
  elif time_mask.shape != prediction.shape[:2]:
    raise ValueError("time_mask must have shape (B, W)")
  else:
    time_mask = time_mask.to(device=prediction.device, dtype=prediction.dtype)
  weights = feature_weights.to(device=prediction.device, dtype=prediction.dtype)
  numerator = ((prediction - target).square() * weights * time_mask[..., None]).sum(
    dim=(-1, -2)
  )
  denominator = time_mask.sum(-1).clamp_min(1.0) * weights.sum()
  return numerator / denominator


def contrastive_score_from_energies(
  correct_energies: torch.Tensor,
  opposite_energies: torch.Tensor,
  reference_means: torch.Tensor,
  shoulder_mask: torch.Tensor,
  temperature: float = CONTRAST_TEMPERATURE,
  epsilon: float = MARGIN_EPSILON,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Combine per-timestep energies into the frozen base x mirror score."""
  if not (
    correct_energies.shape == opposite_energies.shape == reference_means.shape
    and correct_energies.ndim == 2
  ):
    raise ValueError("energies and reference_means must have matching (B, K) shapes")
  if shoulder_mask.shape != correct_energies.shape[:1]:
    raise ValueError("shoulder_mask must have shape (B,)")
  if not torch.isfinite(reference_means).all() or (reference_means <= 0).any():
    raise ValueError("reference means must be finite and positive")
  normalized_correct = correct_energies / reference_means
  base_energy = normalized_correct.mean(-1)
  base_reward = torch.exp(-torch.relu(base_energy - 1.0))
  margins = (opposite_energies - correct_energies) / (
    0.5 * (opposite_energies + correct_energies) + epsilon
  )
  margin = margins.mean(-1)
  contrast_reward = torch.sigmoid(temperature * margin)
  contrast_reward = torch.where(
    shoulder_mask, contrast_reward, torch.ones_like(contrast_reward)
  )
  reward = (base_reward * contrast_reward).clamp_(0.0, 1.0)
  components = {
    "correct_energy": correct_energies.mean(-1),
    "opposite_energy": opposite_energies.mean(-1),
    "normalized_correct_energy": base_energy,
    "contrast_margin": margin,
    "base_reward": base_reward,
    "contrast_reward": contrast_reward,
  }
  return reward, components


def calibration_identity(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
  """Small immutable identity saved in every v3 PPO checkpoint."""
  return {
    "score_schema_version": SCORE_SCHEMA_VERSION,
    "mode": SCORE_MODE,
    "calibration_sha256": sha256(path),
    "prior_sha256": payload["prior_sha256"],
    "data_manifest_sha256": payload["data_manifest_sha256"],
    "fixed_timesteps": list(payload["fixed_timesteps"]),
    "waist_yaw_feature_index": WAIST_YAW_FEATURE_INDEX,
    "waist_yaw_feature_weight": WAIST_YAW_FEATURE_WEIGHT,
    "contrast_temperature": CONTRAST_TEMPERATURE,
  }


def load_score_calibration(
  path: str | Path,
  device: torch.device | str,
  *,
  expected_prior_sha256: str,
  expected_num_conditions: int = 5,
  expected_condition_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Load and fully validate a frozen score calibration artifact."""
  source = Path(path)
  if not source.is_file():
    raise FileNotFoundError(f"Conditional score calibration not found: {source}")
  payload = torch.load(source, map_location="cpu", weights_only=True)
  required = {
    "score_schema_version",
    "mode",
    "prior_sha256",
    "data_manifest_sha256",
    "fixed_timesteps",
    "feature_weights",
    "reference_mean",
    "samples_per_condition",
    "seed",
  }
  if not isinstance(payload, dict) or required.difference(payload):
    raise ValueError("Invalid conditional score calibration payload")
  if payload["score_schema_version"] != SCORE_SCHEMA_VERSION:
    raise ValueError("Conditional score schema version mismatch")
  if payload["mode"] != SCORE_MODE:
    raise ValueError("Conditional score mode mismatch")
  if payload["prior_sha256"] != expected_prior_sha256:
    raise ValueError("Conditional score calibration Prior mismatch")
  if tuple(payload["fixed_timesteps"]) != FIXED_TIMESTEPS:
    raise ValueError("Conditional score timestep mismatch")
  expected_weights = score_feature_weights()
  weights = payload["feature_weights"].float().cpu()
  if weights.shape != (FEATURE_DIM,) or not torch.equal(weights, expected_weights):
    raise ValueError("Conditional score feature weights mismatch")
  means = payload["reference_mean"].float().cpu()
  if means.shape != (expected_num_conditions, len(FIXED_TIMESTEPS)):
    raise ValueError("Conditional score reference mean shape mismatch")
  if not torch.isfinite(means).all() or (means <= 0).any():
    raise ValueError("Conditional score reference means must be finite and positive")
  if expected_condition_schema is not None:
    from smp.pretrain.schema import validate_schema

    expected = validate_schema(expected_condition_schema, expected_num_conditions)
    if payload.get("condition_schema") != expected:
      raise ValueError("Conditional score condition schema mismatch")
  runtime = {
    **payload,
    "feature_weights": weights.to(device),
    "reference_mean": means.to(device),
  }
  return runtime, calibration_identity(source, payload)
