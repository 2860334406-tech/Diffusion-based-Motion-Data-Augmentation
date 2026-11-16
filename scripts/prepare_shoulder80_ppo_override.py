"""Prepare the explicit failed-Prior override for Shoulder80 PPO.

The source Prior and A checkpoint remain immutable.  This creates a runtime
calibration wrapper and a metadata-only PPO migration in the isolated output.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tyro

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder80 import CONDITION_SCHEMA
from smp.rl.conditional_score import (
  FIXED_TIMESTEPS,
  SCORE_MODE,
  SCORE_SCHEMA_VERSION,
  load_score_calibration,
  score_feature_weights,
)


@dataclass(frozen=True)
class Cfg:
  prior: str = (
    "logs/reproduction/conditional_shoulder80_prior_v1_expanded/"
    "prior/checkpoint_02000.pt"
  )
  expected_prior_sha256: str = (
    "73d9b0409066650525c3bb619f8440af947c2a96f3697f8774ed23d859c90cbe"
  )
  source_calibration: str = (
    "logs/reproduction/conditional_shoulder80_prior_v1_expanded/"
    "prior/validation/epoch_02000/calibration.pt"
  )
  expected_source_calibration_sha256: str = (
    "e21aa52a1f68feab39dcc3fe597ca1ff5545dd205a0e3a85edced6de116a0f34"
  )
  source_report: str = (
    "logs/reproduction/conditional_shoulder80_prior_v1_expanded/"
    "prior/validation/epoch_02000/report.json"
  )
  runtime_calibration: str = (
    "logs/reproduction/conditional_shoulder80_ppo_v1_override/"
    "calibration/runtime_calibration.pt"
  )
  source_checkpoint: str = (
    "logs/reproduction/conditional_contact_v2/rsl_rl/"
    "smp_force_unloading_conditional_g1/"
    "2026-11-11_02-34-11_conditional_contact_v2_a_seed42/model_2999.pt"
  )
  expected_source_checkpoint_sha256: str = (
    "b2b6dc8f4fae3a93ce0028f3a594bceb09494590bb4b1d4f29ee24355006c3fb"
  )
  expected_source_prior_sha256: str = (
    "ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734"
  )
  destination_checkpoint: str = (
    "logs/reproduction/conditional_shoulder80_ppo_v1_override/rsl_rl/"
    "smp_force_unloading_conditional_shoulder80_v1/"
    "parent_a_migrated_gsi_v2/model_2999.pt"
  )
  report: str = (
    "logs/reproduction/conditional_shoulder80_ppo_v1_override/migration/report.json"
  )


def _assert_identical(left: Any, right: Any, path: str) -> None:
  if isinstance(left, torch.Tensor):
    if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
      raise RuntimeError(f"Tensor changed during migration: {path}")
  elif isinstance(left, dict):
    if not isinstance(right, dict) or left.keys() != right.keys():
      raise RuntimeError(f"Mapping changed during migration: {path}")
    for key in left:
      _assert_identical(left[key], right[key], f"{path}.{key}")
  elif isinstance(left, (list, tuple)):
    if not isinstance(right, type(left)) or len(left) != len(right):
      raise RuntimeError(f"Sequence changed during migration: {path}")
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
      _assert_identical(a, b, f"{path}[{index}]")
  elif left != right:
    raise RuntimeError(f"Value changed during migration: {path}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  os.replace(temporary, path)


def prepare(cfg: Cfg) -> dict[str, Any]:
  prior = Path(cfg.prior)
  source_calibration = Path(cfg.source_calibration)
  source_checkpoint = Path(cfg.source_checkpoint)
  source_report = Path(cfg.source_report)
  if sha256(prior) != cfg.expected_prior_sha256:
    raise ValueError("Shoulder80 epoch-2000 Prior SHA-256 mismatch")
  if sha256(source_calibration) != cfg.expected_source_calibration_sha256:
    raise ValueError("Shoulder80 source calibration SHA-256 mismatch")
  if sha256(source_checkpoint) != cfg.expected_source_checkpoint_sha256:
    raise ValueError("Conditional-v2 A checkpoint SHA-256 mismatch")

  prior_payload = torch.load(prior, map_location="cpu", weights_only=False)
  if (
    prior_payload.get("condition_schema") != CONDITION_SCHEMA
    or prior_payload.get("cfg", {}).get("num_conditions") != 3
    or prior_payload.get("epoch") != 2000
  ):
    raise ValueError("Unexpected Shoulder80 Prior schema or epoch")
  formal_report = json.loads(source_report.read_text())
  if formal_report["gate"]["passed"]:
    raise ValueError("Override preparation expected a failed formal Prior gate")

  source = torch.load(source_calibration, map_location="cpu", weights_only=True)
  if (
    source.get("prior_sha256") != cfg.expected_prior_sha256
    or source.get("condition_schema") != CONDITION_SCHEMA
    or source.get("score_mode") != SCORE_MODE
    or tuple(source.get("fixed_timesteps", ())) != FIXED_TIMESTEPS
    or not torch.equal(source.get("feature_weights"), score_feature_weights())
    or tuple(source.get("reference_mean", torch.empty(0)).shape)
    != (3, len(FIXED_TIMESTEPS))
  ):
    raise ValueError("Shoulder80 calibration content mismatch")
  runtime_payload = {
    "score_schema_version": SCORE_SCHEMA_VERSION,
    "mode": SCORE_MODE,
    "prior_sha256": cfg.expected_prior_sha256,
    "data_manifest_sha256": source["data_manifest_sha256"],
    "normalization_sha256": source["normalization_sha256"],
    "condition_schema": CONDITION_SCHEMA,
    "fixed_timesteps": list(FIXED_TIMESTEPS),
    "feature_weights": source["feature_weights"].clone(),
    "reference_mean": source["reference_mean"].clone(),
    "samples_per_condition": int(source["samples_per_condition"]),
    "seed": int(source["seed"]),
    "source_calibration_sha256": cfg.expected_source_calibration_sha256,
    "prior_acceptance_override": True,
  }
  runtime_path = Path(cfg.runtime_calibration)
  runtime_path.parent.mkdir(parents=True, exist_ok=True)
  if runtime_path.exists():
    existing = torch.load(runtime_path, map_location="cpu", weights_only=True)
    _assert_identical(runtime_payload, existing, "runtime_calibration")
  else:
    temporary = runtime_path.with_suffix(runtime_path.suffix + ".tmp")
    torch.save(runtime_payload, temporary)
    os.replace(temporary, runtime_path)
  _, score_identity = load_score_calibration(
    runtime_path,
    "cpu",
    expected_prior_sha256=cfg.expected_prior_sha256,
    expected_num_conditions=3,
    expected_condition_schema=CONDITION_SCHEMA,
  )

  original = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
  required = {
    "actor_state_dict",
    "critic_state_dict",
    "optimizer_state_dict",
    "iter",
    "infos",
  }
  if required.difference(original) or int(original["iter"]) != 2999:
    raise ValueError("Parent A checkpoint is incomplete or not model_2999")
  infos = original["infos"]
  if infos.get("smp_prior", {}).get("sha256") != cfg.expected_source_prior_sha256:
    raise ValueError("Parent A checkpoint source Prior mismatch")
  if "smp_normalizer" not in infos:
    raise ValueError("Parent A checkpoint lacks its legacy SMP normalizer")

  migrated = copy.deepcopy(original)
  migrated_infos = migrated["infos"]
  migrated_infos.pop("smp_normalizer")
  migrated_infos["smp_gsi_prior"] = copy.deepcopy(infos["smp_prior"])
  migrated_infos["smp_prior"] = {
    "sha256": cfg.expected_prior_sha256,
    "schema": CONDITION_SCHEMA,
  }
  migrated_infos["smp_score"] = score_identity
  migrated_infos["prior_acceptance_override"] = True
  migrated["migration"] = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_sha256": cfg.expected_source_checkpoint_sha256,
    "source_prior_sha256": cfg.expected_source_prior_sha256,
    "gsi_prior_sha256": cfg.expected_source_prior_sha256,
    "destination_prior_sha256": cfg.expected_prior_sha256,
    "discarded_state": ["infos.smp_normalizer"],
    "metadata_replaced": ["infos.smp_prior"],
    "metadata_added": ["infos.smp_gsi_prior"],
    "score": score_identity,
    "prior_gate_passed": False,
    "prior_gate_failed_checks": formal_report["gate"]["failed_checks"],
    "user_authorized_override": True,
  }
  destination = Path(cfg.destination_checkpoint)
  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    if (
      restored.get("migration", {}).get("source_checkpoint_sha256")
      != cfg.expected_source_checkpoint_sha256
    ):
      raise FileExistsError(f"Refusing to overwrite unrelated file: {destination}")
  else:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(migrated, temporary)
    os.replace(temporary, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=True)
  for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"):
    _assert_identical(original[key], restored[key], key)
  _assert_identical(
    infos["env_state"], restored["infos"]["env_state"], "infos.env_state"
  )

  report = {
    "config": asdict(cfg),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "passed": True,
    "user_authorized_failed_prior_override": True,
    "prior_gate_passed": False,
    "prior_gate_failed_checks": formal_report["gate"]["failed_checks"],
    "runtime_calibration_sha256": sha256(runtime_path),
    "destination_checkpoint_sha256": sha256(destination),
    "preserved_exactly": [
      "actor_state_dict",
      "critic_state_dict",
      "optimizer_state_dict",
      "iter",
      "infos.env_state",
    ],
    "changed_metadata": [
      "infos.smp_prior",
      "infos.smp_gsi_prior",
      "infos.smp_score",
    ],
    "discarded": ["infos.smp_normalizer"],
  }
  _atomic_json(Path(cfg.report), report)
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)
  return report


if __name__ == "__main__":
  prepare(tyro.cli(Cfg))
