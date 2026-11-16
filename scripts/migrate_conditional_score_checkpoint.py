"""Migrate the passed conditional-v2 A checkpoint to frozen V3 score metadata."""

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
from smp.rl.conditional_score import load_score_calibration, score_feature_weights


@dataclass(frozen=True)
class Cfg:
  source: str = (
    "logs/reproduction/conditional_contact_v2/rsl_rl/"
    "smp_force_unloading_conditional_g1/"
    "2026-11-11_02-34-11_conditional_contact_v2_a_seed42/model_2999.pt"
  )
  destination: str = (
    "logs/reproduction/conditional_contact_shoulder_score_v3/"
    "rsl_rl/smp_force_unloading_conditional_shoulder_score_v3/"
    "parent_a_migrated/model_2999.pt"
  )
  calibration: str = (
    "logs/reproduction/conditional_contact_shoulder_score_v3/"
    "calibration/shoulder_score_calibration.pt"
  )
  report: str = (
    "logs/reproduction/conditional_contact_shoulder_score_v3/migration/report.json"
  )
  prior: str = "datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v2.pt"
  expected_source_sha256: str = (
    "b2b6dc8f4fae3a93ce0028f3a594bceb09494590bb4b1d4f29ee24355006c3fb"
  )
  expected_prior_sha256: str = (
    "ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734"
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
  elif isinstance(left, (tuple, list)):
    if not isinstance(right, type(left)) or len(left) != len(right):
      raise RuntimeError(f"Sequence changed during migration: {path}")
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
      _assert_identical(left_item, right_item, f"{path}[{index}]")
  elif left != right:
    raise RuntimeError(f"Value changed during migration: {path}")


def migrate(cfg: Cfg) -> dict:
  source = Path(cfg.source)
  destination = Path(cfg.destination)
  if sha256(source) != cfg.expected_source_sha256:
    raise ValueError("Parent A checkpoint SHA-256 mismatch")
  if sha256(cfg.prior) != cfg.expected_prior_sha256:
    raise ValueError("Conditional v2 Prior SHA-256 mismatch")
  _, score_identity = load_score_calibration(
    cfg.calibration,
    "cpu",
    expected_prior_sha256=cfg.expected_prior_sha256,
  )
  original = torch.load(source, map_location="cpu", weights_only=True)
  required = {
    "actor_state_dict",
    "critic_state_dict",
    "optimizer_state_dict",
    "iter",
    "infos",
  }
  if required.difference(original):
    raise ValueError("Parent checkpoint is incomplete")
  if int(original["iter"]) != 2999:
    raise ValueError("Parent checkpoint is not A model_2999")
  infos = original["infos"]
  if infos.get("smp_prior", {}).get("sha256") != cfg.expected_prior_sha256:
    raise ValueError("Parent checkpoint Prior identity mismatch")
  if "smp_normalizer" not in infos:
    raise ValueError("Parent checkpoint has no legacy SMP normalizer to discard")

  migrated = copy.deepcopy(original)
  migrated_infos = migrated["infos"]
  migrated_infos.pop("smp_normalizer")
  migrated_infos["smp_score"] = score_identity
  migrated["migration"] = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_checkpoint": str(source),
    "source_checkpoint_sha256": cfg.expected_source_sha256,
    "discarded_state": ["infos.smp_normalizer"],
    "prior_sha256": cfg.expected_prior_sha256,
    "score": score_identity,
    "feature_weights": score_feature_weights().tolist(),
  }

  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    existing = torch.load(destination, map_location="cpu", weights_only=True)
    if (
      existing.get("migration", {}).get("source_checkpoint_sha256")
      != cfg.expected_source_sha256
    ):
      raise FileExistsError(
        f"Refusing to overwrite unrelated checkpoint: {destination}"
      )
  else:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(migrated, temporary)
    os.replace(temporary, destination)
  restored = torch.load(destination, map_location="cpu", weights_only=True)
  for name in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"):
    _assert_identical(original[name], restored[name], name)
  _assert_identical(
    infos["env_state"], restored["infos"]["env_state"], "infos.env_state"
  )
  _assert_identical(
    infos["smp_prior"], restored["infos"]["smp_prior"], "infos.smp_prior"
  )
  if "smp_normalizer" in restored["infos"]:
    raise RuntimeError("Legacy SMP normalizer survived migration")
  if restored["infos"].get("smp_score") != score_identity:
    raise RuntimeError("Score identity was not saved exactly")

  report = {
    "config": asdict(cfg),
    "source_sha256": cfg.expected_source_sha256,
    "destination_sha256": sha256(destination),
    "calibration_sha256": score_identity["calibration_sha256"],
    "iteration": int(restored["iter"]),
    "common_step_counter": int(restored["infos"]["env_state"]["common_step_counter"]),
    "preserved": [
      "actor_state_dict",
      "critic_state_dict",
      "optimizer_state_dict",
      "iter",
      "infos.env_state",
      "infos.smp_prior",
    ],
    "discarded": ["infos.smp_normalizer"],
    "passed": True,
  }
  report_path = Path(cfg.report)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
  temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  os.replace(temporary_report, report_path)
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)
  return report


if __name__ == "__main__":
  migrate(tyro.cli(Cfg))
