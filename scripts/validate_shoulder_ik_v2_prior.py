"""Blind-test, GSI-check and publish an accepted G1 shoulder IK-v2 Prior."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from scripts.curate_conditional_shoulder80 import atomic_json, now

from smp.pretrain.conditioning import sample_condition, sha256
from smp.pretrain.shoulder_ik_v2 import CONDITION_NAMES, CONDITION_SCHEMA
from smp.pretrain.shoulder_ik_v2_dataset import DATA_SHA, ShoulderIkV2Dataset
from smp.pretrain.shoulder_ik_v2_score import evaluate_scores
from smp.rl.conditional_score import load_score_calibration
from smp.rl.tasks.force_unloading.conditional_shoulder_ik_v2 import (
  shoulder_ik_v2_env_cfg,
)
from smp.rl.utils import load_denoiser
from smp.sampling.feature_to_state import slice_features

PUBLISHED = Path(
  "datasets/pretrain_ckpt/pretrained_g1_shoulder_ik_v2_conditional_v1.pt"
)


def _atomic_copy(source: Path, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    if sha256(destination) != sha256(source):
      raise RuntimeError(f"Refusing to overwrite a different artifact: {destination}")
    return
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  shutil.copyfile(source, temporary)
  with temporary.open("rb") as stream:
    os.fsync(stream.fileno())
  temporary.replace(destination)


@torch.no_grad()
def validate(
  checkpoint: Path,
  calibration_path: Path,
  output: Path,
  *,
  publish: Path = PUBLISHED,
) -> dict:
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  if not checkpoint.is_file() or not calibration_path.is_file():
    raise FileNotFoundError(
      checkpoint if not checkpoint.is_file() else calibration_path
    )
  checkpoint_sha = sha256(checkpoint)
  bundle = load_denoiser(str(checkpoint), "cuda:0")
  model, scheduler = bundle[:2]
  if model.condition_schema != CONDITION_SCHEMA:
    raise ValueError("Selected candidate does not use the IK-v2 schema")
  calibration, calibration_identity = load_score_calibration(
    calibration_path,
    "cuda:0",
    expected_prior_sha256=checkpoint_sha,
    expected_num_conditions=3,
    expected_condition_schema=CONDITION_SCHEMA,
  )
  if calibration["data_manifest_sha256"] != DATA_SHA:
    raise ValueError("Calibration data identity mismatch")

  test = ShoulderIkV2Dataset(split="test")
  score_report, score_raw = evaluate_scores(
    model,
    scheduler,
    test,
    calibration["reference_mean"].cpu(),
    repeats=64,
  )
  output.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output / "paired_scores_test.npz", **score_raw)

  generation = {}
  generated = {}
  initial_noise = torch.randn(4096, 10, 59, device="cuda:0")
  for condition, name in enumerate(CONDITION_NAMES):
    windows = sample_condition(
      bundle,
      condition,
      4096,
      initial_noise=initial_noise,
      guidance_scale=1.0,
    ).cpu()
    generated[condition] = windows
    generation[name] = {
      "finite": bool(torch.isfinite(windows).all()),
      "samples": len(windows),
    }

  stand_speed = torch.linalg.vector_norm(
    slice_features(generated[0])["root_lin_vel"][..., :2], dim=-1
  ).amax(-1)
  stand_speed_p95 = float(torch.quantile(stand_speed, 0.95))
  generation["stand"]["window_max_root_planar_speed_p95_m_s"] = stand_speed_p95
  direction_rates = {}
  for condition, sign in ((1, 1.0), (2, -1.0)):
    signed_final = sign * generated[condition][:, -1, 21]
    rate = float((signed_final >= 0.05).float().mean())
    direction_rates[CONDITION_NAMES[condition]] = rate
    generation[CONDITION_NAMES[condition]].update(
      expected_direction_rate=rate,
      signed_final_waist_yaw_p50_rad=float(torch.quantile(signed_final, 0.50)),
      signed_final_waist_yaw_p95_rad=float(torch.quantile(signed_final, 0.95)),
    )

  env_cfg = shoulder_ik_v2_env_cfg(
    play=True,
    prior_path=str(checkpoint),
    calibration_path=str(calibration_path),
  )
  env_cfg.scene.num_envs = 64
  init = env_cfg.events["init_smp_state"].params
  init.update(
    gsi_buffer_size=4096,
    gsi_batch_size=256,
    gsi_reject_unsafe=True,
    gsi_max_root_planar_speed=0.15,
    gsi_max_sampling_multiplier=20,
    compile_model=False,
  )
  environment = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    gsi_stats = dict(environment._smp_gsi_safety_stats)
  finally:
    environment.close()
  gsi_acceptance = float(gsi_stats["acceptance_rate"])

  checks = {
    "score_gate_passed": bool(score_report["gate"]["passed"]),
    "all_generated_samples_finite": all(
      record["finite"] for record in generation.values()
    ),
    "stand_speed_p95_le_0.15": stand_speed_p95 <= 0.15,
    "gsi_acceptance_ge_0.50": gsi_acceptance >= 0.50,
    "left_direction_rate_ge_0.80": direction_rates["left_shoulder"] >= 0.80,
    "right_direction_rate_ge_0.80": direction_rates["right_shoulder"] >= 0.80,
  }
  report = {
    "experiment": "EXP-033",
    "created_at_utc": now(),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha,
    "calibration": str(calibration_path),
    "calibration_identity": calibration_identity,
    "data_manifest_sha256": DATA_SHA,
    "test_used_for_selection": False,
    "score": score_report,
    "generation": generation,
    "gsi_safety": gsi_stats,
    "gate": {
      "passed": all(checks.values()),
      "checks": checks,
      "failed_checks": [key for key, passed in checks.items() if not passed],
    },
    "prior_published": False,
    "ppo_authorized": False,
  }
  atomic_json(output / "report.json", report)
  if not report["gate"]["passed"]:
    return report

  final_calibration = output.parent.parent / "calibration" / "final.pt"
  _atomic_copy(checkpoint, publish)
  _atomic_copy(calibration_path, final_calibration)
  report.update(
    prior_published=True,
    ppo_authorized=True,
    published_path=str(publish),
    published_sha256=sha256(publish),
    final_calibration=str(final_calibration),
    final_calibration_sha256=sha256(final_calibration),
  )
  atomic_json(output / "report.json", report)
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--calibration", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--publish", type=Path, default=PUBLISHED)
  args = parser.parse_args()
  report = validate(
    args.checkpoint,
    args.calibration,
    args.output,
    publish=args.publish,
  )
  print(json.dumps(report, indent=2))
  if not report["gate"]["passed"]:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
