"""Run the six fixed endpoint probes for continuous-yaw shoulder IK v2."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import tyro

from smp.data.shoulder_ik import (
  FPS,
  WAIST_YAW_TRACKING_KP,
  ClipSpec,
  ShoulderIkGenerator,
  atomic_json,
  retry_spec,
  sha256,
  write_result,
)


@dataclass(frozen=True)
class Cfg:
  output_dir: str = "logs/reproduction/g1_shoulder_ik_v2/boundary_smoke"
  seed: int = 42
  speed_deg_s: float = 15.0


def make_boundary_specs(cfg: Cfg) -> list[ClipSpec]:
  specs = []
  for slot, amplitude_deg in enumerate((15.0, 42.5, 70.0)):
    family = f"boundary-{amplitude_deg:g}deg"
    duration = math.ceil(amplitude_deg / cfg.speed_deg_s * FPS) / FPS
    for condition, side in ((1, "left"), (2, "right")):
      specs.append(
        ClipSpec(
          clip_id=f"{family}-{side}-a0",
          family_id=family,
          stage="smoke",
          split="smoke",
          condition_id=condition,
          grid_index=0,
          slot_index=slot,
          attempt=0,
          seed=cfg.seed,
          yaw_amplitude=math.radians(amplitude_deg),
          entry_duration_s=duration,
          hold_duration_s=2.0,
          stance_width_m=0.24,
          knee_flexion_rad=0.70,
          arm_variant=(0.0, 0.0, 0.0, 0.0),
          com_offset_y_m=0.0,
          target_mean_yaw_speed_deg_s=cfg.speed_deg_s,
        )
      )
  return specs


def _identity(cfg: Cfg, specs: list[ClipSpec]) -> dict:
  source_paths = (
    Path("src/smp/data/shoulder_ik.py"),
    Path("scripts/smoke_g1_shoulder_ik_v2.py"),
  )
  return {
    "schema_version": 1,
    "experiment": "EXP-032",
    "purpose": "continuous_yaw_boundary_smoke",
    "seed": cfg.seed,
    "speed_deg_s": cfg.speed_deg_s,
    "waist_yaw_tracking_kp": WAIST_YAW_TRACKING_KP,
    "specs": [asdict(spec) for spec in specs],
    "source_identity": {str(path): sha256(path) for path in source_paths},
  }


def main(cfg: Cfg):
  specs = make_boundary_specs(cfg)
  output = Path(cfg.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  manifest_path = output / "manifest.json"
  identity = _identity(cfg, specs)
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != identity:
      raise RuntimeError("Existing boundary-smoke identity mismatch")
  else:
    manifest = {
      "identity": identity,
      "status": "running",
      "gate_passed": False,
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "slots": {},
    }
    atomic_json(manifest_path, manifest)

  generator = ShoulderIkGenerator()
  for index, spec in enumerate(specs, 1):
    slot = f"{spec.family_id}:{spec.condition_id}"
    if slot in manifest["slots"]:
      continue
    attempts = []
    accepted = None
    for candidate in (spec, retry_spec(spec)):
      passed, motion, execution, metrics = generator.produce(candidate)
      record = write_result(
        output,
        candidate,
        passed,
        motion,
        execution,
        metrics,
        generator.sim_model,
      )
      attempts.append(record)
      if passed:
        accepted = record
        break
    manifest["slots"][slot] = (
      {"status": "accepted", "accepted": accepted, "attempts": attempts}
      if accepted
      else {"status": "failed_after_two_attempts", "attempts": attempts}
    )
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(
      f"[{index}/{len(specs)}] {slot} {manifest['slots'][slot]['status']}", flush=True
    )

  accepted = [
    record["accepted"]
    for record in manifest["slots"].values()
    if record["status"] == "accepted"
  ]
  manifest["completed_counts"] = {
    "requested": len(specs),
    "accepted": len(accepted),
    "failed": len(specs) - len(accepted),
  }
  manifest["gate_passed"] = len(accepted) == len(specs)
  manifest["status"] = "complete" if manifest["gate_passed"] else "blocked"
  manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
  atomic_json(manifest_path, manifest)
  print(json.dumps({"status": manifest["status"], **manifest["completed_counts"]}))
  if not manifest["gate_passed"]:
    raise RuntimeError("Continuous-yaw boundary smoke failed")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
