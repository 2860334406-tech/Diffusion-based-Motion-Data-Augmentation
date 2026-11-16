"""Revalidate a frozen G1 shoulder-IK pilot with clip-level endpoint gates.

The original manifest is immutable evidence. This script verifies every persisted
artifact, reuses its unchanged physical metrics, recomputes endpoint semantics
from the executed qpos, and writes a derived manifest plus an audit report.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from smp.data.shoulder_ik import (
  JOINT_NAMES,
  ClipSpec,
  Gates,
  _build_model,
  _qpos_ids,
  atomic_json,
  contract_to_qpos,
  sha256,
  shoulder_endpoint_metrics,
)


@dataclass(frozen=True)
class Cfg:
  input_dir: str = "logs/reproduction/g1_shoulder_ik_v1/pilot"
  source_manifest: str = "manifest.json"
  derived_manifest: str = "manifest.endpoint_v2.json"
  report: str = "endpoint_revalidation_v2.json"


_OBSOLETE_SEMANTIC_CHECKS = {
  "phase_windows",
  "shoulder_retreat",
  "signed_final_waist",
  "stand_waist",
}
_PHYSICAL_METRIC_NAMES = (
  "foot_position_error_m",
  "foot_orientation_error_rad",
  "root_height_min_m",
  "root_roll_pitch_abs_max_rad",
  "pelvis_yaw_change_abs_max_rad",
  "root_displacement_max_m",
  "left_contact_fraction",
  "right_contact_fraction",
  "left_foot_slip_m",
  "right_foot_slip_m",
  "left_foot_yaw_drift_rad",
  "right_foot_yaw_drift_rad",
  "joint_tracking_rmse_rad",
  "waist_tracking_rmse_rad",
  "waist_final_error_rad",
  "effort_p99_ratio",
  "effort_saturation_fraction",
  "max_motor_velocity_ratio",
  "hold_root_speed_p95_m_s",
  "hold_waist_speed_p95_rad_s",
)


def _quantiles(values: list[float]) -> dict[str, float]:
  array = np.asarray(values, dtype=np.float64)
  return {
    "min": float(array.min()),
    "p05": float(np.quantile(array, 0.05)),
    "p50": float(np.quantile(array, 0.50)),
    "p95": float(np.quantile(array, 0.95)),
    "max": float(array.max()),
  }


def _load_verified_qpos(root: Path, metadata_path: Path, metadata: dict, model):
  execution_path = metadata_path.with_suffix(".npz")
  reference_path = metadata_path.with_suffix(".reference.npz")
  csv_path = metadata_path.with_suffix(".csv")
  expected = {
    execution_path: metadata["execution_npz_sha256"],
    reference_path: metadata["reference_npz_sha256"],
    csv_path: metadata["csv_sha256"],
  }
  for path, digest in expected.items():
    if not path.is_file() or sha256(path) != digest:
      raise ValueError(f"Artifact identity mismatch: {path.relative_to(root)}")
  with np.load(execution_path, allow_pickle=False) as data:
    return contract_to_qpos(
      data["root_pos"].astype(np.float64),
      data["root_rot"].astype(np.float64),
      data["dof_pos"].astype(np.float64),
      model,
    )


def _revalidate_attempt(root: Path, attempt: dict, model, gates: Gates) -> dict:
  metadata_path = root / attempt["metadata"]
  metadata = json.loads(metadata_path.read_text())
  qpos = _load_verified_qpos(root, metadata_path, metadata, model)
  spec_dict = dict(metadata["spec"])
  spec_dict["arm_variant"] = tuple(spec_dict["arm_variant"])
  spec = ClipSpec(**spec_dict)
  stored_metrics = metadata["metrics"]
  legacy_checks = stored_metrics["checks"]
  physical_checks = {
    name: bool(value)
    for name, value in legacy_checks.items()
    if name not in _OBSOLETE_SEMANTIC_CHECKS
  }
  checks: dict[str, bool] = {"unchanged_physics": all(physical_checks.values())}
  endpoint: dict[str, Any] = {}
  if spec.condition_id == 0:
    joint_qpos = _qpos_ids(model)
    waist = JOINT_NAMES.index("waist_yaw_joint")
    final_waist = float(qpos[-1, joint_qpos[waist]])
    endpoint["actual_final_waist_rad"] = final_waist
    checks["stand_waist"] = abs(final_waist) <= 0.06
  else:
    endpoint = shoulder_endpoint_metrics(spec, qpos, model, gates)
    checks.update(
      endpoint_shoulder_position=bool(endpoint["endpoint_position_reached"]),
      completion_time=bool(endpoint["completion_time_passed"]),
      hold_shoulder_position=bool(endpoint["hold_position_passed"]),
    )
  passed = all(checks.values())
  return {
    "clip_id": attempt["clip_id"],
    "family_id": attempt.get("family_id", spec.family_id),
    "split": attempt.get("split", spec.split),
    "condition_id": spec.condition_id,
    "condition": spec.condition,
    "attempt": int(attempt["attempt"]),
    "metadata": attempt["metadata"],
    "metadata_sha256": sha256(metadata_path),
    "original_passed": bool(attempt.get("passed", metadata["passed"])),
    "original_failed_checks": metadata["metrics"]["failed_checks"],
    "physical_checks": physical_checks,
    "physical_metrics": {
      name: float(stored_metrics[name]) for name in _PHYSICAL_METRIC_NAMES
    },
    "endpoint_metrics": endpoint,
    "checks": checks,
    "failed_checks": sorted(name for name, value in checks.items() if not value),
    "passed": passed,
  }


def main(cfg: Cfg):
  root = Path(cfg.input_dir)
  source_path = root / cfg.source_manifest
  source = json.loads(source_path.read_text())
  if source["identity"]["stage"] != "pilot" or len(source["slots"]) != 240:
    raise RuntimeError("Endpoint revalidation requires the complete 240-slot pilot")
  gates = Gates()
  model = _build_model(soft_limits=False)
  slots: dict[str, dict] = {}
  metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
  physical_metrics: dict[str, list[float]] = defaultdict(list)
  failures: Counter[str] = Counter()
  selected_attempts: Counter[str] = Counter()
  for slot, record in sorted(source["slots"].items()):
    attempts = []
    selected = None
    for attempt in record["attempts"]:
      result = _revalidate_attempt(root, attempt, model, gates)
      attempts.append(result)
      if selected is None and result["passed"]:
        selected = result
    if selected is None:
      for attempt in attempts:
        failures.update(attempt["failed_checks"])
      slots[slot] = {"status": "failed_after_two_attempts", "attempts": attempts}
      continue
    selected_attempts[str(selected["attempt"])] += 1
    condition = selected["condition"]
    for name, value in selected["physical_metrics"].items():
      physical_metrics[name].append(float(value))
    for name, value in selected["endpoint_metrics"].items():
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics[condition][name].append(float(value))
    slots[slot] = {
      "status": "accepted",
      "accepted": selected,
      "attempts": attempts,
    }
  accepted = sum(record["status"] == "accepted" for record in slots.values())
  gate_passed = accepted == 240
  now = datetime.now(timezone.utc).isoformat()
  source_identity_paths = (
    Path("src/smp/data/shoulder_ik.py"),
    Path("src/smp/data/shoulder_ik_dataset.py"),
    Path("scripts/revalidate_g1_shoulder_ik_endpoint.py"),
    Path("docs/G1_SHOULDER_IK_DATA_PRODUCTION.md"),
    Path("docs/GVHMR_GMR_G1_SHOULDER_DATA_REQUIREMENTS.md"),
  )
  derived = {
    "identity": {
      "schema_version": 2,
      "experiment": "EXP-031",
      "acceptance_schema": "shoulder_endpoint_task_v2",
      "stage": "pilot",
      "seed": source["identity"]["seed"],
      "gates": asdict(gates),
      "requested_counts": source["identity"]["requested_counts"],
      "source_manifest": cfg.source_manifest,
      "source_manifest_sha256": sha256(source_path),
      "source_identity": {str(path): sha256(path) for path in source_identity_paths},
    },
    "status": "complete" if gate_passed else "blocked",
    "gate_passed": gate_passed,
    "created_at_utc": now,
    "output_dir": str(root.resolve()),
    "completed_counts": {
      "requested": 240,
      "accepted": accepted,
      "failed": 240 - accepted,
    },
    "slots": slots,
  }
  derived_path = root / cfg.derived_manifest
  atomic_json(derived_path, derived)
  report = {
    "schema_version": 2,
    "experiment": "EXP-031",
    "acceptance_schema": "shoulder_endpoint_task_v2",
    "created_at_utc": now,
    "source_manifest": cfg.source_manifest,
    "source_manifest_sha256": sha256(source_path),
    "derived_manifest": cfg.derived_manifest,
    "derived_manifest_sha256": sha256(derived_path),
    "status": derived["status"],
    "gate_passed": gate_passed,
    "completed_counts": derived["completed_counts"],
    "selected_attempts": dict(selected_attempts),
    "failed_checks": dict(failures),
    "criteria": {
      "prescribed_position": (
        "selected shoulder local-x retreat in the per-frame pelvis heading frame"
      ),
      "retreat_goal_m": gates.min_shoulder_retreat_m,
      "arrival_tolerance_m": gates.shoulder_position_tolerance_m,
      "completion_deadline": "per-clip entry_duration_s",
      "hold_evaluation_s": 1.0,
      "hold_position_max_deviation_m": (gates.max_hold_shoulder_position_deviation_m),
      "phase_window_gate": False,
      "fixed_waist_amplitude_gate": False,
      "fixed_waist_speed_gate": False,
    },
    "physical_metric_quantiles": {
      name: _quantiles(values) for name, values in sorted(physical_metrics.items())
    },
    "endpoint_metric_quantiles": {
      condition: {
        name: _quantiles(values)
        for name, values in sorted(condition_metrics.items())
        if values
      }
      for condition, condition_metrics in sorted(metrics.items())
    },
    "provenance_note": (
      "The original blocked manifest and metadata are unchanged. All file hashes "
      "were verified; physical gates reuse the frozen metrics, while endpoint "
      "semantics were recomputed from persisted executed qpos."
    ),
  }
  report_path = root / cfg.report
  atomic_json(report_path, report)
  print(
    json.dumps(
      {
        "status": report["status"],
        "accepted": accepted,
        "failed": 240 - accepted,
        "manifest": str(derived_path),
        "report": str(report_path),
      }
    )
  )
  if not gate_passed:
    raise RuntimeError(f"{240 - accepted} pilot slots failed endpoint revalidation")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
