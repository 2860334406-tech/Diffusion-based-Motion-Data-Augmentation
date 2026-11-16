"""Summarize continuous-yaw G1 shoulder-IK generation evidence."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

from smp.data.shoulder_ik import atomic_json, sha256


@dataclass(frozen=True)
class Cfg:
  input_dir: str = "logs/reproduction/g1_shoulder_ik_v2/pilot"
  output: str = ""
  figure: str = ""


def _quantiles(values: list[float]):
  array = np.asarray(values, dtype=np.float64)
  if not len(array):
    return None
  return {
    "min": float(array.min()),
    "p05": float(np.quantile(array, 0.05)),
    "p50": float(np.quantile(array, 0.50)),
    "p95": float(np.quantile(array, 0.95)),
    "max": float(array.max()),
  }


def _plot(rows: list[dict], attempts: list[dict], path: Path):
  shoulder = [row for row in rows if row["condition_id"] in (1, 2)]
  figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
  edges = np.arange(15.0, 75.0, 5.0)
  for condition, label, color in (
    (1, "left", "tab:blue"),
    (2, "right", "tab:orange"),
  ):
    selected = [row for row in shoulder if row["condition_id"] == condition]
    target = np.asarray([row["target_deg"] for row in selected])
    actual = np.asarray([row["actual_deg"] for row in selected])
    speed = np.asarray([row["speed_deg_s"] for row in selected])
    axes[0, 0].hist(actual, bins=edges, alpha=0.55, label=label, color=color)
    order = np.argsort(actual)
    axes[0, 1].plot(
      actual[order],
      (np.arange(len(actual)) + 1) / len(actual),
      label=label,
      color=color,
    )
    axes[1, 0].scatter(target, speed, s=8, alpha=0.45, label=label, color=color)
    axes[1, 1].scatter(
      target, actual - target, s=8, alpha=0.45, label=label, color=color
    )
  attempted_target = np.asarray(
    [item["target_deg"] for item in attempts if item["condition_id"] in (1, 2)]
  )
  attempted_passed = np.asarray(
    [item["passed"] for item in attempts if item["condition_id"] in (1, 2)],
    dtype=np.float64,
  )
  failure_rate = []
  for low, high in zip(edges[:-1], edges[1:], strict=True):
    mask = (attempted_target >= low) & (
      (attempted_target < high) if high < edges[-1] else (attempted_target <= high)
    )
    failure_rate.append(
      float(1.0 - attempted_passed[mask].mean()) if mask.any() else np.nan
    )
  twin = axes[0, 0].twinx()
  twin.plot((edges[:-1] + edges[1:]) / 2, failure_rate, "k.-", label="attempt failure")
  twin.set_ylabel("Attempt failure rate")
  axes[0, 0].set(xlabel="Actual amplitude [deg]", ylabel="Accepted clips")
  axes[0, 1].set(xlabel="Actual amplitude [deg]", ylabel="CDF")
  axes[1, 0].set(xlabel="Target amplitude [deg]", ylabel="Target mean speed [deg/s]")
  axes[1, 1].set(xlabel="Target amplitude [deg]", ylabel="Actual - target [deg]")
  for axis in axes.flat:
    axis.grid(alpha=0.2)
    axis.legend()
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def main(cfg: Cfg):
  root = Path(cfg.input_dir)
  manifest_path = root / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  current_identity = {
    path: sha256(Path(path)) for path in manifest["identity"]["source_identity"]
  }
  identity_ok = current_identity == manifest["identity"]["source_identity"]
  failure_checks: Counter[str] = Counter()
  accepted_attempts: Counter[str] = Counter()
  accepted_metrics: dict[str, list[float]] = defaultdict(list)
  accepted_rows: list[dict] = []
  attempt_rows: list[dict] = []
  failed_slots = []
  complete_families: Counter[str] = Counter()
  metric_names = (
    "root_height_min_m",
    "root_displacement_max_m",
    "left_contact_fraction",
    "right_contact_fraction",
    "left_foot_slip_m",
    "right_foot_slip_m",
    "joint_tracking_rmse_rad",
    "waist_tracking_rmse_rad",
    "waist_final_error_rad",
    "waist_amplitude_target_error_rad",
    "effort_p99_ratio",
    "effort_saturation_fraction",
    "max_motor_velocity_ratio",
    "hold_root_speed_p95_m_s",
    "hold_waist_speed_p95_rad_s",
    "shoulder_retreat_final_m",
  )
  for slot, record in sorted(manifest["slots"].items()):
    details = []
    for attempt in record["attempts"]:
      if "metadata" not in attempt:
        failure_checks[attempt.get("exception", "unknown_exception")] += 1
        details.append({"exception": attempt.get("exception")})
        continue
      metadata_path = root / attempt["metadata"]
      metadata = json.loads(metadata_path.read_text())
      metrics = metadata["metrics"]
      spec = metadata["spec"]
      attempt_rows.append(
        {
          "condition_id": metadata["condition_id"],
          "target_deg": float(metrics["target_waist_amplitude_deg"]),
          "speed_deg_s": float(metrics["target_mean_yaw_speed_deg_s"]),
          "passed": bool(attempt["passed"]),
        }
      )
      if not attempt["passed"]:
        failure_checks.update(metrics["failed_checks"])
      details.append(
        {
          "attempt": attempt["attempt"],
          "clip_id": attempt["clip_id"],
          "target_deg": metrics["target_waist_amplitude_deg"],
          "actual_deg": metrics["actual_waist_amplitude_deg"],
          "speed_deg_s": metrics["target_mean_yaw_speed_deg_s"],
          "entry_duration_s": spec["entry_duration_s"],
          "failed_checks": metrics["failed_checks"],
          "metadata": attempt["metadata"],
          "metadata_sha256": sha256(metadata_path),
        }
      )
    if record["status"] != "accepted":
      failed_slots.append({"slot": slot, "attempts": details})
      continue
    item = record["accepted"]
    metadata = json.loads((root / item["metadata"]).read_text())
    metrics = metadata["metrics"]
    accepted_attempts[str(item["attempt"])] += 1
    for name in metric_names:
      if name in metrics:
        accepted_metrics[name].append(float(metrics[name]))
    accepted_rows.append(
      {
        "split": item["split"],
        "condition_id": item["condition_id"],
        "target_deg": float(metrics["target_waist_amplitude_deg"]),
        "actual_deg": float(metrics["actual_waist_amplitude_deg"]),
        "speed_deg_s": float(metrics["target_mean_yaw_speed_deg_s"]),
      }
    )
    if item["condition_id"]:
      complete_families[item["family_id"]] += 1

  figure_path = Path(cfg.figure) if cfg.figure else root / "amplitude_distribution.png"
  _plot(accepted_rows, attempt_rows, figure_path)
  stage = manifest["identity"]["stage"]
  result = {
    "schema_version": 2,
    "experiment": "EXP-032",
    "manifest": manifest_path.name,
    "manifest_sha256": sha256(manifest_path),
    "status": manifest["status"],
    "gate_passed": manifest["gate_passed"],
    "source_identity_verified": identity_ok,
    "completed_counts": manifest.get("completed_counts"),
    "accepted_attempts": dict(accepted_attempts),
    "complete_shoulder_parameter_families": sum(
      count == 2 for count in complete_families.values()
    ),
    "requested_shoulder_parameter_families": 96 if stage == "pilot" else 960,
    "failure_checks_by_attempt": dict(failure_checks),
    "accepted_metric_quantiles": {
      key: _quantiles(values) for key, values in accepted_metrics.items()
    },
    "amplitude_distribution": manifest.get("amplitude_distribution"),
    "figure": str(figure_path.relative_to(root)),
    "figure_sha256": sha256(figure_path),
    "failed_slots": failed_slots,
    "decision": (
      "generation_passed"
      if manifest["gate_passed"]
      else "stop_before_next_generation_stage"
    ),
  }
  output = Path(cfg.output) if cfg.output else root / "gate_report.json"
  atomic_json(output, result)
  print(
    json.dumps(
      {"output": str(output), "status": result["status"], "failed": len(failed_slots)}
    )
  )


if __name__ == "__main__":
  main(tyro.cli(Cfg))
