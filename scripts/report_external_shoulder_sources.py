"""Create reproducible CPU plots and source summaries from a completed review."""

from __future__ import annotations

import csv
import json
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.csv_to_npz import JOINT_NAMES
from scripts.curate_conditional_contact import screening_model
from scripts.curate_conditional_shoulder80 import (
  Cfg,
  classify_windows,
  nonoverlap_count,
)
from scripts.review_external_shoulder_sources import (
  OUTPUT,
  digest,
  load_motion,
  save_json,
  utc_now,
)


def main():
  directory = OUTPUT / "review"
  manifest = json.loads((directory / "manifest.json").read_text())
  plan = json.loads((OUTPUT / "plan.json").read_text())
  audits = json.loads((directory / "audit.json").read_text())["sources"]
  sources = {r["source_id"]: r for r in plan["sources"]}
  with (directory / "windows.csv").open() as stream:
    rows = list(csv.DictReader(stream))
  report_path = directory / "source_summary.json"
  if report_path.exists():
    raise RuntimeError("Source report already exists; no overwrite")
  summary = {
    "created_at_utc": utc_now(),
    "review_manifest_sha256": digest(directory / "manifest.json"),
    "script_sha256": digest(__file__),
    "gate": manifest["gate"],
    "source_counts": dict(Counter(s["dataset"] for s in sources.values())),
    "amass_subset_counts": dict(
      Counter(
        s["remote_path"].split("/")[1]
        for s in sources.values()
        if s["dataset"] == "amass_g1"
      )
    ),
    "by_dataset": {},
    "g1_moves_clips": [],
    "illustrations": [],
  }
  for dataset in ("lafan", "g1_moves", "amass_g1"):
    selected = [r for r in rows if r["dataset"] == dataset]
    shoulder = [r for r in selected if int(r["condition_id"]) != 0]
    summary["by_dataset"][dataset] = {
      "windows_by_condition": [
        sum(int(r["condition_id"]) == c for r in selected) for c in range(3)
      ],
      "shoulder_nonoverlap_windows": nonoverlap_count(
        [
          (r["source_sha256"], int(r["frame50_start"]), int(r["frame50_end_exclusive"]))
          for r in shoulder
        ]
      ),
      "shoulder_original_files": len({r["source_sha256"] for r in shoulder}),
      "shoulder_actors": len({r["source_group"] for r in shoulder}),
      "shoulder_files_by_split": {
        split: sorted({r["source_id"] for r in shoulder if r["split"] == split})
        for split in ("train", "validation", "test")
      },
    }
  model, _, joint_ids, _, _ = screening_model()
  limits = model.jnt_range[joint_ids]
  for audit in audits:
    if audit["dataset"] != "g1_moves":
      continue
    selected = [r for r in rows if r["source_id"] == audit["source_id"]]
    _, motion = load_motion(
      OUTPUT / "raw" / audit["source_id"], "g1_moves", audit["fps"]
    )
    classified = classify_windows(motion, Cfg())
    frame_ids = np.unique(classified["indices"][classified["labels"] >= 0])
    positions = motion[frame_ids, 7:]
    violation = (
      (positions < limits[:, 0] + 1e-4) | (positions > limits[:, 1] - 1e-4)
    ).sum(0)
    summary["g1_moves_clips"].append(
      {
        "source_id": audit["source_id"],
        "raw_stats": audit["raw_stats"],
        "accepted_including_mirror_by_condition": [
          sum(int(r["condition_id"]) == c for r in selected) for c in range(3)
        ],
        "original_unmirrored_audit": audit["variants"][0],
        "joint_limit_rejections_original_candidate_frames": {
          JOINT_NAMES[i]: int(count) for i, count in enumerate(violation) if count
        },
      }
    )
  # Representative illustrations, not an additional selection/acceptance rule.
  examples = [
    s
    for s in sources
    if any(
      f"/{clip}/" in s for clip in ("M_Move5", "M_Move19", "M_Move3", "B_SpiralDance")
    )
  ]
  for split in ("train", "validation", "test"):
    candidates = sorted(
      {
        r["source_id"]
        for r in rows
        if r["dataset"] == "amass_g1"
        and r["split"] == split
        and int(r["phase_id"]) == 1
      }
    )
    if candidates:
      examples.append(candidates[0])
  plot_dir = directory / "plots"
  plot_dir.mkdir(exist_ok=True)
  for key in examples:
    source = sources[key]
    _, motion = load_motion(OUTPUT / "raw" / key, source["dataset"], source["fps"])
    result = classify_windows(motion, Cfg())
    time = np.arange(len(motion)) / 50
    speed = np.linalg.norm(result["arrays"][2][:, :2], axis=1)
    w, x, y, z = result["arrays"][1].T
    pelvis = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    fig, axes = plt.subplots(
      3, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )
    axes[0].plot(time, motion[:, 19], linewidth=1, label="Waist yaw relative to pelvis")
    for bound in (-0.35, -0.12, -0.08, 0.08, 0.12, 0.35):
      axes[0].axhline(bound, color="gray", alpha=0.3, linewidth=0.7)
    axes[0].set_ylabel("Waist yaw [rad]")
    axes[1].plot(time, speed, linewidth=1)
    axes[1].axhline(0.20, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Root speed [m/s]")
    axes[2].plot(time, pelvis - pelvis[0], linewidth=1)
    axes[2].set_ylabel("Pelvis yaw change [rad]")
    axes[2].set_xlabel("Source time [s]; original speed, no retiming")
    selected = [
      r
      for r in rows
      if r["source_id"] == key
      and str(r["mirrored"]).lower() == "false"
      and int(r["condition_id"]) != 0
    ]
    colors = {1: "orange", 2: "green", 3: "blue"}
    for row in selected:
      start, end = int(row["frame50_start"]), int(row["frame50_end_exclusive"])
      axes[0].axvspan(
        start / 50, end / 50, color=colors[int(row["phase_id"])], alpha=0.2
      )
    axes[0].set_title(
      key + "\nAccepted original windows: orange=early, green=entry, blue=hold"
    )
    name = (
      source["dataset"]
      + "_"
      + source["remote_path"].split("/")[-1].replace(".csv", "").replace(".npy", "")
      + ".png"
    )
    destination = plot_dir / name
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    summary["illustrations"].append(
      {"source_id": key, "path": str(destination), "sha256": digest(destination)}
    )
  save_json(report_path, summary)
  print(
    json.dumps(
      {
        "report": str(report_path),
        "by_dataset": summary["by_dataset"],
        "plots": len(summary["illustrations"]),
      }
    ),
    flush=True,
  )


if __name__ == "__main__":
  main()
