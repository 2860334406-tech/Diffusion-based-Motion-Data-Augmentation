"""Independently verify EXP-034 lineage and summarize all accepted conditions."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import tyro

from smp.data.personal_dataset import publication_gate
from smp.data.personal_retarget import CONDITIONS, inventory, verify_attempt
from smp.data.shoulder_ik import atomic_json, sha256


def main(input_dir: str = "logs/reproduction/g1_personal_retarget_v1"):
  root = Path(input_dir)
  manifest = json.loads((root / "manifest.json").read_text())
  if (
    inventory(Path(manifest["identity"]["source_dir"]))
    != manifest["identity"]["sources"]
  ):
    raise ValueError("Source inventory changed")
  frozen = root / "audit" / "frozen_code"
  frozen.mkdir(parents=True, exist_ok=True)
  for name, expected in manifest["identity"]["code"].items():
    path = Path(name)
    if sha256(path) != expected:
      raise ValueError(f"Frozen implementation changed: {name}")
    copy = frozen / path
    copy.parent.mkdir(parents=True, exist_ok=True)
    if copy.exists() and sha256(copy) != expected:
      raise ValueError(f"Frozen code snapshot corrupted: {name}")
    if not copy.exists():
      shutil.copy2(path, copy)
  accepted, rejected, verified = [], [], 0
  for clip_id, record in sorted(manifest["clips"].items()):
    for attempt in record["attempts"]:
      metadata = verify_attempt(root, attempt)
      verified += 1 + len(metadata["files"])
      if attempt["passed"]:
        with np.load(
          root / metadata["files"]["execution.npz"]["path"], allow_pickle=False
        ) as z:
          csv_values = np.loadtxt(
            root / metadata["files"]["execution.csv"]["path"], delimiter=","
          )
          combined = np.concatenate(
            (z["root_pos"], z["root_rot"], z["dof_pos"]), axis=1
          )
          if not np.isfinite(combined).all() or not np.allclose(
            csv_values, combined, rtol=1e-7, atol=1e-7
          ):
            raise ValueError("Accepted execution/CSV contract mismatch")
        accepted.append(
          {
            "clip_id": clip_id,
            "subject_id": metadata["subject_id"],
            "split": metadata["split"],
            "condition_id": metadata["condition_id"],
            **metadata["metrics"],
          }
        )
      else:
        rejected.append(
          {
            "clip_id": clip_id,
            "attempt": attempt["attempt"],
            "metrics": metadata["metrics"],
          }
        )
  groups = {}
  for condition, name in enumerate(CONDITIONS):
    group = [x for x in accepted if x["condition_id"] == condition]
    metrics = {}
    keys = set().union(*(x.keys() for x in group)) if group else set()
    for key in sorted(keys):
      values = [
        x[key]
        for x in group
        if isinstance(x.get(key), (int, float)) and not isinstance(x[key], bool)
      ]
      if values:
        metrics[key] = {
          "min": float(np.min(values)),
          "p50": float(np.median(values)),
          "max": float(np.max(values)),
        }
    groups[name] = {"clips": len(group), "metrics": metrics}
  dataset_path = root / "dataset" / "manifest.json"
  dataset = json.loads(dataset_path.read_text()) if dataset_path.exists() else None
  if dataset:
    if manifest.get("dataset", {}).get("sha256") != sha256(dataset_path):
      raise ValueError("Published dataset manifest identity mismatch")
    review_path = root / "replay" / "review.json"
    review = json.loads(review_path.read_text())
    snapshot_path = root / "dataset" / "physics_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    if (
      sha256(snapshot_path) != dataset["physics_snapshot_sha256"]
      or snapshot["review_sha256"] != sha256(review_path)
      or not publication_gate(manifest["clips"], review)["passed"]
    ):
      raise ValueError("Published physics/review identity mismatch")
    replay_index = json.loads((root / "replay" / "index.json").read_text())
    for clip_id, item in replay_index["clips"].items():
      if review["clips"][clip_id]["video_sha256"] != item["video_sha256"]:
        raise ValueError("Reviewed video hash differs")
      for key in ("video", "keyframes", "curves"):
        if sha256(root / item[key]) != item[key + "_sha256"]:
          raise ValueError(f"Replay artifact changed: {clip_id}/{key}")
    for key, expected_key in (
      ("clip_table", "clip_table_sha256"),
      ("windows_index", "windows_index_sha256"),
      ("diagnostic_norm", "diagnostic_norm_sha256"),
    ):
      if sha256(root / "dataset" / dataset[key]) != dataset[expected_key]:
        raise ValueError(f"Dataset auxiliary identity mismatch: {key}")
    for record in dataset["shards"]:
      path = dataset_path.parent / record["path"]
      if sha256(path) != record["sha256"]:
        raise ValueError("Published shard checksum mismatch")
      with np.load(path, allow_pickle=False) as z:
        count = len(z["condition_id"])
        if z["windows"].shape != (count, 10, 59) or not np.isfinite(z["windows"]).all():
          raise ValueError("Published feature shape/nonfinite error")
        for index in np.unique(z["clip_index"]):
          if not np.isclose(
            z["clip_balanced_weight"][z["clip_index"] == index].sum(), 1
          ):
            raise ValueError("Clip-balanced weights do not sum to one")
  assessment = {
    "experiment": "EXP-034",
    "identity_sha256": manifest["identity_sha256"],
    "processed": len(manifest["clips"]),
    "accepted": len(accepted),
    "rejected_clips": sum(
      r["status"] == "rejected" for r in manifest["clips"].values()
    ),
    "rejected_attempts": len(rejected),
    "verified_attempt_artifacts": verified,
    "groups": groups,
    "rejections": rejected,
    "dataset": dataset,
    "prior_or_ppo_started": False,
  }
  atomic_json(root / "assessment.json", assessment)
  fields = [
    "clip_id",
    "subject_id",
    "split",
    "condition_id",
    "actual_waist_delta_rad",
    "actual_pelvis_delta_rad",
    "actual_torso_delta_rad",
    "shoulder_retreat_final_world_m",
    "shoulder_retreat_final_compensated_m",
    "hold_shoulder_deviation_max_m",
    "root_displacement_max_m",
    "foot_slip_max_m",
    "joint_tracking_rmse_rad",
    "waist_tracking_rmse_rad",
    "effort_p99_ratio",
    "effort_saturation_fraction",
    "hold_pelvis_yaw_speed_p95_rad_s",
  ]
  with (root / "accepted_metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(accepted)
  lines = [
    "# EXP-034 个人动作二次重定向结果",
    "",
    f"状态：{manifest['status']}。处理27条，物理合格{len(accepted)}条，最终拒绝{assessment['rejected_clips']}条。",
    "",
    "| 条件 | 合格条数 | 实际腰部增量绝对值范围 | 实际骨盆增量绝对值范围 | 肩部世界后退范围 |",
    "|---|---:|---:|---:|---:|",
  ]
  for name, group in groups.items():
    values = [x for x in accepted if CONDITIONS[x["condition_id"]] == name]

    def degree_range(key, selected=values):
      angles = np.degrees(np.abs([x[key] for x in selected]))
      return f"{angles.min():.2f}–{angles.max():.2f}°"

    retreat = [
      x["shoulder_retreat_final_world_m"]
      for x in values
      if "shoulder_retreat_final_world_m" in x
    ]
    shoulder = (
      f"{min(retreat) * 100:.2f}–{max(retreat) * 100:.2f} cm" if retreat else "—"
    )
    lines.append(
      f"| {name} | {group['clips']} | {degree_range('actual_waist_delta_rad')} | {degree_range('actual_pelvis_delta_rad')} | {shoulder} |"
    )
  lines.extend(
    [
      "",
      "验收依据：自由root、原执行器、完整碰撞、双脚固定；肩部同时检查初始heading下世界后退与扣除pelvis平移的后退。腰和骨盆角度不设额外幅度门槛。",
      "",
      f"最大足部水平漂移：{max(x['foot_slip_max_m'] for x in accepted) * 1000:.4f} mm；最大root平移：{max(x['root_displacement_max_m'] for x in accepted) * 100:.3f} cm；最大力矩p99：{max(x['effort_p99_ratio'] for x in accepted) * 100:.3f}%上限；饱和帧比例最大值：{max(x['effort_saturation_fraction'] for x in accepted):.6f}。",
      "",
      "## 拒绝记录",
      "",
    ]
  )
  for record in manifest["clips"].values():
    if record["status"] == "rejected":
      for candidate in record["attempts"]:
        metadata = verify_attempt(root, candidate)
        m = metadata["metrics"]
        tilt = math.degrees(m.get("root_tilt_abs_max_rad", 0))
        lines.append(
          f"- {record['source_clip_id']} attempt {candidate['attempt']}：{', '.join(candidate['failed_checks'])}；最大倾角 {tilt:.4f}°。"
        )
  lines.extend(
    [
      "",
      "## 数据与回放",
      "",
      "- [逐条指标](accepted_metrics.csv)、[完整JSON报告](assessment.json)、[生产manifest](manifest.json)。",
      "- [回放索引](replay/index.json)、[具名复核记录](replay/review.json)；视频为原GMR正/侧面与实际执行正/侧面四视图，50fps，完成后全视频解码核验。",
      "- 末段采用已观察保持姿态的稳定软目标；原始数据和时长保留，实际执行数组未补帧或复制状态。",
      "- 首轮未稳定末段的失败smoke保留在独立历史入口，不混入本数据集。",
      "",
    ]
  )
  if dataset:
    lines.extend(
      ["| split | 独立clip | 窗口数 | stand / left / right |", "|---|---:|---:|---|"]
    )
    for shard in dataset["shards"]:
      count = sum(x["split"] == shard["split"] for x in accepted)
      lines.append(
        f"| {shard['split']} | {count} | {shard['windows']} | {' / '.join(map(str, shard['condition_counts']))} |"
      )
    lines.extend(
      [
        "",
        "[可训练原始特征manifest](dataset/manifest.json)。窗口为未归一化(N,10,59)，无阶段标签；P01/P02/P03分别固定train/validation/test。归一化统计仅作诊断，后续混合Prior须统一重新统计、校准并验收在线Score。",
      ]
    )
  lines.extend(
    [
      "",
      "本轮未启动Prior或PPO；本数据集允许骨盆协同退让，不能将其统计为纯腰部相对旋转。",
      "",
    ]
  )
  (root / "ASSESSMENT.md").write_text("\n".join(lines))
  if dataset:
    tools = root / "audit" / "review_tools"
    for name in (
      "scripts/render_g1_personal_retarget.py",
      "scripts/audit_g1_personal_retarget.py",
      "tests/test_personal_retarget.py",
      "docs/G1_PERSONAL_CONSTRAINED_RETARGET_V1.md",
    ):
      destination = tools / name
      destination.parent.mkdir(parents=True, exist_ok=True)
      if destination.exists() and sha256(destination) != sha256(Path(name)):
        raise ValueError(f"Existing review-tool snapshot differs: {name}")
      if not destination.exists():
        shutil.copy2(name, destination)
    if any((root / "replay" / "videos").glob("*.tmp.mp4")):
      raise RuntimeError("Partial replay files remain; do not seal this release")
    files = sorted(
      path
      for path in root.rglob("*")
      if path.is_file()
      and not {"cache", "__pycache__"}.intersection(path.relative_to(root).parts)
      and path.name not in {"checksums.sha256", "release_manifest.json", "driver.lock"}
    )
    checksums = [(sha256(path), path.relative_to(root).as_posix()) for path in files]
    (root / "checksums.sha256").write_text(
      "".join(f"{value}  {name}\n" for value, name in checksums)
    )
    if any(sha256(root / name) != expected for expected, name in checksums):
      raise ValueError("Release checksum verification failed")
    atomic_json(
      root / "release_manifest.json",
      {
        "experiment": "EXP-034",
        "status": "complete",
        "identity_sha256": manifest["identity_sha256"],
        "accepted_clips": len(accepted),
        "rejected_clips": assessment["rejected_clips"],
        "windows": sum(x["windows"] for x in dataset["shards"]),
        "videos": len(replay_index["clips"]),
        "dataset_manifest_sha256": sha256(dataset_path),
        "review_sha256": sha256(review_path),
        "verified_files": len(checksums),
        "checksums_sha256": sha256(root / "checksums.sha256"),
        "source_data_overwritten": False,
        "prior_or_ppo_started": False,
      },
    )
  print(
    json.dumps(
      {
        "accepted": len(accepted),
        "verified_attempt_artifacts": verified,
        "groups": {k: v["clips"] for k, v in groups.items()},
        "dataset_published": dataset is not None,
      }
    )
  )


if __name__ == "__main__":
  tyro.cli(main)
