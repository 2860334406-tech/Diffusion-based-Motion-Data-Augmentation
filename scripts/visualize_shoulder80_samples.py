"""Create reproducible plots and FK videos for audited Shoulder80 windows.

The selection is deliberately stratified rather than a quality-selected showcase:
train/test x left/right shoulder x early/strong/hold.  Early-entry samples also
include their recorded mature link so the short 10-frame onset can be judged in
context.  This script visualizes stored data only; it does not sample a Prior.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import mediapy
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.generate_viz import _setup_g1_sim, _write_pose_to_robot

from smp.pretrain.conditioning import sha256
from smp.pretrain.shoulder80 import CONDITION_NAMES, PHASE_NAMES
from smp.pretrain.shoulder80_dataset import DATA_SHA, Shoulder80Dataset
from smp.sampling.feature_to_state import window_to_pelvis_trajectory


@dataclass(frozen=True)
class Cfg:
  data_dir: str = "logs/reproduction/external_shoulder_data_review_v1/review"
  output_dir: str = (
    "logs/reproduction/conditional_shoulder80_prior_v1_expanded/"
    "data_visualization_seed42_v1"
  )
  seed: int = 42
  splits: tuple[str, ...] = ("train", "test")
  device: str = "cuda:0"
  width: int = 960
  height: int = 540
  fps: int = 50
  slowdown: int = 10
  record_video: bool = True


DATASET_PREFERENCE = {
  "train": {1: "g1_moves", 2: "lafan", 3: "amass_g1"},
  "test": {1: "amass_g1", 2: "lafan", 3: "amass_g1"},
}


def _atomic_json(path: Path, payload: dict) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  os.replace(temporary, path)


def _rank(row: dict, seed: int) -> str:
  identity = "|".join(
    (
      str(seed),
      row["split"],
      row["condition"],
      row["phase"],
      row["source_id"],
      str(row["frame50_start"]),
      row["mirrored"],
    )
  )
  return hashlib.sha256(identity.encode()).hexdigest()


def _select(dataset: Shoulder80Dataset, seed: int, used: set[tuple]) -> list[int]:
  selected = []
  for condition in (1, 2):
    for phase in (1, 2, 3):
      candidates = [
        i
        for i, row in enumerate(dataset.rows)
        if int(row["condition_id"]) == condition
        and int(row["phase_id"]) == phase
      ]
      preferred = DATASET_PREFERENCE.get(dataset.split, {}).get(phase)
      candidates.sort(
        key=lambda i: (
          dataset.rows[i]["dataset"] != preferred,
          _rank(dataset.rows[i], seed),
        )
      )
      chosen = None
      for index in candidates:
        row = dataset.rows[index]
        # Mirrors of the same source interval are one underlying action.
        identity = (row["source_id"], int(row["frame50_start"]))
        if identity not in used:
          chosen = index
          used.add(identity)
          break
      if chosen is None:
        raise RuntimeError(
          f"No independent sample for {dataset.split}/{condition}/{phase}"
        )
      selected.append(chosen)
  return selected


def _linked_index(dataset: Shoulder80Dataset, index: int) -> int | None:
  row = dataset.rows[index]
  link = int(row["early_link_frame50"])
  if int(row["phase_id"]) != 1:
    return None
  if link < 0:
    raise ValueError("Early-entry row has no mature link")
  candidates = [
    i
    for i, other in enumerate(dataset.rows)
    if other["source_id"] == row["source_id"]
    and other["mirrored"] == row["mirrored"]
    and int(other["condition_id"]) == int(row["condition_id"])
    and int(other["frame50_start"]) == link
    and int(other["phase_id"]) in (2, 3)
  ]
  if len(candidates) != 1:
    raise ValueError(f"Expected one mature link, found {len(candidates)}")
  return candidates[0]


def _series(window: np.ndarray, condition: int) -> dict[str, np.ndarray]:
  sign = 1.0 if condition == 1 else -1.0
  waist = window[:, 21]
  pelvis = np.unwrap(np.arctan2(window[:, 4], window[:, 3]))
  root_xy = window[:, :2] - window[:1, :2]
  return {
    "signed_waist": sign * waist,
    "waist": waist,
    "pelvis_delta": pelvis - pelvis[0],
    "speed": np.linalg.norm(window[:, 53:55], axis=1),
    "root_displacement": np.linalg.norm(root_xy, axis=1),
  }


def _metrics(window: np.ndarray, condition: int, phase: int) -> dict:
  values = _series(window, condition)
  signed = values["signed_waist"]
  delta = float(signed[-1] - signed[0])
  if phase == 1:
    phase_pass = signed[0] >= 0 and signed[-1] < 0.08 and delta >= 0.01
  elif phase == 2:
    phase_pass = signed[0] >= 0 and signed[-1] >= 0.08 and delta >= 0.02
  elif phase == 3:
    phase_pass = (
      signed[-1] >= 0.12 and signed.min() >= 0.10 and abs(delta) <= 0.01
    )
  else:
    raise ValueError(f"Unsupported shoulder phase {phase}")
  pelvis_delta = float(values["pelvis_delta"][-1])
  speed_max = float(values["speed"].max())
  return {
    "signed_waist_start_rad": float(signed[0]),
    "signed_waist_end_rad": float(signed[-1]),
    "signed_waist_delta_rad": delta,
    "signed_waist_min_rad": float(signed.min()),
    "signed_waist_max_rad": float(signed.max()),
    "pelvis_yaw_delta_rad": pelvis_delta,
    "root_planar_speed_max_m_s": speed_max,
    "root_displacement_end_m": float(values["root_displacement"][-1]),
    "phase_rule_passed": bool(phase_pass),
    "pelvis_rule_passed": abs(pelvis_delta) <= 0.05,
    "speed_rule_passed": speed_max <= 0.20,
  }


def _plot_sample(sample: dict, output: Path) -> None:
  fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
  colors = ("tab:blue", "tab:orange")
  for segment_index, segment in enumerate(sample["segments"]):
    window = segment["window"]
    values = _series(window, sample["condition_id"])
    offset = (segment["frame50_start"] - sample["frame50_start"]) / 50.0
    time = offset + np.arange(len(window)) / 50.0
    label = segment["role"] + ":" + segment["phase"]
    color = colors[segment_index]
    axes[0].plot(time, values["signed_waist"], marker="o", color=color, label=label)
    axes[1].plot(time, values["pelvis_delta"], marker="o", color=color)
    axes[2].plot(time, values["speed"], marker="o", color=color)
    axes[3].plot(time, values["root_displacement"], marker="o", color=color)
  for bound in (0.08, 0.12, 0.35):
    axes[0].axhline(bound, color="gray", linestyle="--", linewidth=0.8)
  axes[1].axhline(0.05, color="red", linestyle="--", linewidth=0.8)
  axes[1].axhline(-0.05, color="red", linestyle="--", linewidth=0.8)
  axes[2].axhline(0.20, color="red", linestyle="--", linewidth=0.8)
  axes[0].set_ylabel("Signed waist yaw [rad]")
  axes[1].set_ylabel("Pelvis yaw delta [rad]")
  axes[2].set_ylabel("Root speed [m/s]")
  axes[3].set_ylabel("Root XY displacement [m]")
  axes[3].set_xlabel("Time from selected-window start [s]")
  axes[0].legend(loc="best")
  axes[0].set_title(
    f"{sample['sample_id']} | {sample['split']} | {sample['dataset']} | "
    f"{sample['condition']} / {sample['phase']}\n{sample['source_id']} "
    f"frames {sample['frame50_start']}:{sample['frame50_end_exclusive']} "
    f"mirror={sample['mirrored']}"
  )
  fig.savefig(output, dpi=150)
  plt.close(fig)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _annotate(frame: np.ndarray, lines: list[str], height: int = 86) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0))
  font = _font(21)
  for i, line in enumerate(lines):
    draw.text((12, 7 + 25 * i), line, fill=(255, 255, 255), font=font)
  return np.asarray(image)


def _render(cfg: Cfg, samples: list[dict], output: Path) -> dict[str, list[np.ndarray]]:
  from mjlab.viewer import OffscreenRenderer, ViewerConfig

  sim, scene = _setup_g1_sim(cfg.device)
  robot = scene["robot"]
  anchor_pos = robot.data.default_root_state[0, :3].detach().cpu()
  anchor_quat = robot.data.default_root_state[0, 3:7].detach().cpu()
  viewer = ViewerConfig(
    height=cfg.height,
    width=cfg.width,
    distance=2.8,
    elevation=-7.0,
    azimuth=35.0,
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
  )
  renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer, scene=scene)
  renderer.initialize()
  videos = output / "videos"
  videos.mkdir(exist_ok=True)
  snapshots: dict[str, list[np.ndarray]] = {}
  combined_path = output / "selected_samples_fk_10x_slow.mp4"
  try:
    with mediapy.VideoWriter(
      combined_path,
      shape=(cfg.height, cfg.width),
      fps=cfg.fps,
      crf=18,
    ) as combined:
      for sample in samples:
        rendered_segments = []
        for segment in sample["segments"]:
          window = torch.from_numpy(segment["window"]).float()
          pos, quat, joints = window_to_pelvis_trajectory(
            window, anchor_pos, anchor_quat
          )
          rendered = []
          for frame_index in range(len(window)):
            _write_pose_to_robot(
              robot,
              pos[frame_index].numpy(),
              quat[frame_index].numpy(),
              joints[frame_index].numpy(),
              cfg.device,
            )
            sim.forward()
            scene.update(sim.mj_model.opt.timestep)
            renderer.update(sim.data)
            pixels = renderer.render().copy()
            rendered.append(
              _annotate(
                pixels,
                [
                  f"{sample['sample_id']}  {sample['split']}  {sample['condition']} / {segment['phase']}",
                  f"{sample['dataset']}  frame={segment['frame50_start'] + frame_index}  mirror={sample['mirrored']}",
                  f"signed waist end={segment['metrics']['signed_waist_end_rad']:.3f} rad  pelvis delta={segment['metrics']['pelvis_yaw_delta_rad']:.3f} rad",
                ],
              )
            )
          rendered_segments.append(rendered)
        snapshots[sample["sample_id"]] = [
          rendered_segments[0][0],
          rendered_segments[0][len(rendered_segments[0]) // 2],
          rendered_segments[0][-1],
          rendered_segments[-1][-1],
        ]
        individual_path = videos / f"{sample['sample_id']}_fk_10x_slow.mp4"
        with mediapy.VideoWriter(
          individual_path,
          shape=(cfg.height, cfg.width),
          fps=cfg.fps,
          crf=18,
        ) as individual:
          first = rendered_segments[0][0]
          for _ in range(15):
            individual.add_image(first)
            combined.add_image(first)
          for segment_index, frames in enumerate(rendered_segments):
            if segment_index:
              gap = _annotate(
                np.zeros_like(first),
                [sample["sample_id"], "Jump to recorded mature-link window"],
              )
              for _ in range(15):
                individual.add_image(gap)
                combined.add_image(gap)
            for frame in frames:
              for _ in range(cfg.slowdown):
                individual.add_image(frame)
                combined.add_image(frame)
          last = rendered_segments[-1][-1]
          for _ in range(15):
            individual.add_image(last)
            combined.add_image(last)
        separator = _annotate(
          np.zeros_like(first),
          ["Next sample", sample["sample_id"]],
        )
        for _ in range(20):
          combined.add_image(separator)
        sample["video"] = str(individual_path)
        sample["video_sha256"] = sha256(individual_path)
  finally:
    renderer.close()
  return snapshots


def _contact_sheet(samples: list[dict], snapshots: dict, output: Path) -> None:
  fig, axes = plt.subplots(
    len(samples), 4, figsize=(12, 2.45 * len(samples)), constrained_layout=True
  )
  columns = ("start", "middle", "end", "mature end / end")
  for row_index, sample in enumerate(samples):
    for column_index, frame in enumerate(snapshots[sample["sample_id"]]):
      axes[row_index, column_index].imshow(frame)
      axes[row_index, column_index].axis("off")
      if row_index == 0:
        axes[row_index, column_index].set_title(columns[column_index])
    axes[row_index, 0].set_ylabel(sample["sample_id"], rotation=90)
  fig.savefig(output, dpi=120)
  plt.close(fig)


def _write_csv(samples: list[dict], path: Path) -> None:
  fields = [
    "sample_id",
    "split",
    "dataset",
    "source_id",
    "source_group",
    "mirrored",
    "frame50_start",
    "frame50_end_exclusive",
    "condition",
    "phase",
    "linked_frame50_start",
    "signed_waist_start_rad",
    "signed_waist_end_rad",
    "signed_waist_delta_rad",
    "pelvis_yaw_delta_rad",
    "root_planar_speed_max_m_s",
    "phase_rule_passed",
    "pelvis_rule_passed",
    "speed_rule_passed",
    "plot",
    "video",
  ]
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for sample in samples:
      metrics = sample["segments"][0]["metrics"]
      writer.writerow(
        {
          **{field: sample.get(field, "") for field in fields},
          **{field: metrics[field] for field in fields if field in metrics},
          "linked_frame50_start": (
            sample["segments"][1]["frame50_start"]
            if len(sample["segments"]) > 1
            else ""
          ),
        }
      )


def main(cfg: Cfg) -> None:
  if cfg.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("This audit expects physical GPU0 exposed as cuda:0")
  if cfg.slowdown < 1 or cfg.fps <= 0:
    raise ValueError("Invalid playback configuration")
  output = Path(cfg.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  (output / "plots").mkdir(exist_ok=True)
  datasets = {
    split: Shoulder80Dataset(cfg.data_dir, split, expected_sha=DATA_SHA)
    for split in cfg.splits
  }
  used: set[tuple] = set()
  samples = []
  ordinal = 0
  for split in cfg.splits:
    dataset = datasets[split]
    for index in _select(dataset, cfg.seed, used):
      ordinal += 1
      row = dataset.rows[index]
      condition, phase = int(row["condition_id"]), int(row["phase_id"])
      window = dataset.raw[index].numpy()
      sample_id = (
        f"{ordinal:02d}_{split}_{CONDITION_NAMES[condition]}_"
        f"{PHASE_NAMES[phase]}_{row['dataset']}"
      )
      primary = {
        "role": "selected",
        "phase_id": phase,
        "phase": PHASE_NAMES[phase],
        "frame50_start": int(row["frame50_start"]),
        "window": window,
        "metrics": _metrics(window, condition, phase),
      }
      segments = [primary]
      linked = _linked_index(dataset, index)
      if linked is not None:
        linked_row = dataset.rows[linked]
        linked_phase = int(linked_row["phase_id"])
        linked_window = dataset.raw[linked].numpy()
        segments.append(
          {
            "role": "mature_link",
            "phase_id": linked_phase,
            "phase": PHASE_NAMES[linked_phase],
            "frame50_start": int(linked_row["frame50_start"]),
            "window": linked_window,
            "metrics": _metrics(linked_window, condition, linked_phase),
          }
        )
      sample = {
        "sample_id": sample_id,
        "dataset_index": index,
        "split": split,
        "dataset": row["dataset"],
        "source": row["source"],
        "source_id": row["source_id"],
        "source_group": row["source_group"],
        "source_sha256": row["source_sha256"],
        "shard": row["shard"],
        "shard_index": int(row["index"]),
        "mirrored": row["mirrored"].lower() == "true",
        "frame50_start": int(row["frame50_start"]),
        "frame50_end_exclusive": int(row["frame50_end_exclusive"]),
        "condition_id": condition,
        "condition": CONDITION_NAMES[condition],
        "phase_id": phase,
        "phase": PHASE_NAMES[phase],
        "segments": segments,
      }
      plot = output / "plots" / f"{sample_id}.png"
      _plot_sample(sample, plot)
      sample["plot"] = str(plot)
      sample["plot_sha256"] = sha256(plot)
      samples.append(sample)
  snapshots = {}
  if cfg.record_video:
    snapshots = _render(cfg, samples, output)
    _contact_sheet(samples, snapshots, output / "selected_samples_contact_sheet.png")
  for sample in samples:
    for segment in sample["segments"]:
      segment.pop("window")
  _write_csv(samples, output / "selection.csv")
  manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Stratified visual audit of stored Shoulder80 data; not Prior samples",
    "config": asdict(cfg),
    "data_manifest": str(Path(cfg.data_dir) / "manifest.json"),
    "data_manifest_sha256": sha256(Path(cfg.data_dir) / "manifest.json"),
    "expected_data_sha256": DATA_SHA,
    "selection_policy": {
      "strata": "split x shoulder condition x phase",
      "ranking": "SHA-256(seed + immutable lineage)",
      "mirror_deduplication": "same source_id/frame50_start cannot be selected twice",
      "dataset_preference": DATASET_PREFERENCE,
      "quality_selection": False,
    },
    "samples": samples,
    "all_primary_filter_rules_passed": all(
      all(
        sample["segments"][0]["metrics"][key]
        for key in ("phase_rule_passed", "pelvis_rule_passed", "speed_rule_passed")
      )
      for sample in samples
    ),
    "artifacts": {
      "selection_csv": str(output / "selection.csv"),
      "contact_sheet": (
        str(output / "selected_samples_contact_sheet.png")
        if cfg.record_video
        else None
      ),
      "combined_video": (
        str(output / "selected_samples_fk_10x_slow.mp4")
        if cfg.record_video
        else None
      ),
    },
  }
  if cfg.record_video:
    for key in ("contact_sheet", "combined_video"):
      manifest["artifacts"][key + "_sha256"] = sha256(
        manifest["artifacts"][key]
      )
  _atomic_json(output / "manifest.json", manifest)
  readme = f"""# Shoulder80 data visualization (seed {cfg.seed})

This directory visualizes **stored training/test windows**, not diffusion samples
and not PPO rollouts. Selection is deterministic and stratified; see
`manifest.json` for immutable lineage and exact metrics.

- `selection.csv`: compact sample index and threshold checks.
- `plots/`: signed waist yaw, pelvis yaw, root speed, and displacement.
- `videos/`: individual FK previews, slowed {cfg.slowdown}x.
- `selected_samples_fk_10x_slow.mp4`: samples in manifest order.
- `selected_samples_contact_sheet.png`: start/middle/end pose comparison.

For left shoulder, positive waist yaw is treated as the desired direction; for
right shoulder, negative raw yaw is sign-flipped so a correct response is also
positive in the plots. Early-entry videos include the separately stored mature
window linked within one second; the gap is explicit and is not interpolated.
"""
  (output / "README.md").write_text(readme)
  print(
    json.dumps(
      {
        "output": str(output),
        "samples": len(samples),
        "all_primary_filter_rules_passed": manifest[
          "all_primary_filter_rules_passed"
        ],
        "combined_video": manifest["artifacts"]["combined_video"],
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main(tyro.cli(Cfg))
