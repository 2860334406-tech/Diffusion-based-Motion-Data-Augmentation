"""Render a deterministic, stratified shoulder-IK replay set."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw

from smp.data.shoulder_ik import (
  FPS,
  _build_model,
  _qpos_ids,
  atomic_json,
  sha256,
)


@dataclass(frozen=True)
class Cfg:
  input_dir: str = "logs/reproduction/g1_shoulder_ik_v2/pilot"
  manifest: str = "manifest.json"
  output_dir: str = ""
  count: int = 60
  width: int = 480
  height: int = 360


def _evenly_spaced(items: list[dict], count: int) -> list[dict]:
  if len(items) < count:
    raise ValueError(f"Cannot select {count} from {len(items)}")
  indices = np.round(np.linspace(0, len(items) - 1, count)).astype(int)
  if len(set(indices.tolist())) != count:
    raise AssertionError("Stratified selection produced duplicate indices")
  return [items[index] for index in indices]


def _amplitude_quartile(item: dict) -> int:
  amplitude = float(item["target_yaw_amplitude_deg"])
  return min(3, int((amplitude - 15.0) / ((70.0 - 15.0) / 4.0)))


def _select_amplitude_quartiles(
  candidates: list[dict], per_quartile: int
) -> list[dict]:
  selected = []
  for quartile in range(4):
    group = sorted(
      (item for item in candidates if _amplitude_quartile(item) == quartile),
      key=lambda item: (item["target_yaw_amplitude_deg"], item["clip_id"]),
    )
    selected.extend(_evenly_spaced(group, per_quartile))
  return selected


def select_records(manifest: dict, count: int = 60) -> list[dict]:
  accepted = [
    record["accepted"]
    for _, record in sorted(manifest["slots"].items())
    if record["status"] == "accepted"
  ]
  stage = manifest["identity"]["stage"]
  selected: list[dict] = []
  if count == 10 and stage == "pilot":
    for condition, quota in ((0, 2), (1, 4), (2, 4)):
      candidates = sorted(
        (item for item in accepted if item["condition_id"] == condition),
        key=lambda item: (
          item.get("target_yaw_amplitude_deg", 0.0),
          item["family_id"],
          item["clip_id"],
        ),
      )
      selected.extend(_evenly_spaced(candidates, quota))
  elif count == 60 and stage == "pilot":
    stands = sorted(
      (item for item in accepted if item["condition_id"] == 0),
      key=lambda item: (item["family_id"], item["clip_id"]),
    )
    selected.extend(_evenly_spaced(stands, 12))
    for condition in (1, 2):
      candidates = [item for item in accepted if item["condition_id"] == condition]
      selected.extend(_select_amplitude_quartiles(candidates, 6))
  elif count == 60 and stage == "formal":
    for split in ("train", "validation", "test"):
      stands = sorted(
        (
          item
          for item in accepted
          if item["split"] == split and item["condition_id"] == 0
        ),
        key=lambda item: (item["family_id"], item["clip_id"]),
      )
      selected.extend(_evenly_spaced(stands, 4))
      for condition in (1, 2):
        candidates = [
          item
          for item in accepted
          if item["split"] == split and item["condition_id"] == condition
        ]
        selected.extend(_select_amplitude_quartiles(candidates, 2))
  else:
    raise ValueError("Replay quota must be pilot 10/60 or formal 60")

  expected = [12, 24, 24] if count == 60 else [2, 4, 4]
  actual = [
    sum(item["condition_id"] == condition for item in selected)
    for condition in range(3)
  ]
  if actual != expected or len(selected) != count:
    raise AssertionError(f"Review quota mismatch: {actual} != {expected}")
  return selected


def _load_qpos(root: Path, record: dict, model: mujoco.MjModel):
  metadata_path = root / record["metadata"]
  metadata = json.loads(metadata_path.read_text())
  path = metadata_path.with_suffix(".npz")
  if sha256(path) != metadata["execution_npz_sha256"]:
    raise ValueError(f"Replay identity mismatch: {path}")
  with np.load(path, allow_pickle=False) as data:
    root_pos = data["root_pos"]
    root_rot = data["root_rot"]
    joints = data["dof_pos"]
  qpos = np.zeros((len(root_pos), model.nq), dtype=np.float64)
  qpos[:, :3] = root_pos
  qpos[:, 3:7] = root_rot[:, [3, 0, 1, 2]]
  qpos[:, _qpos_ids(model)] = joints
  return metadata, qpos


def _camera(azimuth: float) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = (0.02, 0.0, 0.72)
  camera.distance = 2.35
  camera.azimuth = azimuth
  camera.elevation = -8.0
  return camera


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, min(image.width, 470), 23), fill=(0, 0, 0))
  draw.text((6, 5), label, fill=(255, 255, 255))
  return np.asarray(image)


def render_clip(
  renderer: mujoco.Renderer,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos: np.ndarray,
  label: str,
) -> list[np.ndarray]:
  cameras = (_camera(90.0), _camera(0.0))
  frames = []
  for pose in qpos:
    data.qpos[:] = pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    views = []
    for camera in cameras:
      renderer.update_scene(data, camera=camera)
      views.append(renderer.render().copy())
    frames.append(_annotate(np.concatenate(views, axis=1), label))
  return frames


def _contact_sheets(keyframes: list[tuple[str, np.ndarray]], output: Path):
  for page, start in enumerate(range(0, len(keyframes), 5), 1):
    rows = []
    for label, image in keyframes[start : start + 5]:
      annotated = _annotate(image, label)
      rows.append(annotated)
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet).save(output / f"contact_sheet_{page:02d}.png")


def main(cfg: Cfg):
  root = Path(cfg.input_dir)
  default_name = "visual_review" if cfg.count == 60 else f"replay_sample_{cfg.count}"
  output = Path(cfg.output_dir) if cfg.output_dir else root / default_name
  manifest_path = root / cfg.manifest
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("status") != "complete" or not manifest.get("gate_passed"):
    raise RuntimeError("Visual review requires a complete, gated pilot set")
  stage = manifest["identity"]["stage"]
  output.mkdir(parents=True, exist_ok=True)
  (output / "videos").mkdir(exist_ok=True)
  (output / "keyframes").mkdir(exist_ok=True)
  selected = select_records(manifest, cfg.count)
  condition_counts = {
    "stand": sum(item["condition_id"] == 0 for item in selected),
    "left_shoulder": sum(item["condition_id"] == 1 for item in selected),
    "right_shoulder": sum(item["condition_id"] == 2 for item in selected),
  }
  identity = {
    "schema_version": 1,
    "source_manifest_sha256": sha256(manifest_path),
    "stage": stage,
    "selection_seed": 42,
    "quota": condition_counts,
    "clip_count": cfg.count,
    "formal_visual_gate": cfg.count == 60,
    "formal_split_quota": (
      {"train": [4, 8, 8], "validation": [4, 8, 8], "test": [4, 8, 8]}
      if stage == "formal"
      else None
    ),
    "shoulder_amplitude_quartile_quota": (
      [6, 6, 6, 6] if stage == "pilot" else [6, 6, 6, 6]
    )
    if cfg.count == 60
    else None,
    "amplitude_quartile_edges_deg": [15.0, 28.75, 42.5, 56.25, 70.0],
    "views": ["front", "side"],
    "render_size_per_view": [cfg.width, cfg.height],
  }
  selection_path = output / "selection.json"
  if selection_path.exists():
    old = json.loads(selection_path.read_text())
    if old.get("identity") != identity:
      raise RuntimeError("Existing visual-review selection has a different identity")

  model = _build_model(soft_limits=False)
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
  rendered, keyframes = [], []
  try:
    for index, record in enumerate(selected):
      metadata, qpos = _load_qpos(root, record, model)
      spec = metadata["spec"]
      label = (
        f"{index + 1:02d}/{cfg.count:02d} {metadata['condition']} "
        f"target={metadata['metrics']['target_waist_amplitude_deg']:.1f}deg "
        f"actual={metadata['metrics']['actual_waist_amplitude_deg']:.1f}deg "
        f"entry={spec['entry_duration_s']:.2f}s "
        f"stance={spec['stance_width_m']:.2f} knee={spec['knee_flexion_rad']:.2f}"
      )
      frames = render_clip(renderer, model, data, qpos, label)
      video_path = output / "videos" / f"{index + 1:02d}_{record['clip_id']}.mp4"
      media.write_video(video_path, frames, fps=FPS, codec="h264", crf=23)
      chosen = [frames[0], frames[len(frames) // 2], frames[-1]]
      strip = np.concatenate(chosen, axis=1)
      keyframe_path = output / "keyframes" / f"{index + 1:02d}_{record['clip_id']}.png"
      Image.fromarray(strip).save(keyframe_path)
      keyframes.append((label, strip))
      rendered.append(
        {
          **record,
          "review_index": index + 1,
          "video": str(video_path.relative_to(root)),
          "video_sha256": sha256(video_path),
          "keyframes": str(keyframe_path.relative_to(root)),
          "keyframes_sha256": sha256(keyframe_path),
        }
      )
      print(f"[{index + 1}/{cfg.count}] {record['clip_id']}", flush=True)
  finally:
    renderer.close()
  _contact_sheets(keyframes, output)
  atomic_json(selection_path, {"identity": identity, "clips": rendered})
  if cfg.count == 60:
    review_template = root / "visual_review.json"
    if not review_template.exists():
      atomic_json(
        review_template,
        {
          "schema_version": 1,
          "selection_sha256": sha256(selection_path),
          "passed": None,
          "reviewed_clips": 0,
          "criteria": {
            "direction_error": 0,
            "stepping": 0,
            "whole_body_turn": 0,
            "visible_jitter": 0,
          },
          "note": "Complete only after all 60 rendered clips have been inspected.",
        },
      )
  else:
    atomic_json(
      output / "preview.json",
      {
        "schema_version": 1,
        "selection_sha256": sha256(selection_path),
        "rendered_clips": cfg.count,
        "condition_counts": condition_counts,
        "formal_visual_gate": False,
        "note": "Exploratory replay only; this does not satisfy the 60-clip gate.",
      },
    )
  print(json.dumps({"rendered": len(rendered), "output": str(output)}))


if __name__ == "__main__":
  main(tyro.cli(Cfg))
