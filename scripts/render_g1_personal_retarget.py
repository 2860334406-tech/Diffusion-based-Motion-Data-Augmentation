"""CPU replays comparing personal GMR sources and measured G1 executions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from smp.data.personal_retarget import PersonalRetargeter, verify_attempt
from smp.data.shoulder_ik import (
  FPS,
  _build_model,
  _quat_yaw_wxyz,
  atomic_json,
  contract_to_qpos,
  sha256,
)


@dataclass(frozen=True)
class Cfg:
  input_dir: str = "logs/reproduction/g1_personal_retarget_v1"
  workers: int = 3
  width: int = 256
  height: int = 224
  action: str = "render"
  clip_ids: tuple[str, ...] = ()
  reviewer: str = ""
  decision: str = ""
  review_note: str = ""


def load_qpos(path: Path, model):
  with np.load(path, allow_pickle=False) as z:
    return contract_to_qpos(z["root_pos"], z["root_rot"], z["dof_pos"], model)


def align_source(qpos):
  result = qpos.copy()
  initial_yaw = _quat_yaw_wxyz(result[0, 3:7])
  align = Rotation.from_rotvec([0, 0, -initial_yaw])
  origin = result[0, :3].copy()
  origin[2] = 0
  result[:, :3] = align.apply(result[:, :3] - origin)
  rot = (align * Rotation.from_quat(result[:, [4, 5, 6, 3]])).as_quat()
  result[:, 3:7] = rot[:, [3, 0, 1, 2]]
  return result


def label_frame(image, label):
  image = Image.fromarray(image)
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, image.width, 21), fill="black")
  draw.text((4, 4), label, fill="white")
  return np.asarray(image)


def plot_curves(output, clip_id, model, raw, actual, metadata):
  helper = PersonalRetargeter()
  _, _, raw_torso = helper.landmarks(raw)
  _, _, actual_torso = helper.landmarks(actual)
  raw_pelvis = np.unwrap([_quat_yaw_wxyz(x[3:7]) for x in raw])
  actual_pelvis = np.unwrap([_quat_yaw_wxyz(x[3:7]) for x in actual])
  waist = int(model.jnt_qposadr[model.joint("waist_yaw_joint").id])
  time = np.arange(len(actual)) / FPS
  fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
  for axis, source, measured, title in zip(
    axes,
    [raw[:, waist], raw_pelvis, raw_torso],
    [actual[:, waist], actual_pelvis, actual_torso],
    ["waist relative to pelvis", "pelvis yaw", "torso world yaw"],
    strict=True,
  ):
    axis.plot(
      time,
      np.degrees(source - np.median(source[:FPS])),
      label="GMR reference",
      alpha=0.7,
    )
    axis.plot(time, np.degrees(measured - np.median(measured[:FPS])), label="executed")
    axis.set_ylabel("degrees")
    axis.set_title(title, fontsize=9)
    axis.axvline(
      metadata["turn_start_frame"] / FPS, color="gray", linestyle="--", linewidth=0.7
    )
    axis.grid(alpha=0.25)
  axes[0].legend(fontsize=8)
  axes[-1].set_xlabel("time (s)")
  fig.suptitle(clip_id)
  fig.tight_layout()
  path = output / "curves" / f"{clip_id}.png"
  fig.savefig(path, dpi=120)
  plt.close(fig)
  return str(path)


def render_one(root_string: str, record: dict, cfg: Cfg):
  root = Path(root_string)
  model = _build_model(soft_limits=False)
  metadata = verify_attempt(root, record["accepted"])
  clip = metadata["source_clip_id"]
  output = root / "replay"
  raw = align_source(load_qpos(root / metadata["files"]["source50.npz"]["path"], model))
  actual = load_qpos(root / metadata["files"]["execution.npz"]["path"], model)
  if len(raw) != len(actual):
    raise ValueError("Replay source/execution length mismatch")
  renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
  data = mujoco.MjData(model)
  cameras = []
  for azimuth in (0, 90):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.02, 0, 0.72)
    camera.distance, camera.elevation, camera.azimuth = 2.45, -8, azimuth
    cameras.append(camera)
  video = output / "videos" / f"{clip}.mp4"
  temporary = video.with_name(video.stem + ".tmp.mp4")
  writer = imageio_ffmpeg.write_frames(
    str(temporary),
    (cfg.width * 4, cfg.height),
    fps=FPS,
    codec="libx264",
    pix_fmt_in="rgb24",
    output_params=["-crf", "23", "-preset", "fast"],
    ffmpeg_log_level="error",
  )
  writer.send(None)
  sample_frames = set(np.linspace(0, len(actual) - 1, 12).round().astype(int).tolist())
  sample_frames.update([metadata["turn_start_frame"], len(actual) - FPS])
  keyframes = []
  try:
    for frame in range(len(actual)):
      views = []
      for poses, title in ((raw, "GMR"), (actual, "EXEC")):
        data.qpos[:] = poses[frame]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for camera, view in zip(cameras, ("front", "side"), strict=True):
          renderer.update_scene(data, camera=camera)
          views.append(
            label_frame(renderer.render().copy(), f"{title} {view} {frame / FPS:.2f}s")
          )
      image = np.concatenate(views, axis=1)
      writer.send(image)
      if frame in sample_frames:
        keyframes.append((frame, Image.fromarray(image)))
  finally:
    writer.close()
    renderer.close()
  temporary.replace(video)
  # Decode the complete replay, not just the container header.
  reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
  video_metadata = next(reader)
  decoded = sum(1 for _ in reader)
  if decoded != len(actual) or abs(video_metadata["fps"] - FPS) > 0.01:
    raise RuntimeError(f"Replay decode mismatch: {clip} {decoded}/{len(actual)}")
  thumb_width = 512
  thumb_height = int(round(cfg.height * thumb_width / (cfg.width * 4)))
  columns = 2
  rows = math.ceil(len(keyframes) / columns)
  sheet = Image.new("RGB", (thumb_width * columns, (thumb_height + 22) * rows), "white")
  draw = ImageDraw.Draw(sheet)
  for index, (frame, image) in enumerate(keyframes):
    x, y = index % columns * thumb_width, index // columns * (thumb_height + 22)
    draw.text((x + 3, y + 3), f"{clip} | {frame / FPS:.2f}s", fill="black")
    sheet.paste(image.resize((thumb_width, thumb_height)), (x, y + 22))
  sheet_path = output / "keyframes" / f"{clip}.png"
  sheet.save(sheet_path)
  curves = plot_curves(output, clip, model, raw, actual, metadata)
  return {
    "clip_id": clip,
    "metadata_sha256": record["accepted"]["metadata_sha256"],
    "video": str(video.relative_to(root)),
    "video_sha256": sha256(video),
    "frames": decoded,
    "fps": FPS,
    "keyframes": str(sheet_path.relative_to(root)),
    "keyframes_sha256": sha256(sheet_path),
    "curves": str(Path(curves).relative_to(root)),
    "curves_sha256": sha256(Path(curves)),
    "sample_frame_indices": sorted(sample_frames),
  }


def main(cfg: Cfg):
  root = Path(cfg.input_dir)
  manifest = json.loads((root / "manifest.json").read_text())
  output = root / "replay"
  for name in ("videos", "keyframes", "curves"):
    (output / name).mkdir(parents=True, exist_ok=True)
  index_path = output / "index.json"
  renderer_identity = hashlib.sha256(
    (sha256(Path(__file__)) + f":{cfg.width}:{cfg.height}").encode()
  ).hexdigest()
  if index_path.exists():
    index = json.loads(index_path.read_text())
    if (
      index["identity_sha256"] != manifest["identity_sha256"]
      or index["renderer_identity"] != renderer_identity
    ):
      raise ValueError("Replay identity mismatch")
  else:
    index = {
      "identity_sha256": manifest["identity_sha256"],
      "renderer_identity": renderer_identity,
      "backend": "osmesa_cpu",
      "views": ["GMR front", "GMR side", "execution front", "execution side"],
      "clips": {},
    }
  review_path = output / "review.json"
  review = (
    json.loads(review_path.read_text())
    if review_path.exists()
    else {"identity_sha256": manifest["identity_sha256"], "clips": {}}
  )
  if review["identity_sha256"] != manifest["identity_sha256"]:
    raise ValueError("Review identity mismatch")
  if cfg.action == "review":
    if (
      not cfg.reviewer
      or cfg.decision not in ("pass", "reject")
      or not cfg.review_note
      or not cfg.clip_ids
    ):
      raise ValueError(
        "Review requires explicit clip IDs, reviewer, decision and evidence note"
      )
    for clip in cfg.clip_ids:
      item = index["clips"][clip]
      if sha256(root / item["video"]) != item["video_sha256"]:
        raise ValueError("Reviewed video identity mismatch")
      review["clips"][clip] = {
        "decision": cfg.decision,
        "reviewer": cfg.reviewer,
        "note": cfg.review_note,
        "video_sha256": item["video_sha256"],
        "time_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
      }
    atomic_json(review_path, review)
    return
  if cfg.action != "render":
    raise ValueError("Action must be render/review")
  selected = {
    clip: record
    for clip, record in manifest["clips"].items()
    if record["status"] == "accepted" and (not cfg.clip_ids or clip in cfg.clip_ids)
  }
  for clip, item in index["clips"].items():
    if clip in selected:
      for key in ("video", "keyframes", "curves"):
        if sha256(root / item[key]) != item[key + "_sha256"]:
          raise ValueError("Existing replay changed")
  pending = [
    record for clip, record in sorted(selected.items()) if clip not in index["clips"]
  ]
  with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
    futures = [pool.submit(render_one, str(root), record, cfg) for record in pending]
    for future in as_completed(futures):
      record = future.result()
      index["clips"][record["clip_id"]] = record
      review["clips"].setdefault(record["clip_id"], {"decision": "pending"})
      atomic_json(index_path, index)
      atomic_json(review_path, review)
      print(
        json.dumps({"rendered": record["clip_id"], "frames": record["frames"]}),
        flush=True,
      )


if __name__ == "__main__":
  main(tyro.cli(Cfg))
