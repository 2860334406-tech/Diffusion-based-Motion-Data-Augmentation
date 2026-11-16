"""EXP-038 frozen-validation comparison, conditional samples, GSI and FK replays."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from smp.data.shoulder_ik import atomic_json
from smp.pretrain.conditioning import sample_condition, sha256
from smp.pretrain.shoulder_mixed import CONDITION_NAMES
from smp.pretrain.shoulder_mixed_lafan90 import (
  DATA_DIR,
  NORM_SHA,
  PARENT,
  PARENT_SHA,
  data_sha,
)
from smp.rl.utils import load_denoiser

OLD_ROOT = PARENT.parent
ROOT = DATA_DIR.parent


def summarize(windows, condition):
  x = torch.as_tensor(windows).cpu().float()
  if x.ndim != 3 or x.shape[1:] != (10, 59) or not torch.isfinite(x).all():
    raise FloatingPointError("Invalid generated motion windows")
  root_speed = torch.linalg.vector_norm(x[..., 53:55], dim=-1).amax(-1)
  joint_speed = (x[:, 1:, 9:38] - x[:, :-1, 9:38]).abs() * 50
  if not torch.isfinite(root_speed).all() or not torch.isfinite(joint_speed).all():
    raise FloatingPointError("Nonfinite motion diagnostics")
  waist_speed = joint_speed[..., 12].amax(-1)
  return {
    "samples": len(x),
    "finite": True,
    "window_max_root_planar_speed_p95_m_s": float(torch.quantile(root_speed, 0.95)),
    "window_max_joint_speed_p95_rad_s": float(
      torch.quantile(joint_speed.amax((1, 2)), 0.95)
    ),
    "window_max_waist_speed_p95_rad_s": float(torch.quantile(waist_speed, 0.95)),
    "root_height_p05_p50_p95_m": torch.quantile(
      x[..., 2].flatten(), torch.tensor([0.05, 0.5, 0.95])
    ).tolist(),
    "waist_final_p05_p50_p95_rad": torch.quantile(
      x[:, -1, 21], torch.tensor([0.05, 0.5, 0.95])
    ).tolist(),
    "waist_delta_abs_p95_rad": float(
      torch.quantile((x[:, -1, 21] - x[:, 0, 21]).abs(), 0.95)
    ),
    "expected_direction_rate": None
    if condition == 0
    else float((((1 if condition == 1 else -1) * x[:, -1, 21]) >= 0.05).float().mean()),
    "joint_speed_definition": "50Hz finite differences of raw adjacent joint angles; not motor commands",
  }


@torch.no_grad()
def generate(checkpoint, label):
  bundle = load_denoiser(str(checkpoint), "cuda:0")
  generated, metrics = {}, {}
  with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    initial = torch.randn(4096, 10, 59, device="cuda:0")
    for condition, name in enumerate(CONDITION_NAMES):
      with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(42035)
        torch.cuda.manual_seed_all(42035)
        windows = sample_condition(
          bundle, condition, 4096, initial_noise=initial, guidance_scale=1
        ).cpu()
      generated[name] = windows.numpy()
      metrics[name] = summarize(windows, condition)
      print(
        json.dumps({"generation": label, "condition": name, "metrics": metrics[name]}),
        flush=True,
      )
  del bundle, initial
  gc.collect()
  torch.cuda.empty_cache()
  return generated, metrics


def gsi_check(checkpoint, calibration):
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.force_unloading.conditional_shoulder_mixed import (
    shoulder_mixed_env_cfg,
  )

  cfg = shoulder_mixed_env_cfg(
    play=True, prior_path=str(checkpoint), calibration_path=str(calibration)
  )
  cfg.scene.num_envs = 64
  cfg.events["init_smp_state"].params.update(
    gsi_buffer_size=4096,
    gsi_batch_size=256,
    gsi_reject_unsafe=True,
    gsi_max_root_planar_speed=0.15,
    gsi_max_sampling_multiplier=20,
    compile_model=False,
  )
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  except RuntimeError as error:
    prefix = "Unable to fill the safe GSI pool within the sampling budget: "
    if not str(error).startswith(prefix):
      raise
    stats = ast.literal_eval(str(error)[len(prefix) :].split(". Increase ", 1)[0])
    if stats["nonfinite_rejected"]:
      raise FloatingPointError("Nonfinite GSI candidates") from error
    stats["pool_filled"] = False
    gc.collect()
    torch.cuda.empty_cache()
    return stats
  try:
    stats = dict(env._smp_gsi_safety_stats)
    stats["pool_filled"] = True
    if stats["nonfinite_rejected"]:
      raise FloatingPointError("Nonfinite GSI candidates")
    identity = env._smp_gsi_prior_identity
    if identity["schema"]["names"] != list(CONDITION_NAMES) or identity[
      "sha256"
    ] != sha256(checkpoint):
      raise ValueError("GSI loaded an unexpected condition schema")
  finally:
    env.close()
  gc.collect()
  torch.cuda.empty_cache()
  return stats


def plot_distributions(output, sets, labels):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
  for condition, name in enumerate(CONDITION_NAMES):
    for label, windows in sets.items():
      x = windows[name]
      waist = np.rad2deg(x[:, -1, 21])
      root_speed = np.linalg.norm(x[..., 53:55], axis=-1).max(-1)
      joint_speed = (np.abs(np.diff(x[..., 9:38], axis=1)) * 50).max((1, 2))
      axes[condition, 0].hist(
        waist, bins=70, density=True, histtype="step", label=labels[label]
      )
      for column, values in ((1, root_speed), (2, joint_speed)):
        values = np.sort(values)
        axes[condition, column].plot(
          values, np.arange(1, len(values) + 1) / len(values), label=labels[label]
        )
    for col, title in enumerate(
      ("Final waist yaw (deg)", "Max root speed (m/s)", "Max joint speed (rad/s)")
    ):
      axes[condition, col].set_title(f"{name}: {title}")
      axes[condition, col].grid(alpha=0.25)
      axes[condition, col].legend(fontsize=7)
  fig.suptitle(
    "EXP-038: fixed reference and equal-noise DDPM samples; kinematic statistics"
  )
  fig.savefig(output / "conditional_distributions.png", dpi=150)
  plt.close(fig)


def render_samples(output, samples):
  import imageio_ffmpeg
  import mujoco
  from PIL import Image, ImageDraw

  from smp.data.shoulder_ik import _build_model, _qpos_ids
  from smp.sampling.feature_to_state import rot6d_to_quat

  model = _build_model(soft_limits=False)
  qids = _qpos_ids(model)
  indices = np.random.default_rng(42).choice(4096, 8, replace=False)
  videos = []
  for name in CONDITION_NAMES:
    windows = samples[name][indices]
    rotations = rot6d_to_quat(torch.from_numpy(windows[..., 3:9])).numpy()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=280, width=320)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0, 0, 0.75)
    camera.distance, camera.elevation, camera.azimuth = 2.45, -8, 35
    frames = []
    try:
      for frame in range(10):
        cells = []
        for i in range(8):
          data.qpos[:] = model.qpos0
          data.qpos[:3] = windows[i, frame, :3]
          data.qpos[3:7] = rotations[i, frame]
          data.qpos[qids] = windows[i, frame, 9:38]
          data.qvel[:] = 0
          mujoco.mj_forward(model, data)
          renderer.update_scene(data, camera=camera)
          image = Image.fromarray(renderer.render().copy())
          draw = ImageDraw.Draw(image)
          draw.rectangle((0, 0, 320, 25), fill="black")
          yaw = np.rad2deg(windows[i, frame, 21])
          draw.text((4, 6), f"{name} #{indices[i]} yaw={yaw:+.1f}deg", fill="white")
          cells.append(np.asarray(image))
        frames.append(
          np.concatenate(
            (np.concatenate(cells[:4], 1), np.concatenate(cells[4:], 1)), 0
          )
        )
    finally:
      renderer.close()
    path = output / f"sample_{name}.mp4"
    temporary = path.with_suffix(".tmp.mp4")
    writer = imageio_ffmpeg.write_frames(
      str(temporary),
      (1280, 560),
      fps=5,
      codec="libx264",
      pix_fmt_in="rgb24",
      ffmpeg_log_level="error",
      output_params=["-crf", "21", "-preset", "fast"],
    )
    writer.send(None)
    try:
      for _ in range(3):
        for frame in frames:
          writer.send(frame)
    finally:
      writer.close()
    reader = imageio_ffmpeg.read_frames(str(temporary))
    meta = next(reader)
    count = sum(1 for _ in reader)
    if count != 30 or abs(meta["fps"] - 5) > 0.01:
      raise RuntimeError("Incomplete Prior video encoding")
    temporary.replace(path)
    Image.fromarray(frames[-1]).save(output / f"sample_{name}.png")
    videos.append(
      {
        "condition": name,
        "path": path.name,
        "sample_indices": indices.tolist(),
        "frames": count,
        "fps": 5,
        "original_fps": 50,
        "loops": 3,
        "slowdown": 10,
        "mode": "kinematic FK of eight independent unfiltered windows; no dynamics or PPO",
      }
    )
  return videos


def assess(output):
  torch.set_num_threads(2)
  output.mkdir(parents=True, exist_ok=True)
  run = json.loads((ROOT / "prior/run.json").read_text())
  if run["completed_epochs"] != 10000 or run["data_sha256"] != data_sha():
    raise ValueError("Assessment requires completed EXP-038 formal training")
  chosen = run["selected"] or min(
    run["evaluations"], key=lambda x: x["weighted_validation_loss"]
  )
  checkpoint, calibration = Path(chosen["checkpoint"]), Path(chosen["calibration"])
  for path, expected in (
    (checkpoint, chosen["checkpoint_sha256"]),
    (calibration, chosen["calibration_sha256"]),
  ):
    if sha256(path) != expected:
      raise ValueError("Assessment candidate identity mismatch")
  if (output / "artifact_manifest.json").exists():
    index = json.loads((output / "artifact_manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    if report["checkpoint_sha256"] != chosen["checkpoint_sha256"]:
      raise ValueError("Cached assessment belongs to a different candidate")
    for name, digest in index["files"].items():
      if sha256(output / name) != digest:
        raise ValueError(f"Assessment artifact changed: {name}")
    return report

  old_run = json.loads((OLD_ROOT / "prior/run.json").read_text())
  old_best = min(old_run["evaluations"], key=lambda x: x["weighted_validation_loss"])
  old_final = next(item for item in old_run["evaluations"] if item["epoch"] == 10000)
  sets, metrics, labels, comparison = {}, {}, {}, {}
  new_report = json.loads(Path(chosen["report"]).read_text())
  with np.load(Path(chosen["report"]).parent / "paired_scores.npz") as scores:
    reference_indices = {
      key: scores[key].copy() for key in scores.files if key.endswith("_indices")
    }
  for label, item in (
    ("new", chosen),
    ("old_best", old_best),
    ("old_final", old_final),
  ):
    if sha256(item["checkpoint"]) != item["checkpoint_sha256"]:
      raise ValueError("Comparison checkpoint identity mismatch")
    old_payload = torch.load(item["checkpoint"], map_location="cpu", weights_only=False)
    if old_payload["normalization_sha256"] != NORM_SHA:
      raise ValueError("Comparison normalization is not the frozen EXP-036 version")
    del old_payload
    report = json.loads(Path(item["report"]).read_text())
    with np.load(Path(item["report"]).parent / "paired_scores.npz") as scores:
      for key, expected in reference_indices.items():
        if not np.array_equal(scores[key], expected):
          raise ValueError("Validation comparison windows differ")
    sets[label], metrics[label] = generate(item["checkpoint"], label)
    labels[label] = f"{label} epoch {item['epoch']}"
    comparison[label] = {
      "epoch": item["epoch"],
      "checkpoint_sha256": item["checkpoint_sha256"],
      "shoulders": report["shoulders"],
      "stand_bias": report["stand_bias"],
      "gate": report["gate"],
      "weighted_validation_loss": report["losses"]["weighted_loss"],
    }
    np.savez_compressed(output / f"generated_{label}.npz", **sets[label])

  from smp.pretrain.shoulder_mixed_p01p02_dataset import ShoulderMixedP01P02Dataset

  reference = ShoulderMixedP01P02Dataset(PARENT, "train", PARENT_SHA)
  sets["reference"] = {
    name: reference.raw[reference.sample_indices(c, 4096, 42036 + c)].numpy()
    for c, name in enumerate(CONDITION_NAMES)
  }
  metrics["reference"] = {
    name: summarize(sets["reference"][name], c)
    for c, name in enumerate(CONDITION_NAMES)
  }
  labels["reference"] = "Frozen original training reference"
  np.savez_compressed(output / "training_reference.npz", **sets["reference"])
  del reference
  gsi_stats = gsi_check(checkpoint, calibration)
  plot_distributions(output, sets, labels)
  videos = render_samples(output, sets["new"])
  checks = {
    "validation_score_gate_passed": bool(new_report["gate"]["passed"]),
    "all_generated_samples_finite": all(m["finite"] for m in metrics["new"].values()),
    "stand_speed_p95_le_0.15": metrics["new"]["stand"][
      "window_max_root_planar_speed_p95_m_s"
    ]
    <= 0.15,
    "gsi_acceptance_ge_0.50": float(gsi_stats["acceptance_rate"]) >= 0.5,
    "left_direction_rate_ge_0.80": metrics["new"]["left_shoulder"][
      "expected_direction_rate"
    ]
    >= 0.8,
    "right_direction_rate_ge_0.80": metrics["new"]["right_shoulder"][
      "expected_direction_rate"
    ]
    >= 0.8,
  }
  report = {
    "experiment": "EXP-038",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": chosen["checkpoint_sha256"],
    "calibration": str(calibration),
    "calibration_sha256": chosen["calibration_sha256"],
    "data_sha256": data_sha(),
    "normalization_sha256": NORM_SHA,
    "normalization_recomputed": False,
    "formal_selected": run["selected"],
    "diagnostic_if_no_passing_candidate": run["selected"] is None,
    "completed_epochs": run["completed_epochs"],
    "optimizer_steps": run["optimizer_steps"],
    "generation": metrics,
    "gsi_safety": gsi_stats,
    "comparison": comparison,
    "videos": videos,
    "comparison_protocol": {
      "validation_windows_verified_identical": True,
      "validation_weights_and_negative_selection_unchanged": True,
      "paired_score_noise_repeats": 64,
      "generation_initial_noise_seed": 42,
      "generation_posterior_noise_seed": 42035,
      "guidance_scale": 1,
    },
    "gate": {
      "passed": all(checks.values()),
      "checks": checks,
      "failed_checks": [k for k, v in checks.items() if not v],
    },
    "test_or_p03_evaluated": False,
    "ppo_started": False,
    "prior_published": False,
    "wandb_url": run["wandb_url"],
    "script_sha256": sha256(__file__),
  }
  atomic_json(output / "report.json", report)
  rows = [
    {"model": label, "condition": name, **values}
    for label, values_by_condition in metrics.items()
    for name, values in values_by_condition.items()
  ]
  with (output / "distribution_summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  videos_html = "\n".join(
    f'<h2>{v["condition"]}</h2><video controls loop src="{v["path"]}" width="960"></video>'
    for v in videos
  )
  (output / "overview.html").write_text(
    '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EXP-038 Prior 分布</title>'
    "<h1>EXP-038 Prior 结果</h1><p>静止内部 LAFAN90%；整体静止20%、左右肩各40%。固定旧归一化和验证基准。</p>"
    "<p>回放为8个独立10帧采样窗口，慢放10倍、循环3次；采用FK显示，不代表动力学或PPO结果。</p>"
    '<p><a href="report.json">评估JSON</a> | <a href="distribution_summary.csv">分布CSV</a></p>'
    '<img src="conditional_distributions.png" style="max-width:100%">'
    + videos_html
    + "</html>"
  )
  atomic_json(
    output / "artifact_manifest.json",
    {
      "files": {
        str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
      }
    },
  )
  print(
    json.dumps({"assessment": str(output / "report.json"), "gate": report["gate"]}),
    flush=True,
  )
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  assess(parser.parse_args().output)
