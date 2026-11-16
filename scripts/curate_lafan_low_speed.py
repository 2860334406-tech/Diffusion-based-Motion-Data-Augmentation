"""Build the balanced low-speed LAFAN G1 CSV corpus from a clip manifest.

The manifest uses half-open time intervals [start_s, end_s). Each accepted
three-second clip produces one original and one sagittally mirrored CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro

CSV_COLUMNS = 36
NUM_JOINTS = 29

_MIRROR_SOURCE = np.array(
  [
    6,
    7,
    8,
    9,
    10,
    11,
    0,
    1,
    2,
    3,
    4,
    5,
    12,
    13,
    14,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
  ],
  dtype=np.int64,
)
_MIRROR_SIGN = np.array(
  [
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
  ],
  dtype=np.float64,
)


@dataclass(frozen=True)
class ClipSpec:
  clip_id: str
  source: str
  start_s: float
  end_s: float
  category: str
  status: str
  reason: str = ""


@dataclass
class Cfg:
  input_dir: str = "datasets/csv/lafan"
  """Directory containing the full, unmodified LAFAN G1 CSV files."""
  manifest: str = "docs/LAFAN_LOW_SPEED_BALANCE_MANIFEST.csv"
  output_dir: str = "datasets/csv/lafan_low_speed_balance"
  input_fps: int = 30
  expected_duration_s: float = 3.0


def load_manifest(path: str | Path) -> list[ClipSpec]:
  with Path(path).open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  required = {"clip_id", "source", "start_s", "end_s", "category", "status"}
  if not rows or not required.issubset(rows[0]):
    missing = required - (set(rows[0]) if rows else set())
    raise ValueError(f"Manifest is empty or missing columns: {sorted(missing)}")
  clips = [
    ClipSpec(
      clip_id=row["clip_id"].strip(),
      source=row["source"].strip(),
      start_s=float(row["start_s"]),
      end_s=float(row["end_s"]),
      category=row["category"].strip(),
      status=row["status"].strip(),
      reason=row.get("reason", "").strip(),
    )
    for row in rows
  ]
  ids = [clip.clip_id for clip in clips]
  if len(ids) != len(set(ids)):
    raise ValueError("Manifest clip_id values must be unique")
  return clips


def mirror_g1_motion(motion: np.ndarray) -> np.ndarray:
  """Mirror G1 CSV rows across the sagittal x-z plane."""
  motion = np.asarray(motion)
  if motion.ndim != 2 or motion.shape[1] != CSV_COLUMNS:
    raise ValueError(f"Expected (N, {CSV_COLUMNS}) motion, got {motion.shape}")
  mirrored = motion.copy()
  mirrored[:, 1] = -motion[:, 1]
  mirrored[:, 3] = -motion[:, 3]
  mirrored[:, 4] = motion[:, 4]
  mirrored[:, 5] = -motion[:, 5]
  mirrored[:, 6] = motion[:, 6]
  joints = motion[:, 7 : 7 + NUM_JOINTS]
  mirrored[:, 7 : 7 + NUM_JOINTS] = joints[:, _MIRROR_SOURCE] * _MIRROR_SIGN[None, :]
  return mirrored


def crop_clip(
  motion: np.ndarray,
  start_s: float,
  end_s: float,
  input_fps: int,
  expected_duration_s: float,
) -> np.ndarray:
  if motion.ndim != 2 or motion.shape[1] != CSV_COLUMNS:
    raise ValueError(f"Expected (N, {CSV_COLUMNS}) motion, got {motion.shape}")
  if not np.isfinite(motion).all():
    raise ValueError("Motion contains NaN or Inf")
  start = round(start_s * input_fps)
  end = round(end_s * input_fps)
  expected = round(expected_duration_s * input_fps)
  if start < 0 or end > motion.shape[0] or end <= start:
    raise ValueError(f"Invalid frame range [{start}, {end}) for {motion.shape[0]} rows")
  clip = motion[start:end].copy()
  if clip.shape[0] != expected:
    raise ValueError(
      f"Expected exactly {expected} frames, got {clip.shape[0]} "
      f"from [{start_s}, {end_s})"
    )
  return clip


def curate(cfg: Cfg) -> dict[str, object]:
  input_dir = Path(cfg.input_dir)
  output_dir = Path(cfg.output_dir)
  clips = [clip for clip in load_manifest(cfg.manifest) if clip.status == "accepted"]
  if not clips:
    raise ValueError("Manifest has no rows with status=accepted")
  categories = {category: 0 for category in ("stand", "dynamic")}
  for clip in clips:
    if clip.category not in categories:
      raise ValueError(f"Unsupported category {clip.category!r} for {clip.clip_id}")
    categories[clip.category] += 1
  if categories != {"stand": 6, "dynamic": 6}:
    raise ValueError(f"Expected six accepted clips per category, got {categories}")

  expected_names = {
    name
    for clip in clips
    for name in (f"{clip.clip_id}.csv", f"{clip.clip_id}_mirror.csv")
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  unexpected = {path.name for path in output_dir.glob("*.csv")} - expected_names
  if unexpected:
    raise RuntimeError(
      "Refusing to mix curated data with unexpected CSV files: "
      + ", ".join(sorted(unexpected))
    )

  records: list[dict[str, object]] = []
  for clip in clips:
    source_path = input_dir / clip.source
    motion = np.atleast_2d(np.loadtxt(source_path, delimiter=","))
    cropped = crop_clip(
      motion, clip.start_s, clip.end_s, cfg.input_fps, cfg.expected_duration_s
    )
    original_path = output_dir / f"{clip.clip_id}.csv"
    mirror_path = output_dir / f"{clip.clip_id}_mirror.csv"
    np.savetxt(original_path, cropped, delimiter=",", fmt="%.10g")
    np.savetxt(mirror_path, mirror_g1_motion(cropped), delimiter=",", fmt="%.10g")
    records.append(
      {
        **asdict(clip),
        "start_frame": round(clip.start_s * cfg.input_fps),
        "end_frame_exclusive": round(clip.end_s * cfg.input_fps),
        "frames": int(cropped.shape[0]),
        "outputs": [original_path.name, mirror_path.name],
      }
    )

  summary: dict[str, object] = {
    "input_fps": cfg.input_fps,
    "categories": categories,
    "accepted_source_clips": len(clips),
    "output_csv_files": len(expected_names),
    "source_seconds_per_category": {
      key: value * cfg.expected_duration_s for key, value in categories.items()
    },
    "augmented_seconds_per_category": {
      key: value * cfg.expected_duration_s * 2 for key, value in categories.items()
    },
    "clips": records,
  }
  summary_path = output_dir / "curation_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))
  return summary


if __name__ == "__main__":
  curate(tyro.cli(Cfg))
