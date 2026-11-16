"""Pinned, CPU-only G1 Moves / AMASS-G1 review with unchanged EXP-027 filters.

No training, retargeting, pose correction, angle scaling, or synthetic motion.
Only public numeric CSV/NPY files are loaded (never pickle or remote Python).
The 0.793 m AMASS root offset is the publisher's coordinate conversion, not
per-clip ground fitting. Source/subject assignments are frozen before screening.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import threading
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/reproduction/external_shoulder_data_review_v1"
PERSISTENT = Path(
  "/personal_fast_storage/smp_experiments/external_shoulder_data_review_v1_20261112"
)
REPOS = {
  "g1_moves": (
    "exptech/g1-moves",
    "895d064b2385725e6aabf540461c219afc3e0916",
  ),
  "amass_g1": (
    "fleaven/Retargeted_AMASS_for_robotics",
    "275c4c4a028cadf5241c10a613c2f11913531ad2",
  ),
}
G1_CLIPS = (
  "karate/M_Move5",
  "karate/M_Move19",
  "karate/M_Move3",
  "karate/M_Move1",
  "dance/B_StretchDance",
  "dance/B_SpiralDance",
  "dance/B_WiggleDance",
)
SUBSET_CAPS = {"ACCAD": 250, "MPI_HDM05": 200, "CMU": 300, "MPI_mosh": 100}
MAX_FILE_BYTES = 64 * 1024**2
MAX_DOWNLOAD_BYTES = 2 * 1024**3
SPLITS = ("train", "validation", "test")
BASE_DATA = ROOT / "logs/reproduction/conditional_shoulder80_prior_v1/data"
BASE_DATA_SHA = "7188e78d79d70dae7001fb69aa8c3fd357de220a01c258ed2dd348450b5252d9"


def digest(path):
  h = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024**2), b""):
      h.update(block)
  return h.hexdigest()


def canonical_hash(value):
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def utc_now():
  from datetime import datetime, timezone

  return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w") as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


def get_bytes(url, limit=MAX_FILE_BYTES):
  request = urllib.request.Request(url, headers={"User-Agent": "SMP-data-review/1"})
  with urllib.request.urlopen(request, timeout=45) as response:
    chunks, count = [], 0
    expected = response.headers.get("Content-Length")
    if expected is not None and int(expected) > limit:
      raise ValueError(f"Remote file exceeds per-file limit: {url}")
    while True:
      chunk = response.read(min(1024**2, limit + 1 - count))
      if not chunk:
        break
      chunks.append(chunk)
      count += len(chunk)
      if count > limit:
        raise ValueError(f"Download exceeds per-file limit: {url}")
    payload = b"".join(chunks)
    if expected is not None and len(payload) != int(expected):
      raise ValueError(f"HTTP body truncated: {len(payload)} != {expected}")
  if len(payload) > limit:
    raise ValueError(f"Download exceeds per-file limit: {url}")
  return payload


def url_for(dataset, path):
  repo, revision = REPOS[dataset]
  return (
    f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
    + urllib.parse.quote(path, safe="/")
  )


def stable_key(value):
  return hashlib.sha256(f"42:{value}".encode()).hexdigest()


def actor_group(dataset, path):
  parts = Path(path).parts
  if dataset == "g1_moves":
    prefix = parts[1].split("_")[0]
    if prefix not in ("B", "J", "M"):
      raise ValueError("Unverified performer or video-extracted G1 clip")
    return f"g1_moves:performer:{prefix}"
  if dataset != "amass_g1" or len(parts) != 4:
    raise ValueError(f"Unverified source grouping: {path}")
  subset, actor = parts[1:3]
  if subset == "ACCAD":
    match = re.match(r"(Female\d+|Male\d+)", actor)
    if not match:
      raise ValueError(f"Unverified ACCAD actor: {path}")
    actor = match[1]
  elif subset == "CMU":
    if not actor.isdigit():
      raise ValueError(f"Unverified CMU actor: {path}")
  elif subset == "MPI_HDM05":
    if not re.fullmatch(r"[a-z]{2}", actor):
      raise ValueError(f"Unverified HDM05 actor: {path}")
  elif subset == "MPI_mosh":
    if not actor.isdigit():
      raise ValueError(f"Unverified MPI actor: {path}")
  else:
    raise ValueError(f"Unsupported subject namespace: {path}")
  return f"amass:{subset}:subject:{actor}"


def group_split(group):
  # Never use acceptance counts to move actors between splits.
  bucket = int(stable_key(group), 16) % 5
  return "train" if bucket < 3 else "validation" if bucket == 3 else "test"


def source_fps(dataset, path):
  if dataset == "g1_moves":
    return 60.0
  match = re.search(r"_(\d+(?:\.\d+)?)_jpos\.npy$", path)
  if not match or not 20 <= float(match[1]) <= 300:
    raise ValueError(f"Unverified source frame rate: {path}")
  return float(match[1])


def resample_motion(raw, fps, target_fps=50):
  raw = np.asarray(raw, dtype=np.float64)
  if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 3:
    raise ValueError("Expected numeric (N,36) root+xyzw+29 joints")
  if not np.isfinite(raw).all():
    raise ValueError("NaN/Inf in source motion")
  if fps <= 0 or not np.isfinite(fps):
    raise ValueError("Invalid frame rate")
  if not np.allclose(np.linalg.norm(raw[:, 3:7], axis=1), 1, atol=1e-3):
    raise ValueError("Source root quaternion is not unit xyzw")
  times = np.arange(len(raw), dtype=np.float64) / fps
  target = np.arange(0.0, times[-1], 1.0 / target_fps)
  motion = np.empty((len(target), 36), dtype=np.float64)
  for col in (*range(3), *range(7, 36)):
    motion[:, col] = np.interp(target, times, raw[:, col])
  motion[:, 3:7] = Slerp(times, Rotation.from_quat(raw[:, 3:7]))(target).as_quat()
  return motion


def load_motion(path, dataset, fps):
  raw = (
    np.loadtxt(path, delimiter=",")
    if dataset == "g1_moves"
    else np.load(path, allow_pickle=False)
  )
  if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != 36:
    raise ValueError(f"Malformed numeric motion: {path}")
  raw = raw.astype(np.float64, copy=True)
  if dataset == "amass_g1":
    raw[:, 2] += 0.793
  return raw, resample_motion(raw, fps)


def check_storage():
  if not Path("/personal_fast_storage").is_mount():
    raise RuntimeError("Persistent storage is not mounted; no system disk fallback")
  if OUTPUT.is_symlink():
    if OUTPUT.resolve() != PERSISTENT:
      raise RuntimeError("Output symlink identity mismatch")
  elif OUTPUT.exists() or PERSISTENT.exists():
    raise RuntimeError("Refuse to reuse an unrelated experiment directory")
  else:
    PERSISTENT.mkdir(parents=True)
    OUTPUT.symlink_to(PERSISTENT, target_is_directory=True)
  free = shutil.disk_usage(OUTPUT).free
  if free < 5 * 1024**3:
    raise RuntimeError("Insufficient shared filesystem space")
  return {
    "physical_path": str(PERSISTENT),
    "free_bytes": free,
    "personal_quota_verified": False,
  }


def build_plan():
  storage = check_storage()
  path = OUTPUT / "plan.json"
  if path.exists():
    plan = json.loads(path.read_text())
    if plan["repos"] != {k: list(v) for k, v in REPOS.items()}:
      raise ValueError("Pinned dataset revisions changed")
    return plan
  from scripts.curate_conditional_shoulder80 import Cfg

  if digest(BASE_DATA / "manifest.json") != BASE_DATA_SHA:
    raise ValueError("Historical EXP-027 data identity mismatch")
  inventories, documents, sources, excluded = {}, [], [], []
  for dataset, (repo, revision) in REPOS.items():
    metadata = json.loads(
      get_bytes(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}")
    )
    if metadata["sha"] != revision:
      raise ValueError("Remote revision identity mismatch")
    save_json(OUTPUT / "metadata" / f"{dataset}_inventory.json", metadata)
    files = [item["rfilename"] for item in metadata["siblings"]]
    inventories[dataset] = {
      "files": len(files),
      "revision": revision,
      "sha256": digest(OUTPUT / "metadata" / f"{dataset}_inventory.json"),
    }
    docpaths = (
      ["DATASET_CARD.md", "README.md", "manifest.json", "quality_report.json"]
      + [f"{clip}/README.md" for clip in G1_CLIPS]
      if dataset == "g1_moves"
      else [
        "README.md",
        "g1/README.md",
        "g1/visualize.py",
        "g1/license.txt",
        "g1/amass.bib",
      ]
      + [
        p
        for p in files
        if p.startswith(tuple(f"g1/{s}/" for s in SUBSET_CAPS))
        and any(x in Path(p).name.lower() for x in ("license", "readme"))
      ]
    )
    for item in dict.fromkeys(docpaths):
      if item in files:
        documents.append({"dataset": dataset, "remote_path": item})
    if dataset == "g1_moves":
      selected = [f"{clip}/retarget/{Path(clip).name}.csv" for clip in G1_CLIPS]
      if not set(selected).issubset(files):
        raise ValueError("Required G1 Moves numeric files are absent")
    else:
      selected = []
      for subset, cap in SUBSET_CAPS.items():
        pools = defaultdict(list)
        for item in files:
          if not item.startswith(f"g1/{subset}/") or not item.endswith(".npy"):
            continue
          try:
            group = actor_group(dataset, item)
          except ValueError as error:
            excluded.append({"path": item, "reason": str(error)})
            continue
          pools[group].append(item)
        # Round-robin actors, not the first actor's many similar clips.
        pools = {group: sorted(items, key=stable_key) for group, items in pools.items()}
        ranked = []
        for index in range(max(map(len, pools.values()), default=0)):
          for group in sorted(pools, key=stable_key):
            if index < len(pools[group]):
              ranked.append(pools[group][index])
        selected.extend(ranked[:cap])
    for item in selected:
      group = actor_group(dataset, item)
      sources.append(
        {
          "dataset": dataset,
          "remote_path": item,
          "source_id": f"{dataset}/{item}",
          "source_group": group,
          "split": group_split(group),
          "fps": source_fps(dataset, item),
          "root_z_offset_m": 0.793 if dataset == "amass_g1" else 0.0,
        }
      )
  plan = {
    "created_at_utc": utc_now(),
    "experiment": "EXP-028",
    "scope": "data_review_only",
    "repos": REPOS,
    "storage": storage,
    "inventory": inventories,
    "documents": documents,
    "sources": sources,
    "excluded_unverified_groups": excluded,
    "selection": {
      "g1_clips": G1_CLIPS,
      "amass_subset_caps": SUBSET_CAPS,
      "algorithm": "seed42_sha256_actor_round_robin_no_motion_outcomes",
    },
    "split": "sha256('42:'+dataset_subject_id) mod 5: 0,1,2 train; 3 validation; 4 test",
    "lafan_split_unchanged": True,
    "filter_config": asdict(Cfg()),
    "filter_source_sha256": digest(ROOT / "scripts/curate_conditional_shoulder80.py"),
    "baseline_manifest_sha256": BASE_DATA_SHA,
    "max_download_bytes": MAX_DOWNLOAD_BYTES,
    "max_file_bytes": MAX_FILE_BYTES,
    "training_authorized": False,
  }
  save_json(path, plan)
  print(
    json.dumps(
      {
        "status": "plan_frozen",
        "sources": len(sources),
        "groups_by_split": {
          s: len({r["source_group"] for r in sources if r["split"] == s})
          for s in SPLITS
        },
      }
    ),
    flush=True,
  )
  return plan


def acquire(plan):
  items = plan["documents"] + plan["sources"]
  previous_path = OUTPUT / "downloads.json"
  previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
  results = dict(previous.get("files", {}))
  if previous and previous["plan_sha256"] != digest(OUTPUT / "plan.json"):
    raise ValueError("Download plan identity mismatch")

  budget_lock = threading.Lock()
  acquired_bytes = sum(v["bytes"] for v in results.values())

  def download(item):
    key = item["dataset"] + "/" + item["remote_path"]
    nonlocal acquired_bytes
    path = OUTPUT / "raw" / key
    if key in results:
      if digest(path) != results[key]["sha256"]:
        raise ValueError(f"Local source checksum mismatch: {key}")
      return key, results[key]
    payload = get_bytes(url_for(item["dataset"], item["remote_path"]))
    with budget_lock:
      if acquired_bytes + len(payload) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("Download storage budget exceeded; source not written")
      acquired_bytes += len(payload)
    if path.exists():
      # Reacquire the pinned bytes to authenticate an unrecorded interrupted write.
      if digest(path) != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"Unrecorded source differs from pinned revision: {key}")
    else:
      path.parent.mkdir(parents=True, exist_ok=True)
      temporary = path.with_suffix(path.suffix + ".partial")
      with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
      os.replace(temporary, path)
    return key, {
      "sha256": digest(path),
      "bytes": path.stat().st_size,
      "url": url_for(item["dataset"], item["remote_path"]),
    }

  errors = []
  with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(download, item): item for item in items}
    for index, future in enumerate(as_completed(futures), 1):
      try:
        key, result = future.result()
        results[key] = result
        if sum(v["bytes"] for v in results.values()) > MAX_DOWNLOAD_BYTES:
          raise RuntimeError("Download budget exceeded; stop acquisition")
      except Exception as error:
        errors.append({"item": futures[future], "error": str(error)})
      if index % 20 == 0 or index == len(items):
        save_json(
          previous_path,
          {
            "plan_sha256": digest(OUTPUT / "plan.json"),
            "files": results,
            "errors": errors,
            "updated_at_utc": utc_now(),
          },
        )
        print(
          json.dumps(
            {
              "downloaded": len(results),
              "total": len(items),
              "bytes": sum(v["bytes"] for v in results.values()),
              "errors": len(errors),
            }
          ),
          flush=True,
        )
  if errors:
    raise RuntimeError(
      f"{len(errors)} source downloads failed; rerun acquisition to resume"
    )
  return results


def coverage_gate(records, baseline):
  """Same numeric coverage/diversity gates as EXP-027; new subject namespaces."""
  from scripts.curate_conditional_shoulder80 import nonoverlap_count

  from smp.pretrain.shoulder80 import CONDITION_NAMES, PHASE_NAMES, subject_split

  groups, content_groups = {}, {}
  for r in records:
    if r["dataset"] == "lafan":
      expected = subject_split(r["source"])
    else:
      expected = group_split(r["source_group"])
    if r["split"] != expected:
      raise ValueError("Source subject split mismatch")
    group = r["source_group"]
    if group in groups and groups[group] != r["split"]:
      raise ValueError("Actor/mirror split leakage")
    groups[group] = r["split"]
    h = r["source_sha256"]
    if h in content_groups and content_groups[h] != r["split"]:
      raise ValueError("Identical source bytes cross splits")
    content_groups[h] = r["split"]
    c, p = int(r["condition_id"]), int(r["phase_id"])
    if c not in (0, 1, 2) or p not in ((0,) if c == 0 else (1, 2, 3)):
      raise ValueError("Invalid condition or phase")
  checks, coverage, comparisons = {}, {}, {}
  for split in SPLITS:
    rows = [r for r in records if r["split"] == split]
    coverage[split] = {}
    for c in range(3):
      for p in (0,) if c == 0 else (1, 2, 3):
        selected = [
          r for r in rows if int(r["condition_id"]) == c and int(r["phase_id"]) == p
        ]
        sources = {r["source_id"] for r in selected}
        key = f"{CONDITION_NAMES[c]}/{PHASE_NAMES[p]}"
        coverage[split][key] = {
          "windows": len(selected),
          "original_files": len(sources),
          "actors": len({r["source_group"] for r in selected}),
          "nonoverlap_windows_without_mirror_duplication": nonoverlap_count(
            [
              (
                r["source_sha256"],
                int(r["frame50_start"]),
                int(r["frame50_end_exclusive"]),
              )
              for r in selected
            ]
          ),
        }
        checks[f"{split}/{key}/nonempty"] = bool(selected)
        if split == "train" and c:
          checks[f"{split}/{key}/at_least_two_original_files"] = (
            len({r["source_sha256"] for r in selected}) >= 2
          )
    intervals = {
      (r["source_id"], int(r["frame50_start"]), int(r["frame50_end_exclusive"]))
      for r in rows
      if int(r["condition_id"])
    }
    prior = defaultdict(list)
    for source, a, b in baseline[split]:
      prior["lafan/" + source].append((a, b))
    novel = [
      r for r in intervals if not any(r[1] < b and a < r[2] for a, b in prior[r[0]])
    ]
    comparisons[split] = {
      "baseline_nonoverlap_windows": nonoverlap_count(baseline[split]),
      "new_nonoverlap_windows": nonoverlap_count(intervals),
      "novel_nonoverlap_windows_outside_baseline_intervals": nonoverlap_count(novel),
      "mirrors_count_as_independent": False,
    }
  comp = comparisons["train"]
  checks["train/shoulder_nonoverlap_count_increased"] = (
    comp["new_nonoverlap_windows"] > comp["baseline_nonoverlap_windows"]
  )
  checks["train/novel_shoulder_intervals_present"] = (
    comp["novel_nonoverlap_windows_outside_baseline_intervals"] > 0
  )
  return {
    "passed": all(checks.values()),
    "checks": checks,
    "failed_checks": [k for k, v in checks.items() if not v],
    "coverage": coverage,
    "baseline_comparison": comparisons,
  }


def review(plan, downloads):
  import torch
  from scripts.csv_to_npz import _compute_windows
  from scripts.curate_conditional_contact import screening_model
  from scripts.curate_conditional_shoulder80 import (
    Cfg,
    baseline_intervals,
    classify_windows,
    link_early_windows,
    phase_counts,
    screen_frames,
  )
  from scripts.curate_lafan_low_speed import mirror_g1_motion

  from smp.pretrain.shoulder80 import CONDITION_SCHEMA, PHASE_NAMES, SAMPLING

  torch.set_num_threads(2)
  cfg = Cfg()
  integrity = json.loads((OUTPUT / "publisher_integrity.json").read_text())
  if not integrity["passed"] or integrity["plan_sha256"] != digest(
    OUTPUT / "plan.json"
  ):
    raise RuntimeError("Publisher identity audit is missing or mismatched")
  identities = {r["source_id"]: r["sha256"] for r in integrity["verified"]}
  if (
    asdict(cfg) != plan["filter_config"]
    or digest(ROOT / "scripts/curate_conditional_shoulder80.py")
    != plan["filter_source_sha256"]
  ):
    raise RuntimeError("Frozen filtering rules changed")
  if digest(BASE_DATA / "manifest.json") != BASE_DATA_SHA:
    raise RuntimeError("Historical data changed")
  base = json.loads((BASE_DATA / "manifest.json").read_text())
  old_intervals, old_sha = baseline_intervals(
    ROOT / "logs/reproduction/conditional_contact_v2/data"
  )
  output = OUTPUT / "review"
  output.mkdir(exist_ok=True)
  state_path = output / "manifest.json"
  if state_path.exists():
    raise RuntimeError("Review already completed; do not overwrite immutable review")
  for item in base["files"]:
    if digest(BASE_DATA / item["path"]) != item["sha256"]:
      raise ValueError("Historical shard checksum mismatch")
  with (BASE_DATA / "windows.csv").open() as stream:
    records = list(csv.DictReader(stream))
  for row in records:
    row.update(
      dataset="lafan",
      source_id="lafan/" + row["source"],
      source_group="lafan:subject:" + re.search(r"subject(\d)", row["source"])[1],
      shard=str(BASE_DATA / row["shard"]),
    )
  existing_hashes = {row["source_sha256"] for row in records}
  files, audits, duplicate_sources = [], [], []
  screening = screening_model()
  for number, source in enumerate(plan["sources"], 1):
    key = source["source_id"]
    path = OUTPUT / "raw" / key
    source_sha = downloads[key]["sha256"]
    if source_sha != identities[key]:
      raise ValueError("Download does not match publisher integrity report")
    if digest(path) != source_sha:
      raise ValueError(f"Source checksum mismatch: {key}")
    if source_sha in existing_hashes:
      duplicate_sources.append(key)
      continue
    existing_hashes.add(source_sha)
    raw, original = load_motion(path, source["dataset"], source["fps"])
    raw_stats = {
      "frames": len(raw),
      "duration_s": (len(raw) - 1) / source["fps"],
      "waist_yaw_min": float(raw[:, 19].min()),
      "waist_yaw_max": float(raw[:, 19].max()),
      "root_z_min": float(raw[:, 2].min()),
      "root_z_max": float(raw[:, 2].max()),
    }
    per_source = []
    for mirrored in (False, True):
      motion = mirror_g1_motion(original) if mirrored else original
      result = classify_windows(motion, cfg)
      valid, ee, frame_audit = screen_frames(result, screening)
      starts, labels, phases = (result[k] for k in ("starts", "labels", "phases"))
      keep = (labels >= 0) & valid[result["indices"]].all(1)
      after_fk = phase_counts(labels, phases, keep)
      keep, links = link_early_windows(starts, labels, phases, keep, motion[:, 19], cfg)
      per_source.append(
        {
          "mirrored": mirrored,
          **result["audit"],
          "frames": frame_audit,
          "after_fk": after_fk,
          "after_early_link": phase_counts(labels, phases, keep),
        }
      )
      if not keep.any():
        continue
      pos, quat, vel, ang, joints = result["arrays"]
      windows = _compute_windows(
        *[
          torch.tensor(a, dtype=torch.float32)
          for a in (pos, quat, vel, ang, ee, joints)
        ],
        cfg.window_size,
        cfg.stride,
      ).numpy()[keep]
      if windows.shape[1:] != (10, 59) or not np.isfinite(windows).all():
        raise ValueError("Nonfinite or invalid features")
      stem = hashlib.sha256(key.encode()).hexdigest()[:16]
      relative = f"{source['split']}/{source['dataset']}_{stem}{'_mirror' if mirrored else ''}.npz"
      destination = output / relative
      destination.parent.mkdir(exist_ok=True)
      if destination.exists():
        raise RuntimeError(
          "Incomplete previous review: preserve it and investigate before retry"
        )
      np.savez_compressed(
        destination,
        windows=windows,
        condition_id=labels[keep],
        phase_id=phases[keep],
        frame50_start=starts[keep],
        metrics=result["metrics"][keep],
        early_link_frame50=links[keep],
      )
      files.append(
        {
          "path": relative,
          "sha256": digest(destination),
          "source_id": key,
          "split": source["split"],
          "source_sha256": source_sha,
          "mirrored": mirrored,
          "accepted_windows": len(windows),
        }
      )
      for index, old_index in enumerate(np.flatnonzero(keep)):
        c, p, start = (
          int(labels[old_index]),
          int(phases[old_index]),
          int(starts[old_index]),
        )
        records.append(
          {
            "dataset": source["dataset"],
            "source_id": key,
            "source": source["remote_path"],
            "source_group": source["source_group"],
            "source_sha256": source_sha,
            "split": source["split"],
            "mirrored": mirrored,
            "frame50_start": start,
            "frame50_end_exclusive": start + 10,
            "condition_id": c,
            "condition": CONDITION_SCHEMA["names"][c],
            "phase_id": p,
            "phase": PHASE_NAMES[p],
            "early_link_frame50": int(links[old_index]),
            "shard": str(destination),
            "index": index,
          }
        )
    audits.append(
      {
        **source,
        "source_sha256": source_sha,
        "raw_stats": raw_stats,
        "variants": per_source,
      }
    )
    if (
      number % 20 == 0
      or source["dataset"] == "g1_moves"
      or number == len(plan["sources"])
    ):
      save_json(
        output / "progress.json",
        {
          "processed": number,
          "total": len(plan["sources"]),
          "shards": len(files),
          "updated_at_utc": utc_now(),
        },
      )
      print(
        json.dumps(
          {
            "processed": number,
            "source": key,
            "kept": [v["after_early_link"] for v in per_source],
          }
        ),
        flush=True,
      )
  fields = list(records[0])
  with (output / "windows.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
  training = []
  for shard in sorted({r["shard"] for r in records if r["split"] == "train"}):
    with np.load(shard, allow_pickle=False) as data:
      training.append(data["windows"])
  if training:
    flat = np.concatenate(training).reshape(-1, 59)
    low, high = np.percentile(flat, [1, 99], axis=0).astype(np.float32)
    high = np.where(high - low < 1e-6, low + 1.0, high)
    np.savez(output / "norm_stats.npz", q_low=low, q_high=high)
  gate = coverage_gate(records, old_intervals)
  counts, phase_totals, per_dataset = {}, {}, {}
  for split in SPLITS:
    rows = [r for r in records if r["split"] == split]
    counts[split] = [sum(int(r["condition_id"]) == c for r in rows) for c in range(3)]
    phase_totals[split] = [
      [
        sum(int(r["condition_id"]) == c and int(r["phase_id"]) == p for r in rows)
        for p in range(4)
      ]
      for c in range(3)
    ]
  for dataset in ("lafan", "g1_moves", "amass_g1"):
    rows = [r for r in records if r["dataset"] == dataset]
    per_dataset[dataset] = {
      "counts": [sum(int(r["condition_id"]) == c for r in rows) for c in range(3)],
      "accepted_original_files": len({r["source_id"] for r in rows}),
      "accepted_actors": len({r["source_group"] for r in rows}),
    }
  save_json(
    output / "audit.json", {"sources": audits, "duplicates_excluded": duplicate_sources}
  )
  manifest = {
    "experiment": "EXP-028",
    "completed_at_utc": utc_now(),
    "status": "data_gate_passed_not_trained" if gate["passed"] else "blocked_data_gate",
    "plan_sha256": digest(OUTPUT / "plan.json"),
    "filter_config": asdict(cfg),
    "filter_source_sha256": plan["filter_source_sha256"],
    "review_script_sha256": digest(__file__),
    "publisher_integrity_sha256": digest(OUTPUT / "publisher_integrity.json"),
    "condition_schema": CONDITION_SCHEMA,
    "sampling": SAMPLING,
    "counts": counts,
    "phase_counts": phase_totals,
    "by_dataset": per_dataset,
    "gate": gate,
    "external_files": files,
    "historical_data_manifest": str(BASE_DATA / "manifest.json"),
    "historical_data_sha256": BASE_DATA_SHA,
    "old_v2_data_sha256": old_sha,
    "windows_manifest_sha256": digest(output / "windows.csv"),
    "audit_sha256": digest(output / "audit.json"),
    "norm_stats_sha256": digest(output / "norm_stats.npz") if training else None,
    "normalization": "only train actors, percentile 1/99; no validation/test data",
    "training_started": False,
    "prior_acceptance_evaluated": False,
    "limits": [
      "FK screening does not prove dynamic balance",
      "motion labels are desired responses, not measured shoulder contact",
      "partial AMASS inventory, not a claim about all AMASS motions",
      "subject namespaces are source-scoped; unreported cross-dataset actor aliases remain unknown",
    ],
  }
  save_json(state_path, manifest)
  print(
    json.dumps(
      {
        "status": manifest["status"],
        "counts": counts,
        "phase_counts": phase_totals,
        "by_dataset": per_dataset,
        "failed_checks": gate["failed_checks"],
      }
    ),
    flush=True,
  )
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("stage", choices=("plan", "download", "review", "all"))
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  plan = build_plan()
  if args.stage == "plan":
    return
  if args.stage in ("download", "all"):
    downloads = acquire(plan)
  else:
    saved = json.loads((OUTPUT / "downloads.json").read_text())
    if saved["plan_sha256"] != digest(OUTPUT / "plan.json") or saved["errors"]:
      raise ValueError("Incomplete or mismatched downloads")
    downloads = saved["files"]
  if args.stage in ("review", "all"):
    result = review(plan, downloads)
    raise SystemExit(0 if result["gate"]["passed"] else 2)


if __name__ == "__main__":
  main()
