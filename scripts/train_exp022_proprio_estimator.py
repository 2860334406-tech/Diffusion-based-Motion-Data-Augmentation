"""Fit an EXP-022-only observable force head with complete-run validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from smp.force_estimation.exp022_proprio import Exp022ProprioModel, episode_features


@dataclass
class Episode:
  name: str
  x: np.ndarray
  tau: np.ndarray
  truth: np.ndarray
  contact: np.ndarray


def load_episodes(directory: Path) -> list[Episode]:
  episodes = []
  for path in sorted(directory.glob("*/shadow_inputs.npz")):
    with np.load(path, allow_pickle=False) as z:
      x = z["x"]
      tau = z["tau_hist"]
      truth = z["true_force_body_n"]
      contact = z["contact_geometries"] != ""
    if truth.shape != (len(x), 3):
      raise ValueError(f"Bad force labels: {path}")
    episodes.append(Episode(path.parent.name, x, tau, truth, contact))
  if not episodes:
    raise RuntimeError(f"No episodes in {directory}")
  return episodes


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
  cosine = np.sum(a * b, axis=1) / np.maximum(norm, 1e-8)
  return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def evaluate(bundle: dict, episodes: list[Episode]) -> dict:
  model = Exp022ProprioModel(bundle)
  rows = []
  for episode in episodes:
    force, probability = model.predict_episode(episode.x, episode.tau)
    truth = episode.truth
    true_n = np.linalg.norm(truth, axis=1)
    pred_n = np.linalg.norm(force, axis=1)
    filtered = np.empty_like(pred_n)
    previous = 0.0
    for i, magnitude in enumerate(pred_n):
      previous = 0.8 * previous + 0.2 * float(magnitude)
      filtered[i] = previous
    actor_active = filtered > 0.0075 * 327.07660302
    contacts = np.flatnonzero(episode.contact)
    first = int(contacts[0]) if len(contacts) else len(truth)
    pre = np.arange(len(truth)) < first
    zero = true_n < 1
    active = true_n >= 10
    first_active = int(np.flatnonzero(active)[0]) if active.any() else None
    if first_active is not None:
      future_detected = np.flatnonzero(pred_n[first_active : first_active + 11] >= 10)
      detection_delay = int(future_detected[0]) if len(future_detected) else 11
    else:
      detection_delay = None
    direction = angle_deg(truth, force)
    direction_valid = active & (pred_n >= 10)
    peak = int(np.argmax(true_n))
    rows.append(
      {
        "episode": episode.name,
        "frames": len(truth),
        "precontact_max_est_n": float(pred_n[pre].max()) if pre.any() else 0.0,
        "precontact_fp10_frames": int(np.count_nonzero(pred_n[pre] >= 10)),
        "precontact_actor_active_frames": int(np.count_nonzero(actor_active & pre)),
        "zero_force_fp10_frames": int(np.count_nonzero(pred_n[zero] >= 10)),
        "precontact_max_probability": float(probability[pre].max())
        if pre.any()
        else None,
        "first_loaded_missed_10n": bool(pred_n[first_active] < 10)
        if first_active is not None
        else None,
        "first_loaded_detection_delay_frames": detection_delay,
        "loaded_frames": int(active.sum()),
        "loaded_miss10_frames": int(np.count_nonzero(pred_n[active] < 10)),
        "loaded_vector_mae_n": float(
          np.linalg.norm(truth[active] - force[active], axis=1).mean()
        )
        if active.any()
        else None,
        "loaded_magnitude_mae_n": float(np.abs(true_n[active] - pred_n[active]).mean())
        if active.any()
        else None,
        "loaded_direction_detected_mean_deg": float(direction[direction_valid].mean())
        if direction_valid.any()
        else None,
        "true_peak_n": float(true_n[peak]),
        "estimate_at_true_peak_n": float(pred_n[peak]),
        "true_peak_magnitude_abs_error_n": float(abs(true_n[peak] - pred_n[peak])),
        "true_peak_direction_deg": float(direction[peak])
        if pred_n[peak] >= 10
        else None,
        "worst_loaded_vector_n": float(
          np.linalg.norm(truth[active] - force[active], axis=1).max()
        )
        if active.any()
        else None,
      }
    )
  active_rows = [row for row in rows if row["loaded_frames"]]
  summary = {
    "episodes": len(rows),
    "precontact_fp_episodes": sum(row["precontact_fp10_frames"] > 0 for row in rows),
    "precontact_fp_frames": sum(row["precontact_fp10_frames"] for row in rows),
    "precontact_actor_active_episodes": sum(
      row["precontact_actor_active_frames"] > 0 for row in rows
    ),
    "precontact_actor_active_frames": sum(
      row["precontact_actor_active_frames"] for row in rows
    ),
    "zero_force_fp10_frames": sum(row["zero_force_fp10_frames"] for row in rows),
    "precontact_worst_n": max(row["precontact_max_est_n"] for row in rows),
    "first_loaded_misses": sum(row["first_loaded_missed_10n"] for row in active_rows),
    "median_first_loaded_delay_frames": float(
      np.median([row["first_loaded_detection_delay_frames"] for row in active_rows])
    ),
    "loaded_frames": sum(row["loaded_frames"] for row in active_rows),
    "loaded_miss10_frames": sum(row["loaded_miss10_frames"] for row in active_rows),
    "median_run_vector_mae_n": float(
      np.median([row["loaded_vector_mae_n"] for row in active_rows])
    ),
    "worst_run_vector_mae_n": max(row["loaded_vector_mae_n"] for row in active_rows),
    "median_run_magnitude_mae_n": float(
      np.median([row["loaded_magnitude_mae_n"] for row in active_rows])
    ),
    "median_detected_direction_deg": float(
      np.median(
        [
          row["loaded_direction_detected_mean_deg"]
          for row in active_rows
          if row["loaded_direction_detected_mean_deg"] is not None
        ]
      )
    ),
    "median_peak_magnitude_abs_error_n": float(
      np.median([row["true_peak_magnitude_abs_error_n"] for row in rows])
    ),
    "worst_single_loaded_vector_n": max(
      row["worst_loaded_vector_n"] for row in active_rows
    ),
  }
  summary["selection_score"] = (
    summary["median_run_vector_mae_n"]
    + 0.5 * summary["worst_run_vector_mae_n"]
    + 25 * summary["precontact_fp_episodes"] / summary["episodes"]
    + 0.1 * summary["precontact_fp_frames"]
    + 40 * summary["precontact_actor_active_episodes"] / summary["episodes"]
    + 0.1 * summary["precontact_actor_active_frames"]
    + 0.03 * summary["zero_force_fp10_frames"]
    + 25 * summary["loaded_miss10_frames"] / max(1, summary["loaded_frames"])
    + 10 * summary["first_loaded_misses"] / max(1, len(active_rows))
  )
  return {"summary": summary, "per_episode": rows}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-dir", type=Path, required=True)
  parser.add_argument("--validation-dir", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument(
    "--train-modes", choices=("both", "oracle", "masked"), default="both"
  )
  parser.add_argument("--test-seed-start", type=int, default=121)
  parser.add_argument("--test-seed-end", type=int, default=132)
  args = parser.parse_args()
  if args.test_seed_start > args.test_seed_end:
    raise ValueError("Invalid locked test seed range")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  train = load_episodes(args.train_dir)
  if args.train_modes != "both":
    train = [episode for episode in train if episode.name.endswith(args.train_modes)]
  validation = load_episodes(args.validation_dir)
  train_seeds = {episode.name.split("_")[0] for episode in train}
  val_seeds = {episode.name.split("_")[0] for episode in validation}
  if train_seeds & val_seeds:
    raise ValueError("Train and validation seeds overlap")
  results = []
  best = None
  for variant in (
    "raw",
    "raw_plus_acc",
    "raw_plus_initial_delta",
    "raw_plus_acc_and_initial_delta",
    "full_history",
  ):
    chunks = [episode_features(ep.x, ep.tau, variant=variant) for ep in train]
    features = np.concatenate(chunks)
    truth = np.concatenate([ep.truth for ep in train])
    run_weight = np.concatenate(
      [np.full(len(ep.x), 1.0 / len(ep.x), dtype=np.float64) for ep in train]
    )
    run_weight *= len(features) / len(train)
    scaler = StandardScaler().fit(features)
    scaled = np.clip(scaler.transform(features), -10, 10)
    contact = (np.linalg.norm(truth, axis=1) >= 2).astype(int)
    for alpha in (300.0, 1000.0):
      regressor = Ridge(alpha=alpha).fit(
        scaled[contact > 0], truth[contact > 0], sample_weight=run_weight[contact > 0]
      )
      for c in (0.03, 0.3):
        classifier = LogisticRegression(
          C=c, max_iter=1000, class_weight="balanced", random_state=20261117
        ).fit(scaled, contact, sample_weight=run_weight)
        for warmup_frames in (10, 20, 30):
          for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
            bundle = {
              "feature_contract": "EXP022 50Hz observable x930 and tau_hist10x29",
              "variant": variant,
              "scaler": scaler,
              "classifier": classifier,
              "regressor": regressor,
              "threshold": threshold,
              "warmup_frames": warmup_frames,
            }
            metrics = evaluate(bundle, validation)
            summary = {
              "variant": variant,
              "ridge_alpha": alpha,
              "logistic_c": c,
              "threshold": threshold,
              "warmup_frames": warmup_frames,
              **metrics["summary"],
            }
            results.append(summary)
            if best is None or summary["selection_score"] < best[0]["selection_score"]:
              best = (summary, bundle, metrics)
            print(json.dumps(summary), flush=True)
  assert best is not None
  selected, bundle, metrics = best
  model_path = args.out_dir / "candidate.joblib"
  joblib.dump(bundle, model_path)
  (args.out_dir / "validation_metrics.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
  )
  (args.out_dir / "selection.json").write_text(json.dumps(results, indent=2) + "\n")
  manifest = {
    "name": "EXP022 observable-only domain adaptation pilot",
    "train_modes": args.train_modes,
    "status": "frozen before independent test",
    "sklearn_version": sklearn.__version__,
    "feature_code_sha256": sha256(
      Path(__file__).resolve().parents[1] / "src/smp/force_estimation/exp022_proprio.py"
    ),
    "training_code_sha256": sha256(Path(__file__)),
    "model_sha256": sha256(model_path),
    "train_episodes": [ep.name for ep in train],
    "validation_episodes": [ep.name for ep in validation],
    "train_input_sha256": {
      ep.name: sha256(args.train_dir / ep.name / "shadow_inputs.npz") for ep in train
    },
    "validation_input_sha256": {
      ep.name: sha256(args.validation_dir / ep.name / "shadow_inputs.npz")
      for ep in validation
    },
    "selected": selected,
    "test_seeds": [args.test_seed_start, args.test_seed_end],
    "limitations": "Chest target, sphere contact and EXP022 policy only; not a V3.92 global replacement",
  }
  design_file = args.train_dir.parent / "design.json"
  if design_file.exists():
    manifest["dataset_design_sha256"] = sha256(design_file)
  (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print("SELECTED", json.dumps(selected), flush=True)


if __name__ == "__main__":
  main()
