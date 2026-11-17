"""Run a frozen EXP-022 candidate once on complete held-out episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from smp.force_estimation.exp022_proprio import Exp022ProprioModel
from train_exp022_proprio_estimator import evaluate, load_episodes, sha256


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--candidate-dir", type=Path, required=True)
  parser.add_argument("--capture-dir", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument(
    "--split", choices=("validation", "test", "repaired_test"), required=True
  )
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  manifest = json.loads((args.candidate_dir / "manifest.json").read_text())
  model_file = args.candidate_dir / "candidate.joblib"
  if sha256(model_file) != manifest["model_sha256"]:
    raise ValueError("Frozen model checksum mismatch")
  episodes = load_episodes(args.capture_dir)
  if args.split == "repaired_test":
    quality_file = args.capture_dir.parent / "quality_selection.json"
    quality = json.loads(quality_file.read_text())
    if {episode.name for episode in episodes} != set(quality["selected_episodes"]):
      raise ValueError("Quality-selected test episodes differ from frozen plan")
    if (args.out_dir / "metrics.json").exists():
      raise FileExistsError("Repaired locked test already evaluated here")
  elif args.split == "test":
    lower, upper = manifest["test_seeds"]
    expected = {
      f"{seed}_{mode}"
      for seed in range(lower, upper + 1)
      for mode in ("oracle", "masked")
    }
    if {episode.name for episode in episodes} != expected:
      raise ValueError("Locked test episodes differ from frozen manifest")
    if (args.out_dir / "metrics.json").exists():
      raise FileExistsError("Locked test has already been evaluated here")
  else:
    if {episode.name for episode in episodes} != set(manifest["validation_episodes"]):
      raise ValueError("Validation episodes differ from frozen manifest")
  bundle = joblib.load(model_file)
  result = evaluate(bundle, episodes)
  prediction_dir = args.out_dir / "predictions"
  prediction_dir.mkdir(exist_ok=True)
  model = Exp022ProprioModel(bundle)
  for episode in episodes:
    force, probability = model.predict_episode(episode.x, episode.tau)
    np.savez_compressed(
      prediction_dir / f"{episode.name}.npz",
      estimated_force_body_n=force,
      contact_probability=probability,
      step=np.arange(1, len(force) + 1, dtype=np.int64),
    )
  first = episodes[0]
  batch_force, batch_probability = model.predict_episode(first.x, first.tau)
  model.reset()
  stream = [model.step(first.x[i], first.tau[i]) for i in range(len(first.x))]
  force_difference = float(
    np.max(np.abs(batch_force - np.stack([item[0] for item in stream])))
  )
  probability_difference = float(
    np.max(np.abs(batch_probability - np.array([item[1] for item in stream])))
  )
  if force_difference > 1e-4 or probability_difference > 1e-5:
    raise ValueError("Stream and batch prediction differ")
  result["model_sha256"] = manifest["model_sha256"]
  result["streaming_parity"] = {
    "episode": first.name,
    "max_force_difference_n": force_difference,
    "max_probability_difference": probability_difference,
  }
  (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
  main()
