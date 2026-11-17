"""Causal V3.92 shadow inference from observation-only episode files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from smp.force_estimation import load_force_estimator


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--model-dir", type=Path, required=True)
  parser.add_argument("--version", choices=("v357", "v382", "v392"), default="v392")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--max-gpu-memory-gib", type=float, default=30.0)
  args = parser.parse_args()
  files = sorted(args.input_dir.glob("*.npz"))
  if not files:
    raise RuntimeError(f"No input files in {args.input_dir}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  if args.device.startswith("cuda"):
    free, total = torch.cuda.mem_get_info(0)
    if args.max_gpu_memory_gib >= total / 1024**3:
      raise ValueError("GPU limit must be below physical device capacity")
    torch.cuda.set_per_process_memory_fraction(
      args.max_gpu_memory_gib / (total / 1024**3), 0
    )
    print(
      f"GPU free before load: {free / 1024**3:.2f} GiB; "
      f"own allocation limit: {args.max_gpu_memory_gib:.2f} GiB",
      flush=True,
    )
  estimator = load_force_estimator(
    args.model_dir, version=args.version, device=args.device
  )
  audits = []
  for source in files:
    destination = args.output_dir / source.name
    if destination.exists():
      raise FileExistsError(destination)
    with np.load(source, allow_pickle=False) as z:
      if set(z.files) != {"x", "tau_hist", "target_site_id", "step"}:
        raise ValueError(f"Observation-only schema mismatch in {source}: {z.files}")
      x = z["x"]
      tau = z["tau_hist"]
      site = z["target_site_id"]
      step = z["step"]
    n = len(step)
    if x.shape != (n, 930) or tau.shape != (n, 10, 29) or site.shape != (n,):
      raise ValueError(f"Invalid shapes in {source}")
    if not np.array_equal(step, np.arange(1, n + 1)):
      raise ValueError(f"Noncausal or missing step order in {source}")
    estimator.reset()
    forces = np.empty((n, 3), np.float32)
    points = np.empty((n, 3), np.float32)
    probabilities = np.empty(n, np.float32)
    start = time.monotonic()
    for i in range(n):
      prediction = estimator.step(x[i], tau[i], target_site_id=int(site[i]))
      forces[i] = prediction["force_body_n"]
      points[i] = prediction["point_body_m"]
      probabilities[i] = prediction["contact_probability"]
    np.savez_compressed(
      destination,
      step=step,
      estimated_force_body_n=forces,
      estimated_point_body_m=points,
      contact_probability=probabilities,
    )
    audit = {
      "episode": source.stem,
      "frames": n,
      "seconds": round(time.monotonic() - start, 3),
      "max_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 2)
      if args.device.startswith("cuda")
      else None,
    }
    audits.append(audit)
    print(json.dumps(audit), flush=True)
  (args.output_dir / "inference_audit.json").write_text(
    json.dumps(
      {"model": args.version, "device": args.device, "episodes": audits}, indent=2
    )
    + "\n",
    encoding="utf-8",
  )


if __name__ == "__main__":
  main()
