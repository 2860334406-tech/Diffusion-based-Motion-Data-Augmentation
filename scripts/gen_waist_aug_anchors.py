"""Generate MuJoCo-validated anchor trajectories for waist augmentation.

Anchors are the "physical prior" the diffusion model learns from: parameterised
stand-to-twist trajectories produced by ``ShoulderIkGenerator`` (Mink IK + MuJoCo
tracking + acceptance gates). The grid comes from ``smp.data.waist_aug_specs``.

Usage:
    uv run --no-sync python scripts/gen_waist_aug_anchors.py \
        --out-dir logs/reproduction/waist_aug_v1/anchors --workers 8

Re-running is safe: clips whose CSV already exists are skipped.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import tyro

from smp.data.shoulder_ik import ClipSpec, ShoulderIkGenerator, write_result
from smp.data.waist_aug_specs import build_specs, build_stand_specs

_GENERATOR: ShoulderIkGenerator | None = None


def generator() -> ShoulderIkGenerator:
  global _GENERATOR
  if _GENERATOR is None:
    _GENERATOR = ShoulderIkGenerator()
  return _GENERATOR


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def run_one(args: tuple[dict, str]) -> dict:
  """Worker: generate + validate one anchor. Runs in a subprocess."""
  payload, out_dir = args
  spec = ClipSpec(**payload)
  out = Path(out_dir)
  try:
    gen = generator()
    passed, motion, execution, metrics = gen.produce(spec)
    record = write_result(out, spec, passed, motion, execution, metrics, gen.sim_model)
    return {
      "clip_id": spec.clip_id,
      "family_id": spec.family_id,
      "condition_id": spec.condition_id,
      "passed": bool(passed),
      "failed_checks": metrics.get("failed_checks"),
      "yaw_amplitude_deg": float(metrics.get("actual_waist_amplitude_rad", 0.0) * 57.2957795),
      "entry_duration_s": spec.entry_duration_s,
      "csv": record.get("csv"),
    }
  except Exception as error:  # noqa: BLE001 - surface every failure in the manifest
    return {
      "clip_id": spec.clip_id,
      "family_id": spec.family_id,
      "condition_id": spec.condition_id,
      "passed": False,
      "exception": f"{type(error).__name__}: {error}",
      "traceback": traceback.format_exc(limit=3),
    }


def spec_payload(spec: ClipSpec) -> dict:
  return asdict(spec)


def main(
  out_dir: str = "logs/reproduction/waist_aug_v1/anchors",
  workers: int = 8,
  seed: int = 42,
  per_family: int = 200,
  limit: int = 0,
  include_stand: bool = True,
) -> None:
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  manifest_path = out / "manifest.json"

  plan = build_specs(seed=seed, per_family=per_family)
  specs: list[ClipSpec] = list(plan.specs)
  if include_stand:
    specs += build_stand_specs(seed=seed)
  if limit:
    specs = specs[:limit]

  done: dict[str, dict] = {}
  if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    done = {r["clip_id"]: r for r in existing.get("records", [])}

  pending = []
  for spec in specs:
    rec = done.get(spec.clip_id)
    if rec is not None and (rec.get("passed") or rec.get("exception")):
      continue
    pending.append(spec)

  print(json.dumps({
    "requested": len(specs), "already_done": len(specs) - len(pending),
    "pending": len(pending), "workers": workers, "out": str(out.resolve()),
  }), flush=True)

  if not pending:
    summarize(out, specs, done)
    return

  workers = max(1, min(workers, os.cpu_count() or 1))
  tasks = [(spec_payload(s), str(out)) for s in pending]
  with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(run_one, t): t[0]["clip_id"] for t in tasks}
    for i, future in enumerate(as_completed(futures), 1):
      rec = future.result()
      done[rec["clip_id"]] = rec
      if i % 20 == 0 or i == len(futures):
        ok = sum(1 for r in done.values() if r.get("passed"))
        print(f"[{i}/{len(futures)}] accepted={ok}", flush=True)
        manifest_path.write_text(json.dumps({
          "updated_at_utc": now(), "records": list(done.values()),
        }, indent=1))

  summarize(out, specs, done)


def summarize(out: Path, specs: list[ClipSpec], done: dict[str, dict]) -> None:
  records = [done[s.clip_id] for s in specs if s.clip_id in done]
  accepted = [r for r in records if r.get("passed")]
  by_cond: dict[str, int] = {}
  for r in accepted:
    key = str(r.get("condition_id"))
    by_cond[key] = by_cond.get(key, 0) + 1
  summary = {
    "finished_at_utc": now(),
    "requested": len(specs),
    "attempted": len(records),
    "accepted": len(accepted),
    "acceptance_rate": round(len(accepted) / max(1, len(records)), 3),
    "accepted_by_condition": by_cond,
    "records": records,
  }
  (out / "manifest.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))


if __name__ == "__main__":
  tyro.cli(main)