"""Summarize the fixed oracle/estimated EXP-022 paired rollout design."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


def longest(mask: np.ndarray) -> int:
  best = count = 0
  for value in mask:
    count = count + 1 if value else 0
    best = max(best, count)
  return best


def read_episode(root: Path, seed: int, mode: str) -> dict:
  folder = root / f"{seed}_{mode}"
  audit = json.loads((folder / "audit.json").read_text())
  evaluation = json.loads((folder / "evaluation.json").read_text())
  with (folder / "contact_trace.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  with np.load(folder / "shadow_inputs.npz", allow_pickle=False) as z:
    shadow = {key: z[key] for key in z.files}
  if len(rows) != audit["frames"] or len(rows) != len(shadow["x"]):
    raise ValueError(f"Row count mismatch: {folder}")
  if audit["seed"] != seed or audit["actor_contact_source"] != mode:
    raise ValueError(f"Condition mismatch: {folder}")
  if not np.array_equal(
    shadow["oracle_history"][:, -8:], shadow["oracle_contact_frame"]
  ):
    raise ValueError(f"Oracle current-frame mismatch: {folder}")
  if mode == "oracle" and not np.array_equal(
    shadow["actor_contact_history"], shadow["oracle_history"]
  ):
    raise ValueError(f"Oracle actor input mismatch: {folder}")
  force = np.array([float(row["force_resultant_n"]) for row in rows])
  estimate = np.linalg.norm(shadow["estimated_force_body_n"], axis=1)
  force_vector_error = np.linalg.norm(
    shadow["estimated_force_body_n"] - shadow["true_force_body_n"], axis=1
  )
  if np.max(np.abs(force - np.linalg.norm(shadow["true_force_body_n"], axis=1))) > 1e-3:
    raise ValueError(f"Force trace mismatch: {folder}")
  contact = np.array([bool(row["contact_geometries"]) for row in rows])
  first = int(np.flatnonzero(contact)[0]) if contact.any() else None
  pre_end = first if first is not None else len(force)
  loaded = force >= 10
  first_loaded = int(np.flatnonzero(loaded)[0]) if loaded.any() else None
  peak_idx = int(np.argmax(force))
  fail = [
    key.removeprefix("termination/")
    for key, value in audit["terminations"].items()
    if value and key != "termination/time_out"
  ]
  return {
    "audit": audit,
    "event_success": float(evaluation["metrics"]["event_success"]["mean"]),
    "unload_time_s": float(evaluation["metrics"]["unload_time_s"]["mean"]),
    "rows": rows,
    "shadow": shadow,
    "force": force,
    "estimate": estimate,
    "contact": contact,
    "first_contact": first,
    "loaded": loaded,
    "failure": ";".join(fail),
    "peak_n": float(force.max()),
    "contact_s": float(contact.sum() * 0.02),
    "loaded_s": float(loaded.sum() * 0.02),
    "longest_loaded_s": float(longest(loaded) * 0.02),
    "impulse_ns": float(force.sum() * 0.02),
    "first_contact_est_n": float(estimate[first]) if first is not None else None,
    "first_loaded_est_n": float(estimate[first_loaded])
    if first_loaded is not None
    else None,
    "first_loaded_miss": bool(estimate[first_loaded] < 10)
    if first_loaded is not None
    else None,
    "loaded_vector_mae_n": float(force_vector_error[loaded].mean())
    if loaded.any()
    else None,
    "true_peak_abs_magnitude_error_n": float(abs(force[peak_idx] - estimate[peak_idx]))
    if loaded.any()
    else None,
    "precontact_max_est_n": float(estimate[:pre_end].max()) if pre_end else None,
    "precontact_est_ge10_frames": int(np.count_nonzero(estimate[:pre_end] >= 10)),
    "precontact_actor_active_frames": int(
      np.count_nonzero(shadow["actor_contact_history"][:pre_end, -8] > 0.5)
    ),
    "first_contact_actor_active": bool(shadow["actor_contact_history"][first, -8] > 0.5)
    if first is not None
    else None,
  }


def main(root: Path = Path("logs/reproduction/exp022_closed_loop_v1")) -> None:
  design = json.loads((root / "design.json").read_text())
  seeds = range(*[design["seed_range"][0], design["seed_range"][1] + 1])
  pairs = {}
  summary = []
  for seed in seeds:
    oracle = read_episode(root, seed, "oracle")
    estimated = read_episode(root, seed, "estimated")
    original_pusher = design.get("pusher_mode") == "original"
    force = None if original_pusher else (10, 20, 30, 40)[(seed - 181) % 4]
    mass = 5 if original_pusher else (1, 5)[(seed - 181) // 4]
    for mode, episode in (("oracle", oracle), ("estimated", estimated)):
      audit = episode["audit"]
      if (
        audit["sphere_mass_kg"] != mass
        or (force is not None and audit["target_force_n"] != force)
        or (original_pusher and audit["pusher_mode"] != "original")
      ):
        raise ValueError(f"Unexpected pusher condition: {seed} {mode}")
      summary.append(
        {
          "seed": seed,
          "mode": mode,
          "target_force_n": force if force is not None else "original_protocol",
          "mass_kg": mass,
          "frames": audit["frames"],
          "first_contact_step": episode["first_contact"] + 1
          if episode["first_contact"] is not None
          else "",
          "first_contact_est_n": episode["first_contact_est_n"],
          "first_loaded_est_n": episode["first_loaded_est_n"],
          "first_loaded_miss": episode["first_loaded_miss"],
          "loaded_vector_mae_n": episode["loaded_vector_mae_n"],
          "true_peak_abs_magnitude_error_n": episode["true_peak_abs_magnitude_error_n"],
          "first_contact_actor_active": episode["first_contact_actor_active"],
          "precontact_max_est_n": episode["precontact_max_est_n"],
          "precontact_est_ge10_frames": episode["precontact_est_ge10_frames"],
          "precontact_actor_active_frames": episode["precontact_actor_active_frames"],
          "contact_s": episode["contact_s"],
          "loaded_s": episode["loaded_s"],
          "longest_loaded_s": episode["longest_loaded_s"],
          "peak_n": episode["peak_n"],
          "impulse_ns": episode["impulse_ns"],
          "failure": episode["failure"],
          "event_success": episode["event_success"],
          "unload_time_s": episode["unload_time_s"],
          "peak_gpu_mib": audit["peak_own_gpu_mib"],
        }
      )
    common_pre = min(
      oracle["first_contact"]
      if oracle["first_contact"] is not None
      else len(oracle["force"]),
      estimated["first_contact"]
      if estimated["first_contact"] is not None
      else len(estimated["force"]),
    )
    oa = oracle["shadow"]["action"][:common_pre]
    ea = estimated["shadow"]["action"][:common_pre]
    if oa.shape != ea.shape:
      raise ValueError(f"Paired precontact length mismatch: {seed}")
    pairs[seed] = {
      "oracle": oracle,
      "estimated": estimated,
      "precontact_action_max_abs": float(np.max(np.abs(oa - ea))) if len(oa) else None,
      "precontact_action_mean_l2": float(np.linalg.norm(oa - ea, axis=1).mean())
      if len(oa)
      else None,
    }
  with (root / "per_episode.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)

  pair_rows = []
  for seed, pair in pairs.items():
    o, e = pair["oracle"], pair["estimated"]
    pair_rows.append(
      {
        "seed": seed,
        "force_n": o["audit"]["target_force_n"]
        if not original_pusher
        else "original_protocol",
        "mass_kg": o["audit"]["sphere_mass_kg"],
        "first_contact_step_diff": e["first_contact"] - o["first_contact"]
        if e["first_contact"] is not None and o["first_contact"] is not None
        else "",
        "precontact_action_max_abs": pair["precontact_action_max_abs"],
        "precontact_action_mean_l2": pair["precontact_action_mean_l2"],
        "delta_contact_s": e["contact_s"] - o["contact_s"],
        "delta_longest_loaded_s": e["longest_loaded_s"] - o["longest_loaded_s"],
        "delta_peak_n": e["peak_n"] - o["peak_n"],
        "delta_impulse_ns": e["impulse_ns"] - o["impulse_ns"],
        "oracle_failure": o["failure"],
        "estimated_failure": e["failure"],
        "oracle_event_success": o["event_success"],
        "estimated_event_success": e["event_success"],
      }
    )
  with (root / "per_pair.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(pair_rows)

  fig, axes = plt.subplots(4, 2, figsize=(13, 12))
  for ax, seed in zip(axes.flat, seeds, strict=True):
    pair = pairs[seed]
    for mode, color in (("oracle", "#1768ac"), ("estimated", "#c24d2c")):
      episode = pair[mode]
      time = np.arange(1, len(episode["force"]) + 1) * 0.02
      ax.plot(time, episode["force"], color=color, lw=1.2, label=f"True force, {mode}")
      if mode == "estimated":
        ax.plot(
          time,
          episode["estimate"],
          color=color,
          lw=0.9,
          ls="--",
          alpha=0.75,
          label="Estimator",
        )
      if episode["first_contact"] is not None:
        ax.axvline(time[episode["first_contact"]], color=color, lw=0.7, ls=":")
    ax.axhline(10, color="0.5", lw=0.7, ls="--")
    if original_pusher:
      ax.set_title(f"{seed} | Original EXP-022 pusher, 5 kg ball")
    else:
      ax.set_title(
        f"{seed} | {pair['oracle']['audit']['target_force_n']:.0f} N command, "
        f"{pair['oracle']['audit']['sphere_mass_kg']:.0f} kg ball"
      )
    ax.set_ylabel("Resultant force (N)")
    ax.set_xlim(
      0, max(len(pair[mode]["force"]) for mode in ("oracle", "estimated")) * 0.02
    )
    ax.grid(alpha=0.18)
  for ax in axes[-1]:
    ax.set_xlabel("Time (s)")
  axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
  fig.tight_layout()
  fig.savefig(root / "paired_force_traces.png", dpi=190)
  fig.savefig(root / "paired_force_traces.pdf")
  plt.close(fig)

  if original_pusher and 181 in pairs:
    pair = pairs[181]
    o, e = pair["oracle"], pair["estimated"]
    n = min(len(o["force"]), len(e["force"]))
    time = np.arange(1, n + 1) * 0.02
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
      time, o["force"][:n], color="#1768ac", label="True force, oracle actor"
    )
    axes[0].plot(
      time, e["force"][:n], color="#c24d2c", label="True force, estimated actor"
    )
    axes[0].plot(
      time, e["estimate"][:n], color="#c24d2c", ls="--", label="Estimated force"
    )
    axes[0].set_ylabel("Force (N)")
    axes[0].legend(frameon=False)
    axes[1].step(
      time,
      o["shadow"]["actor_contact_history"][:n, -8],
      where="post",
      color="#1768ac",
      label="Oracle actor contact switch",
    )
    axes[1].step(
      time,
      e["shadow"]["actor_contact_history"][:n, -8],
      where="post",
      color="#c24d2c",
      label="Estimated actor contact switch",
    )
    axes[1].set_ylabel("Actor contact switch")
    axes[1].legend(frameon=False)
    action_diff = np.linalg.norm(
      o["shadow"]["action"][:n] - e["shadow"]["action"][:n], axis=1
    )
    axes[2].plot(time, action_diff, color="#624c8c")
    axes[2].set_ylabel("Action L2 difference")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
      if o["first_contact"] is not None:
        ax.axvline(
          (o["first_contact"] + 1) * 0.02,
          color="#1768ac",
          ls=":",
          label="Oracle first contact",
        )
      ax.set_xlim(0, min(5.0, time[-1]))
      ax.grid(alpha=0.18)
    fig.suptitle("Seed 181: precontact false force changes frozen actor behavior")
    fig.tight_layout()
    fig.savefig(root / "seed181_precontact_feedback.png", dpi=190)
    fig.savefig(root / "seed181_precontact_feedback.pdf")
    plt.close(fig)

  report = {
    "design": design,
    "pairs": len(pair_rows),
    "failure_oracle": sum(bool(p["oracle_failure"]) for p in pair_rows),
    "failure_estimated": sum(bool(p["estimated_failure"]) for p in pair_rows),
    "event_success_oracle": sum(p["oracle_event_success"] > 0.5 for p in pair_rows),
    "event_success_estimated": sum(
      p["estimated_event_success"] > 0.5 for p in pair_rows
    ),
    "no_contact_oracle": sum(
      p["oracle"]["first_contact"] is None for p in pairs.values()
    ),
    "no_contact_estimated": sum(
      p["estimated"]["first_contact"] is None for p in pairs.values()
    ),
    "estimated_longer_loaded": sum(p["delta_longest_loaded_s"] > 0 for p in pair_rows),
    "estimated_shorter_loaded": sum(p["delta_longest_loaded_s"] < 0 for p in pair_rows),
    "estimated_equal_loaded": sum(p["delta_longest_loaded_s"] == 0 for p in pair_rows),
    "max_precontact_action_abs": max(p["precontact_action_max_abs"] for p in pair_rows),
    "max_first_contact_step_diff": max(
      (
        abs(p["first_contact_step_diff"])
        for p in pair_rows
        if p["first_contact_step_diff"] != ""
      ),
      default=None,
    ),
    "max_gpu_mib": max(row["peak_gpu_mib"] for row in summary),
  }
  (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  tyro.cli(main)
