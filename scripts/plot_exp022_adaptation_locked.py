"""Plot two explicitly named locked episodes: worst run and delayed onset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--capture-dir", default="test_selected")
  parser.add_argument("--prediction-dir", default="test_frozen/predictions")
  parser.add_argument("--baseline-dir", default="test_v392_predictions")
  parser.add_argument("--metrics-file", default="test_frozen/metrics.json")
  parser.add_argument("--worst-episode", default="124_oracle")
  parser.add_argument("--delay-episode", default="129_masked")
  args = parser.parse_args()
  root = args.root
  metrics = json.loads((root / args.metrics_file).read_text())
  by_name = {row["episode"]: row for row in metrics["per_episode"]}
  cases = (
    (args.worst_episode, "Worst loaded run MAE"),
    (args.delay_episode, "Short contact; delayed first detection"),
  )
  fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4))
  for ax, (name, description) in zip(axes, cases, strict=True):
    with np.load(
      root / args.capture_dir / name / "shadow_inputs.npz", allow_pickle=False
    ) as z:
      true_n = np.linalg.norm(z["true_force_body_n"], axis=1)
      contact = z["contact_geometries"] != ""
      steps = z["step"]
    with np.load(root / args.prediction_dir / f"{name}.npz", allow_pickle=False) as z:
      predicted_n = np.linalg.norm(z["estimated_force_body_n"], axis=1)
    with np.load(root / args.baseline_dir / f"{name}.npz", allow_pickle=False) as z:
      old_n = np.linalg.norm(z["estimated_force_body_n"], axis=1)
    first = int(np.flatnonzero(contact)[0])
    t = (steps - steps[first]) * 0.02
    visible = (t >= -0.4) & (t <= 1.2)
    ax.plot(t[visible], true_n[visible], color="#202020", lw=2, label="True resultant")
    ax.plot(
      t[visible],
      predicted_n[visible],
      color="#1678b4",
      lw=1.7,
      label="EXP-022 adapted",
    )
    ax.plot(t[visible], old_n[visible], color="#d9792d", lw=1.4, label="Frozen V3.92")
    ax.axvline(0, color="#666666", ls="--", lw=1)
    ax.set_xlim(-0.4, 1.2)
    upper = max(
      25, min(160, 1.3 * max(true_n[visible].max(), predicted_n[visible].max()))
    )
    ax.set_ylim(0, upper)
    if old_n[visible].max() > upper:
      ax.text(
        0.98,
        0.91,
        f"V3.92 reaches {old_n[visible].max():.0f} N (clipped)",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color="#9e4e19",
      )
    ax.grid(alpha=0.25)
    ax.set_ylabel("Force norm (N)")
    row = by_name[name]
    ax.set_title(
      f"{name}: {description} | run MAE {row['loaded_vector_mae_n']:.1f} N, "
      f"first delay {row['first_loaded_detection_delay_frames']} frame(s)",
      loc="left",
      fontsize=10,
    )
  axes[0].legend(frameon=False, ncol=3, fontsize=9)
  axes[-1].set_xlabel("Time relative to first physical contact (s)")
  fig.tight_layout(pad=2.2, h_pad=2.5)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=200)
  fig.savefig(args.out.with_suffix(".pdf"))
  plt.close(fig)


if __name__ == "__main__":
  main()
