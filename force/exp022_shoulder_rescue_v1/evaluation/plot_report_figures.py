"""Render three evidence figures for the EXP-022 collaborator report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})


def save(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def force_trace() -> None:
    with (HERE / "report_seed248_same_trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    t = np.array([float(row["relative_time_s"]) for row in rows])
    mask = (t >= -0.12) & (t <= 2.12)
    vectors = {}
    for prefix in ("true", "v1", "candidate"):
        vectors[prefix] = np.array([
            [float(row[f"{prefix}_f{axis}_n"]) for axis in "xyz"]
            for row in rows
        ])
    norms = {key: np.linalg.norm(value, axis=1) for key, value in vectors.items()}
    shoulder = np.array([int(row["shoulder_or_elbow_contact"]) for row in rows], dtype=bool)
    peak = int(np.argmax(norms["true"]))
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.fill_between(t[mask], 0, 115, where=shoulder[mask], step="mid",
                    color="#e6a642", alpha=0.18, label="实际接触到右肩/右肘")
    ax.plot(t[mask], norms["true"][mask], color="#1c3358", lw=2.4, label="真实三维合力")
    ax.plot(t[mask], norms["v1"][mask], color="#d66b3e", lw=1.8, label="V1 估计")
    ax.plot(t[mask], norms["candidate"][mask], color="#168774", lw=1.8,
            label="肩部补偿估计")
    ax.axhline(10, color="#707782", lw=1, ls="--")
    ax.axvline(t[peak], color="#63728a", lw=1, ls=":")
    ax.scatter([t[peak]] * 3,
               [norms[key][peak] for key in ("true", "v1", "candidate")],
               color=["#1c3358", "#d66b3e", "#168774"], s=30, zorder=5)
    ax.text(0.985, 0.965, "峰值：真实 94.8 N｜V1 0 N｜补偿 98.2 N",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color="#27354a", bbox={"facecolor": "white", "alpha": 0.84,
                                   "edgecolor": "none", "pad": 2})
    ax.set(xlim=(-0.12, 2.12), ylim=(0, 116),
           xlabel="首次物理接触后的时间（秒）", ylabel="力的模长（N）",
           title="图 1｜同一条保存轨迹：真实力、V1 与肩部补偿")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    fig.text(0.01, 0.005,
             "Seed 248；50 Hz。阴影表示实际发生右肩或右肘接触。三条线共用同一状态轨迹；此图不代表闭环卸力效果。",
             fontsize=8.5, color="#57616d")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, "report_01_same_trajectory_force")


def closed_loop() -> None:
    oracle = json.loads((HERE / "oracle_gain_fresh_result.json").read_text())
    est = json.loads((HERE / "estimator_fresh15_result.json").read_text())
    groups = [
        ("遮蔽接触输入", oracle, "masked", "#537c8e"),
        ("真值接触输入", oracle, "full_oracle", "#5b6480"),
        ("V1 估计输入", est, "v1_estimated", "#d66b3e"),
        ("肩部补偿输入", est, "candidate_estimated", "#168774"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for idx, (_label, source, arm, color) in enumerate(groups):
        vals = np.array([row["longest_torso_loaded_ge10n_s"]
                         for row in source["episodes"] if row["arm"] == arm])
        assert len(vals) == 8 and np.all(vals > 0)
        y = 3 - idx
        jitter = np.linspace(-0.13, 0.13, len(vals))
        ax.scatter(vals, np.full(len(vals), y) + jitter, color=color,
                   s=34, alpha=0.74, zorder=3)
        med = float(np.median(vals))
        ax.scatter(med, y, color="#172239", marker="D", s=58, zorder=4)
        fails = source["summary"][arm]["failure_episodes"]
        ax.text(12.5, y, f"中位 {med:.2f} s · 失败 {fails}/8",
                va="center", fontsize=9, color="#27354a")
    ax.axvline(1.0, color="#758494", ls="--", lw=1)
    ax.text(1.05, 3.42, "1 秒参考线", color="#667586", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.016, 90)
    ax.set_ylim(-0.5, 3.6)
    ax.set_yticks([3, 2, 1, 0], [g[0] for g in groups])
    ax.set_xlabel("躯干真实力连续 ≥10 N 的最长时间（秒；对数轴）")
    ax.set_title("图 2｜15 N 标称持续球头：四种策略输入的闭环结果")
    ax.grid(axis="x", alpha=0.22, which="both")
    fig.text(0.01, 0.005,
             "每点一条回合；四组使用相同 277–284 号种子，但各自闭环轨迹不同。黑色菱形为中位数。",
             fontsize=8.5, color="#57616d")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "report_02_sustained_closed_loop")


def actor_shift() -> None:
    normal = json.loads((HERE / "actor_normalizer_shift_audit.json").read_text())["groups"]
    ablation = json.loads((HERE / "point_normal_ablation_result.json").read_text())["summary"]
    chest = normal["original_short_pulse_chest"]
    shoulder = normal["full_candidate"]
    x = np.arange(2)
    old = [chest["point_y_abs_normalized"]["median"],
           chest["normal_y_abs_normalized"]["median"]]
    new = [shoulder["point_y_abs_normalized"]["median"],
           shoulder["normal_y_abs_normalized"]["median"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    bars1 = ax.bar(x - 0.18, old, 0.35, color="#61768b", label="原胸前短脉冲")
    bars2 = ax.bar(x + 0.18, new, 0.35, color="#c87842", label="25 N 完整肩部映射")
    ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(x, ["作用点横向", "法向横向"])
    ax.set_ylabel("距训练均值的绝对标准差（中位数）")
    ax.set_ylim(0, 108)
    ax.set_title("Actor 输入分布偏移")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.2)
    ax2 = axes[1]
    names = ["V1", "完整肩部映射", "仅改力", "仅改点", "仅改法向"]
    keys = ["v1", "full_candidate", "force_only", "point_only", "normal_only"]
    counts = [ablation[key]["failure_codes"].get("joint_hard_limit", 0) for key in keys]
    bars = ax2.bar(np.arange(5), counts, color=["#61768b", "#c87842", "#168774", "#8399a7", "#8399a7"])
    ax2.bar_label(bars, labels=[f"{count}/4" for count in counts], padding=3, fontsize=9)
    ax2.set_xticks(np.arange(5), names, rotation=24, ha="right")
    ax2.set_ylim(0, 4.8)
    ax2.set_yticks(range(5))
    ax2.set_ylabel("触发关节硬限位的回合数")
    ax2.set_title("25 N 诊断消融")
    ax2.grid(axis="y", alpha=0.2)
    fig.suptitle("图 3｜肩部接触映射进入原 Actor 的分布外区域", fontsize=13)
    fig.text(0.01, 0.005,
             "各消融组均为不同闭环轨迹且 4/4 回合失败；图示输入偏移与硬限位同时出现，不证明单通道因果。",
             fontsize=8.5, color="#57616d")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, "report_03_actor_input_shift")


if __name__ == "__main__":
    force_trace()
    closed_loop()
    actor_shift()
    print("Rendered three PNG and three PDF report figures")
