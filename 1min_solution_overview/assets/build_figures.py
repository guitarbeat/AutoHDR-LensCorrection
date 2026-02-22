#!/usr/bin/env python3
"""Build slide-ready figures for the 1-minute science overview."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
FIG_DIR = REPO / "1min_solution_overview" / "assets" / "figures"

TOP_MANIFEST = (
    REPO
    / "backend"
    / "scripts"
    / "heuristics"
    / "submission_v4_oracle_valid_allzip_20260222_175058_manifest.json"
)
FAILSAFE_MANIFEST = (
    REPO
    / "backend"
    / "outputs"
    / "kaggle"
    / "submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_manifest.json"
)
LINEAGE_MD = REPO / "docs" / "ops" / "real_submission_lineage.md"

CORRECTED_DIR = REPO / "backend" / "outputs" / "corrected_v4"
TEST_ORIGINAL_DIR = Path("/Volumes/Love SSD/test-originals")


def setup_plot_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d6dce5",
            "axes.labelcolor": "#243447",
            "text.color": "#243447",
            "xtick.color": "#3a4b5c",
            "ytick.color": "#3a4b5c",
            "axes.titleweight": "bold",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
        }
    )


def short_source_name(full_name: str) -> str:
    mapping = {
        "submission_v4_fallback_cycle2_t0_20260222_080059_scored.csv": "cycle2_t0",
        "submission_calibguard_cycle2_aggressive_20260222_135105_scored.csv": "cycle2_aggr",
        "submission_calibguard_cycle1_safe_scored.csv": "cycle1_safe",
        "submission_v4_mix_zle0_mix_batch_20260222_1_scored.csv": "mix_zle0",
        "submission_v4_fallback_learned_pos30_20260222_082633_scored.csv": "learned_pos30",
    }
    return mapping.get(full_name, full_name.replace("_scored.csv", ""))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_lineage_rows(lines: list[str], heading: str) -> int:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i + 1
            break
    if start is None:
        return 0
    count = 0
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## ") and stripped != heading:
            break
        # Count markdown table rows that begin with a backticked timestamp.
        if re.match(r"^\|\s*`\d{4}-\d{2}-\d{2}\s", stripped):
            count += 1
    return count


def plot_score_progression() -> None:
    labels = ["Baseline", "Uplift", "Top Real Tie"]
    scores = [30.14024, 30.79062, 31.63214]
    colors = ["#8aa1b8", "#2a9d8f", "#0f766e"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, scores, color=colors, width=0.62)
    ax.set_title("Real-Lineage Progression (Public Kaggle Score)")
    ax.set_ylabel("Public Score")
    ax.set_ylim(29.9, 31.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.025,
            f"{score:.5f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    delta_1 = scores[1] - scores[0]
    delta_2 = scores[2] - scores[1]
    ax.text(0.5, 31.78, f"+{delta_1:.5f}", ha="center", va="center", color="#2a9d8f", weight="bold")
    ax.text(1.5, 31.78, f"+{delta_2:.5f}", ha="center", va="center", color="#0f766e", weight="bold")

    fig.text(
        0.5,
        0.01,
        "Scores shown are ZIP-backed lineage rows only.",
        ha="center",
        fontsize=10,
        color="#5a6b7b",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_score_progression.png", dpi=200)
    plt.close(fig)


def plot_source_mix() -> None:
    top = json.loads(TOP_MANIFEST.read_text(encoding="utf-8"))
    source_counts = top["source_counts"]

    labels: list[str] = []
    values: list[int] = []
    for source, count in source_counts.items():
        labels.append(short_source_name(source))
        values.append(int(count))

    pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels = [name for name, _ in pairs]
    values = [count for _, count in pairs]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(labels, values, color=["#0f766e", "#2a9d8f", "#3fbf9b", "#8fd1bd", "#c9e9df"])
    ax.invert_yaxis()
    ax.set_title("Top Fusion Source Mix (Per-Image Selection)")
    ax.set_ylabel("Image Count")
    ax.set_xlabel("Selected images (out of 1000)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        pct = value / 1000.0 * 100.0
        ax.text(
            value + 8,
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=10,
        )
    ax.set_xlim(0, max(values) + 90)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_top_source_mix.png", dpi=200)
    plt.close(fig)


def plot_real_vs_probe_counts() -> None:
    lines = LINEAGE_MD.read_text(encoding="utf-8").splitlines()
    real_count = count_lineage_rows(lines, "## Real (Select These)")
    probe_count = count_lineage_rows(lines, "## Probe / Non-Lineage (Do Not Select As Real)")

    labels = ["Real lineage", "Probe / non-lineage"]
    values = [real_count, probe_count]
    colors = ["#0f766e", "#d97706"]

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p / 100 * sum(values)))})",
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_color("#22313f")
        t.set_weight("bold")

    ax.set_title("Lineage Classification Snapshot")
    ax.text(
        0,
        0,
        "Use only\nreal lineage",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="#22313f",
    )
    fig.text(
        0.5,
        0.02,
        "Counts are submission rows from real_submission_lineage.md",
        ha="center",
        fontsize=10,
        color="#5a6b7b",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_03_real_vs_probe_counts.png", dpi=200)
    plt.close(fig)


def plot_failsafe_patch() -> None:
    failsafe = json.loads(FAILSAFE_MANIFEST.read_text(encoding="utf-8"))
    replaced = int(failsafe.get("replaced_count", 0))
    total = 1000
    unchanged = total - replaced

    fig, ax = plt.subplots(figsize=(9, 6))
    values = [replaced, unchanged]
    colors = ["#d62828", "#2a9d8f"]
    labels = [f"Patched ({replaced})", f"Unchanged ({unchanged})"]

    ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    ax.set_title("Failsafe8 Patch Coverage")
    ax.text(0, 0, f"{replaced / total * 100:.1f}%\npatched", ha="center", va="center", fontsize=12, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Targeted replacement from test originals for high-risk IDs.",
        ha="center",
        fontsize=10,
        color="#5a6b7b",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_04_failsafe8_patch.png", dpi=200)
    plt.close(fig)


def _pick_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def build_before_after_example(test_original_dir: Path) -> None:
    failsafe = json.loads(FAILSAFE_MANIFEST.read_text(encoding="utf-8"))
    ids = [str(x) for x in failsafe.get("replaced_ids", [])]
    if not ids:
        return

    selected_id = ids[0]
    orig_path = test_original_dir / f"{selected_id}.jpg"
    corrected_path = CORRECTED_DIR / f"{selected_id}.jpg"

    if not orig_path.exists() or not corrected_path.exists():
        fallback_orig = _pick_first_existing(test_original_dir.glob("*.jpg"))
        fallback_corr = _pick_first_existing(CORRECTED_DIR.glob("*.jpg"))
        if fallback_orig is None or fallback_corr is None:
            print(
                "Skipping fig_05_before_after_example.png: "
                f"missing source images in {test_original_dir} and/or {CORRECTED_DIR}"
            )
            return
        orig_path = fallback_orig
        corrected_path = fallback_corr
        selected_id = corrected_path.stem

    orig = Image.open(orig_path).convert("RGB")
    corr = Image.open(corrected_path).convert("RGB")

    # Match sizes for side-by-side display.
    h = min(orig.height, corr.height)
    ow = int(orig.width * (h / orig.height))
    cw = int(corr.width * (h / corr.height))
    orig = orig.resize((ow, h))
    corr = corr.resize((cw, h))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(orig)
    axes[0].set_title("Original (test-originals)", fontsize=12, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(corr)
    axes[1].set_title("Corrected (corrected_v4)", fontsize=12, weight="bold")
    axes[1].axis("off")

    fig.suptitle(f"Before vs After Correction (Example ID: {selected_id})", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_05_before_after_example.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1-minute overview figures.")
    parser.add_argument(
        "--test-originals-dir",
        default=str(TEST_ORIGINAL_DIR),
        help="Directory containing original test JPGs for before/after figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_originals_dir = Path(args.test_originals_dir).expanduser().resolve()

    setup_plot_theme()
    ensure_dir(FIG_DIR)
    plot_score_progression()
    plot_source_mix()
    plot_real_vs_probe_counts()
    plot_failsafe_patch()
    build_before_after_example(test_originals_dir)
    print(f"Generated figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
