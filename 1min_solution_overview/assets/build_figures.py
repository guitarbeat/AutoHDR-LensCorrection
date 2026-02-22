#!/usr/bin/env python3
"""Build slide-ready figures for the 1-minute science overview."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from PIL import Image


REPO = Path("/Users/aaron/Desktop/AutoHDR")
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
        if stripped.startswith("| `2026-"):
            count += 1
    return count


def plot_score_progression() -> None:
    labels = ["v4 baseline", "oracle allbest", "top real tie"]
    scores = [30.14024, 30.79062, 31.63214]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(labels, scores, marker="o", linewidth=2.5, color="#146356")
    ax.set_title("Real-Lineage Score Progression", fontsize=14, weight="bold")
    ax.set_ylabel("Public Score")
    ax.set_ylim(29.8, 31.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for x, y in zip(labels, scores):
        ax.text(x, y + 0.03, f"{y:.5f}", ha="center", va="bottom", fontsize=10)
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

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars = ax.bar(labels, values, color=["#0b525b", "#1b9aaa", "#f4a259", "#e76f51", "#264653"])
    ax.set_title("Top Fusion Source Mix (Per-Image Selection)", fontsize=14, weight="bold")
    ax.set_ylabel("Image Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_top_source_mix.png", dpi=200)
    plt.close(fig)


def plot_real_vs_probe_counts() -> None:
    lines = LINEAGE_MD.read_text(encoding="utf-8").splitlines()
    real_count = count_lineage_rows(lines, "## Real (Select These)")
    probe_count = count_lineage_rows(lines, "## Probe / Non-Lineage (Do Not Select As Real)")

    labels = ["Real lineage rows", "Probe rows"]
    values = [real_count, probe_count]
    colors = ["#2a9d8f", "#e63946"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Lineage Classification Snapshot", fontsize=14, weight="bold")
    ax.set_ylabel("Submission rows")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_03_real_vs_probe_counts.png", dpi=200)
    plt.close(fig)


def plot_failsafe_patch() -> None:
    failsafe = json.loads(FAILSAFE_MANIFEST.read_text(encoding="utf-8"))
    replaced = int(failsafe.get("replaced_count", 0))
    total = 1000
    unchanged = total - replaced

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(["submission_v4_oracle_valid_allzip_failsafe8"], [unchanged], color="#2a9d8f", label="unchanged")
    ax.barh(
        ["submission_v4_oracle_valid_allzip_failsafe8"],
        [replaced],
        left=[unchanged],
        color="#e76f51",
        label="patched",
    )
    ax.set_title("Failsafe8 Patch Coverage", fontsize=14, weight="bold")
    ax.set_xlabel("Image count")
    ax.text(unchanged + replaced / 2, 0, f"patched={replaced}", ha="center", va="center", color="white", weight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_04_failsafe8_patch.png", dpi=200)
    plt.close(fig)


def _pick_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def build_before_after_example() -> None:
    failsafe = json.loads(FAILSAFE_MANIFEST.read_text(encoding="utf-8"))
    ids = [str(x) for x in failsafe.get("replaced_ids", [])]
    if not ids:
        return

    selected_id = ids[0]
    orig_path = TEST_ORIGINAL_DIR / f"{selected_id}.jpg"
    corrected_path = CORRECTED_DIR / f"{selected_id}.jpg"

    if not orig_path.exists() or not corrected_path.exists():
        fallback_orig = _pick_first_existing(TEST_ORIGINAL_DIR.glob("*.jpg"))
        fallback_corr = _pick_first_existing(CORRECTED_DIR.glob("*.jpg"))
        if fallback_orig is None or fallback_corr is None:
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].imshow(orig)
    axes[0].set_title("Original (test-originals)")
    axes[0].axis("off")

    axes[1].imshow(corr)
    axes[1].set_title("Corrected (corrected_v4)")
    axes[1].axis("off")

    fig.suptitle(f"Failsafe Example ID: {selected_id}", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_05_before_after_example.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dir(FIG_DIR)
    plot_score_progression()
    plot_source_mix()
    plot_real_vs_probe_counts()
    plot_failsafe_patch()
    build_before_after_example()
    print(f"Generated figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
