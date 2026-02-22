#!/usr/bin/env python3
"""Analyze dimension-bucket heuristic failure patterns on training pairs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from backend.config import get_config, require_existing_dir
from backend.core.undistort_ops import undistort_via_maps


def apply_undistortion(image: np.ndarray, k1: float, k2: float) -> np.ndarray:
    h, w = image.shape[:2]
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)
    corrected, _ = undistort_via_maps(
        image,
        dist_coeffs=dist_coeffs,
        alpha=0.0,
        interpolation="linear",
        border_mode="constant",
    )
    if corrected.shape[:2] != (h, w):
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return corrected


def calculate_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))


def classify_image(h: int, w: int) -> str:
    if h > w:
        return "portrait"
    if h == 1367:
        return "standard"
    if 1365 <= h <= 1369:
        return "near_standard"
    return "nonstandard"


BUCKET_COEFFICIENTS = {
    "standard": (-0.170, 0.350),
    "near_standard": (-0.170, 0.350),
    "nonstandard": (-0.080, 0.150),
    "portrait": (-0.050, 0.050),
}


def analyze_dim_bucket_errors(train_dir: Path, limit: int = 2000) -> None:
    require_existing_dir(train_dir, "Training images directory")
    all_files = sorted(os.listdir(train_dir))
    originals = [f for f in all_files if f.endswith("_original.jpg")]

    np.random.seed(42)
    indices = np.random.choice(len(originals), min(limit, len(originals)), replace=False)
    scale = 0.25
    mae_by_dim: list[dict] = []

    for idx in indices:
        orig_name = originals[idx]
        gen_name = orig_name.replace("_original.jpg", "_generated.jpg")
        orig_path = train_dir / orig_name
        gen_path = train_dir / gen_name
        if not gen_path.exists():
            continue

        orig = cv2.imread(str(orig_path))
        gen = cv2.imread(str(gen_path))
        if orig is None or gen is None:
            continue

        h, w = orig.shape[:2]
        bucket = classify_image(h, w)
        k1, k2 = BUCKET_COEFFICIENTS[bucket]

        orig_small = cv2.resize(orig, None, fx=scale, fy=scale)
        gen_small = cv2.resize(gen, None, fx=scale, fy=scale)
        corrected = apply_undistortion(orig_small, k1, k2)
        mae = calculate_mae(corrected, gen_small)

        mae_by_dim.append({"dim": f"{w}x{h}", "bucket": bucket, "mae": mae})

    dim_stats: dict[str, dict] = {}
    for entry in mae_by_dim:
        dim = entry["dim"]
        if dim not in dim_stats:
            dim_stats[dim] = {"count": 0, "mae_sum": 0.0, "bucket": entry["bucket"], "bad_count": 0}
        dim_stats[dim]["count"] += 1
        dim_stats[dim]["mae_sum"] += entry["mae"]
        if entry["mae"] > 15.0:
            dim_stats[dim]["bad_count"] += 1

    print("\n--- Error Analysis by Dimension ---")
    sorted_dims = sorted(dim_stats.items(), key=lambda x: x[1]["mae_sum"] / x[1]["count"], reverse=True)
    for dim, stats in sorted_dims:
        avg_mae = stats["mae_sum"] / stats["count"]
        bad_pct = (stats["bad_count"] / stats["count"]) * 100
        print(
            f"{dim:>10} ({stats['bucket']:>13}) | "
            f"Count: {stats['count']:>4} | Avg MAE: {avg_mae:.2f} | Bad(>15): {bad_pct:.1f}%"
        )


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Diagnose dimension-bucket errors by dimensions")
    parser.add_argument("--train-dir", default=str(cfg.train_dir))
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    analyze_dim_bucket_errors(Path(args.train_dir).expanduser().resolve(), args.limit)


if __name__ == "__main__":
    main()
