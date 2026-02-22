#!/usr/bin/env python3
"""Analyze scored submission CSV against test image dimensions and parent buckets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from backend.config import ensure_dir, get_config, require_existing_dir


def classify_parent(height: int, width: int) -> str:
    is_portrait = height > width
    short_edge = width if is_portrait else height

    if is_portrait:
        if short_edge == 1367:
            return "portrait_standard"
        return "portrait_cropped"

    if short_edge == 1367:
        return "standard"
    if short_edge in (1368, 1369, 1370, 1371):
        return "near_standard_tall"
    if short_edge in (1365, 1366):
        return "near_standard_short"
    if 1360 <= short_edge <= 1364:
        return "moderate_crop"
    return "heavy_crop"


def _stats(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "zeros": 0,
            "zero_rate": 0.0,
        }
    values = sorted(scores)
    n = len(values)
    mid = n // 2
    if n % 2 == 0:
        median = (values[mid - 1] + values[mid]) / 2.0
    else:
        median = values[mid]

    zeros = sum(1 for v in values if v == 0.0)
    return {
        "count": n,
        "mean": float(sum(values) / n),
        "median": float(median),
        "min": float(values[0]),
        "max": float(values[-1]),
        "zeros": int(zeros),
        "zero_rate": float((zeros / n) * 100.0),
    }


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Analyze scored submission by dimension and bucket")
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--test-dir", default=str(cfg.test_dir))
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    score_csv = Path(args.score_csv).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    if not score_csv.exists():
        raise FileNotFoundError(f"Scored CSV not found: {score_csv}")
    require_existing_dir(test_dir, "Test images directory")
    ensure_dir(out_json.parent)

    all_scores: list[float] = []
    by_dim: dict[str, list[float]] = defaultdict(list)
    by_bucket: dict[str, list[float]] = defaultdict(list)

    with score_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id") or row.get("id")
            score_raw = row.get("score")
            if not image_id or score_raw is None:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                continue

            img_path = test_dir / f"{image_id}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            dim_key = f"{w}x{h}"
            bucket = classify_parent(h, w)

            all_scores.append(score)
            by_dim[dim_key].append(score)
            by_bucket[bucket].append(score)

    summary = _stats(all_scores)

    by_dim_rows = []
    for dim_key, values in sorted(by_dim.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True):
        row = {"dimension": dim_key}
        row.update(_stats(values))
        by_dim_rows.append(row)

    by_bucket_rows = []
    for bucket, values in sorted(by_bucket.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True):
        row = {"bucket": bucket}
        row.update(_stats(values))
        by_bucket_rows.append(row)

    dim_blockers = [
        row
        for row in by_dim_rows
        if int(row["count"]) >= 20 and (float(row["mean"]) < 15.0 or float(row["zero_rate"]) > 30.0)
    ]

    blockers = {
        "overall_zero_rate_gt_12": float(summary["zero_rate"]) > 12.0,
        "dim_blockers_count": len(dim_blockers),
        "dim_blockers": dim_blockers,
    }

    out_payload: dict[str, Any] = {
        "score_csv": str(score_csv),
        "test_dir": str(test_dir),
        "summary": summary,
        "by_bucket": by_bucket_rows,
        "by_dimension": by_dim_rows,
        "blockers": blockers,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, sort_keys=True)

    print("=== Scored Submission Analysis ===")
    print(f"CSV: {score_csv}")
    print(f"Rows analyzed: {summary['count']}")
    print(f"Mean: {summary['mean']:.4f}")
    print(f"Median: {summary['median']:.4f}")
    print(f"Zero rate: {summary['zero_rate']:.2f}% ({summary['zeros']}/{summary['count']})")
    print(f"Dimension blockers (count>=20): {len(dim_blockers)}")
    print(f"JSON report: {out_json}")


if __name__ == "__main__":
    main()
