#!/usr/bin/env python3
"""Analyze image dimension distribution for train/test sets."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import cv2

from backend.config import get_config, require_existing_dir


def analyze_dimensions(directory: Path, limit: int | None = None) -> Counter:
    require_existing_dir(directory, "Image directory")
    files = sorted([f for f in os.listdir(directory) if f.endswith(".jpg")])
    if limit:
        files = files[:limit]

    dim_counter: Counter = Counter()
    for idx, fname in enumerate(files, start=1):
        img_path = directory / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        dim_counter[(w, h)] += 1
        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(files)}")

    total = sum(dim_counter.values())
    print(f"\nAnalyzing dimensions in {directory}")
    print(f"Total scanned: {total}")
    for (w, h), count in dim_counter.most_common():
        pct = (count / max(len(files), 1)) * 100
        print(f"  {w}x{h}: {count} images ({pct:.1f}%)")
    return dim_counter


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Analyze image dimensions")
    parser.add_argument("--directory", default=str(cfg.test_dir))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    analyze_dimensions(Path(args.directory).expanduser().resolve(), args.limit)


if __name__ == "__main__":
    main()
