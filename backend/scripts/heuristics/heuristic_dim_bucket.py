#!/usr/bin/env python3
"""
Dimension-Bucket Micro-Grid Heuristic.

Canonical module entrypoint: `backend.scripts.heuristics.heuristic_dim_bucket`.

KEY INSIGHT from zeros analysis:
- Portrait images (h>w, i.e. 2048xN) → 100% zero scores with standard correction
- Non-standard heights (1352-1363) → 46-100% zero rates
- Standard landscape (1367x2048) → only 8.8% zeros, mean=28.6

Strategy:
1. For PORTRAIT images: skip correction or use very mild coefficients
   (the camera matrix assumes landscape; applying it to portrait DESTROYS the image)
2. For NON-STANDARD dimensions: use milder coefficients
3. For STANDARD 1367x2048: use stronger correction tuned for this bucket
4. Run per-bucket micro-grid search to refine all coefficients

This is NOT a per-image proxy (which failed 3x). It's a simple dimension-based
classifier with pre-computed coefficients per bucket.

Usage:
    python -m backend.scripts.heuristics.heuristic_dim_bucket
    python -m backend.scripts.heuristics.heuristic_dim_bucket --search
"""

import argparse
import cv2
import numpy as np
import os
import time
import zipfile
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.core.undistort_ops import undistort_via_maps


CFG = get_config()
DEFAULT_TRAIN_DIR = CFG.train_dir
DEFAULT_TEST_DIR = CFG.test_dir
DEFAULT_OUTPUT_DIR = CFG.output_root


def apply_undistortion(image: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Apply Brown-Conrady undistortion with alpha=0."""
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)

    corrected, _ = undistort_via_maps(
        image,
        dist_coeffs=dist_coeffs,
        alpha=0.0,
        interpolation="linear",
        border_mode="constant",
    )
    h, w = image.shape[:2]
    if corrected.shape[:2] != (h, w):
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return corrected


def calculate_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))


def classify_image(h: int, w: int) -> str:
    """Classify image into correction bucket based on dimensions and optical center preservation."""
    is_portrait = h > w
    short_edge = w if is_portrait else h

    if is_portrait:
        if short_edge == 1367:
            return "portrait_standard"
        return "portrait_cropped"

    if short_edge == 1367:
        return "standard"
    elif short_edge in (1368, 1369, 1370, 1371):
        return "near_standard_tall"
    elif short_edge in (1365, 1366):
        return "near_standard_short"
    elif 1360 <= short_edge <= 1364:
        return "moderate_crop"
    else:
        # Heavily cropped images (< 1360) have unmoored optical centers.
        # Global warping around w/2, h/2 becomes destructive.
        return "heavy_crop"


# Coefficients per bucket (refined via micro-grid search)
BUCKET_COEFFICIENTS = {
    # Tuned via 7-bucket micro-grid search (2026-02-22, --search-pairs 500)
    "standard": (-0.178, 0.368),
    "near_standard_tall": (-0.180, 0.378),
    "near_standard_short": (-0.180, 0.378),
    "moderate_crop": (-0.095, 0.195),
    "heavy_crop": (-0.006, 0.028),
    "portrait_standard": (-0.015, 0.045),
    "portrait_cropped": (0.008, 0.008),
}
RISKY_BUCKETS = {"heavy_crop", "portrait_cropped"}


def parse_bucket_set(raw: str, allowed: set[str]) -> set[str]:
    if not raw.strip():
        return set()
    values = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(values - allowed)
    if unknown:
        raise ValueError(f"Unknown buckets in list: {', '.join(unknown)}")
    return values


def parse_bucket_overrides(
    raw_overrides: list[str], allowed: set[str]
) -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    for item in raw_overrides:
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --override-bucket '{item}'. Expected format: bucket:k1:k2"
            )
        bucket, k1_str, k2_str = parts
        if bucket not in allowed:
            raise ValueError(f"Unknown bucket in --override-bucket: {bucket}")
        try:
            overrides[bucket] = (float(k1_str), float(k2_str))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric values in --override-bucket '{item}'"
            ) from exc
    return overrides


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_training_pairs(train_dir: str) -> list[tuple[str, str]]:
    all_files = sorted(os.listdir(train_dir))
    originals = [f for f in all_files if f.endswith("_original.jpg")]
    pairs: list[tuple[str, str]] = []
    for orig_name in originals:
        gen_name = orig_name.replace("_original.jpg", "_generated.jpg")
        orig_path = os.path.join(train_dir, orig_name)
        gen_path = os.path.join(train_dir, gen_name)
        if os.path.exists(gen_path):
            pairs.append((orig_path, gen_path))
    return pairs


def run_grid_search_on_pairs(
    loaded_pairs: list[tuple[np.ndarray, np.ndarray]],
    k1_values: np.ndarray,
    k2_values: np.ndarray,
) -> tuple[float, float, float, float]:
    baseline_mae = float(np.mean([calculate_mae(o, g) for o, g in loaded_pairs]))
    best_k1, best_k2, best_mae = 0.0, 0.0, baseline_mae

    for k1 in k1_values:
        for k2 in k2_values:
            maes = []
            for orig, gen in loaded_pairs:
                corrected = apply_undistortion(orig, float(k1), float(k2))
                maes.append(calculate_mae(corrected, gen))
            mean_mae = float(np.mean(maes))
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_k1, best_k2 = float(k1), float(k2)

    return best_k1, best_k2, best_mae, baseline_mae


def search_exact_dimension_coefficients(
    train_dir: str,
    min_pairs: int = 40,
    search_pairs: int = 400,
    max_dimensions: int | None = None,
) -> dict[str, dict]:
    pair_paths = list_training_pairs(train_dir)
    pairs_by_dim: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)

    print("Indexing train pairs by exact dimensions...")
    for orig_path, gen_path in pair_paths:
        img = cv2.imread(orig_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        pairs_by_dim[(h, w)].append((orig_path, gen_path))

    eligible = [
        (dim, paths) for dim, paths in pairs_by_dim.items() if len(paths) >= min_pairs
    ]
    eligible.sort(key=lambda item: len(item[1]), reverse=True)
    if max_dimensions is not None:
        eligible = eligible[: max(max_dimensions, 0)]

    rng = np.random.default_rng(42)
    report: dict[str, dict] = {}
    print(
        f"Running exact-dimension search on {len(eligible)} groups "
        f"(min_pairs={min_pairs}, search_pairs={search_pairs})..."
    )

    for i, ((h, w), paths) in enumerate(eligible, start=1):
        dim_name = f"{w}x{h}"
        bucket = classify_image(h, w)
        base_k1, base_k2 = BUCKET_COEFFICIENTS[bucket]

        sample_size = min(len(paths), search_pairs)
        sample_indices = rng.choice(len(paths), size=sample_size, replace=False)

        loaded_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for idx in sample_indices:
            orig_path, gen_path = paths[int(idx)]
            orig = cv2.imread(orig_path)
            gen = cv2.imread(gen_path)
            if orig is None or gen is None:
                continue
            orig_small = cv2.resize(orig, None, fx=0.25, fy=0.25)
            gen_small = cv2.resize(gen, None, fx=0.25, fy=0.25)
            loaded_pairs.append((orig_small, gen_small))

        if not loaded_pairs:
            continue

        k1_values = np.arange(base_k1 - 0.04, base_k1 + 0.041, 0.004)
        k2_values = np.arange(base_k2 - 0.08, base_k2 + 0.081, 0.004)
        best_k1, best_k2, best_mae, baseline_mae = run_grid_search_on_pairs(
            loaded_pairs,
            k1_values,
            k2_values,
        )

        report[dim_name] = {
            "k1": round(best_k1, 6),
            "k2": round(best_k2, 6),
            "bucket": bucket,
            "train_pairs_total": len(paths),
            "train_pairs_used": len(loaded_pairs),
            "baseline_mae": round(baseline_mae, 6),
            "best_mae": round(best_mae, 6),
        }
        print(
            f"  [{i}/{len(eligible)}] {dim_name}: "
            f"k1={best_k1:.4f}, k2={best_k2:.4f}, "
            f"MAE={best_mae:.4f} (baseline={baseline_mae:.4f}, n={len(loaded_pairs)})"
        )

    return report


def load_dimension_coefficients(
    path: Path,
) -> dict[tuple[int, int], tuple[float, float]]:
    raw = json.loads(path.read_text())
    output: dict[tuple[int, int], tuple[float, float]] = {}
    for dim_name, entry in raw.items():
        if "x" not in dim_name:
            continue
        w_s, h_s = dim_name.split("x", 1)
        try:
            w = int(w_s)
            h = int(h_s)
            k1 = float(entry["k1"])
            k2 = float(entry["k2"])
        except (KeyError, TypeError, ValueError):
            continue
        output[(h, w)] = (k1, k2)
    return output


def fine_grid_search(bucket: str, train_dir: str, num_pairs: int = 200):
    """Run a fine grid search for a specific dimension bucket."""
    all_files = sorted(os.listdir(train_dir))
    originals = [f for f in all_files if f.endswith("_original.jpg")]

    np.random.seed(42)
    indices = np.random.choice(
        len(originals), min(num_pairs, len(originals)), replace=False
    )

    # Filter to images of the right dimension bucket
    scale = 0.25
    loaded_pairs = []
    for idx in indices:
        orig_name = originals[idx]
        gen_name = orig_name.replace("_original.jpg", "_generated.jpg")
        orig_path = os.path.join(train_dir, orig_name)
        gen_path = os.path.join(train_dir, gen_name)
        if not os.path.exists(gen_path):
            continue

        orig = cv2.imread(orig_path)
        gen = cv2.imread(gen_path)
        if orig is None or gen is None:
            continue

        h, w = orig.shape[:2]
        img_bucket = classify_image(h, w)

        if img_bucket == bucket:
            orig_small = cv2.resize(orig, None, fx=scale, fy=scale)
            gen_small = cv2.resize(gen, None, fx=scale, fy=scale)
            loaded_pairs.append((orig_small, gen_small))

    if not loaded_pairs:
        print(f"  No training pairs found for bucket '{bucket}', using defaults")
        return BUCKET_COEFFICIENTS[bucket]

    print(f"  Grid search for '{bucket}': {len(loaded_pairs)} pairs")

    # Define search ranges based on bucket
    if bucket == "standard":
        k1_values = np.arange(-0.190, -0.150, 0.002)
        k2_values = np.arange(0.330, 0.370, 0.002)
    elif bucket in ("near_standard_tall", "near_standard_short"):
        k1_values = np.arange(-0.190, -0.150, 0.002)
        k2_values = np.arange(0.320, 0.380, 0.002)
    elif bucket == "moderate_crop":
        k1_values = np.arange(-0.120, -0.040, 0.005)
        k2_values = np.arange(0.100, 0.200, 0.005)
    elif bucket == "heavy_crop":
        # Massive shift requires finding exact near-zero balance point
        k1_values = np.arange(-0.030, 0.030, 0.002)
        k2_values = np.arange(-0.030, 0.030, 0.002)
    elif bucket == "portrait_standard":
        k1_values = np.arange(-0.050, 0.010, 0.005)
        k2_values = np.arange(0.000, 0.050, 0.005)
    else:  # portrait_cropped
        k1_values = np.arange(-0.010, 0.010, 0.002)
        k2_values = np.arange(-0.010, 0.010, 0.002)

    best_k1, best_k2, best_mae, baseline_mae = run_grid_search_on_pairs(
        loaded_pairs,
        k1_values,
        k2_values,
    )

    print(
        f"    Best: k1={best_k1:.3f}, k2={best_k2:.3f}, MAE={best_mae:.4f} (baseline={baseline_mae:.4f})"
    )
    return best_k1, best_k2


def main():
    parser = argparse.ArgumentParser(
        description="Dimension-bucket micro-grid heuristic"
    )
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--search", action="store_true", help="Run fine grid search per bucket"
    )
    parser.add_argument(
        "--search-pairs", type=int, default=500, help="Pairs for grid search"
    )
    parser.add_argument(
        "--search-dimensions",
        action="store_true",
        help="Run exact-dimension coefficient search on training pairs",
    )
    parser.add_argument(
        "--dimension-coeffs-path",
        default=None,
        help="Path to dimension coefficients JSON (read for inference, write when --search-dimensions)",
    )
    parser.add_argument(
        "--min-dim-pairs",
        type=int,
        default=40,
        help="Minimum pairs per dimension group",
    )
    parser.add_argument(
        "--dim-search-pairs",
        type=int,
        default=400,
        help="Pairs per dimension for search",
    )
    parser.add_argument(
        "--max-dimensions",
        type=int,
        default=None,
        help="Optional cap on searched dimensions",
    )
    parser.add_argument(
        "--identity-buckets",
        default="",
        help="Comma-separated bucket list to bypass correction and write original image",
    )
    parser.add_argument(
        "--override-bucket",
        action="append",
        default=[],
        help="Override a bucket coefficient as bucket:k1:k2 (repeatable)",
    )
    parser.add_argument(
        "--artifact-tag",
        default="",
        help="Optional tag for unique output folder/zip names (recommended for rapid iteration)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--safe-fallback",
        action="store_true",
        help="Force no-correction for risky buckets when no exact-dimension coeff is available",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    require_existing_dir(train_dir, "Training images directory")
    require_existing_dir(test_dir, "Test images directory")

    print("=== Heuristic: Dimension-Bucket Micro-Grid ===")

    allowed_buckets = set(BUCKET_COEFFICIENTS.keys())
    try:
        identity_buckets = parse_bucket_set(args.identity_buckets, allowed_buckets)
        bucket_overrides = parse_bucket_overrides(args.override_bucket, allowed_buckets)
    except ValueError as exc:
        raise SystemExit(str(exc))

    bucket_coefficients = dict(BUCKET_COEFFICIENTS)

    if bucket_overrides:
        for bucket, coeffs in bucket_overrides.items():
            bucket_coefficients[bucket] = coeffs

    # Optionally run grid search
    if args.search:
        print("\nRunning fine grid search per bucket...")
        for bucket in bucket_coefficients:
            k1, k2 = fine_grid_search(bucket, str(train_dir), args.search_pairs)
            bucket_coefficients[bucket] = (k1, k2)
        print()

    print("Bucket coefficients:")
    for bucket, (k1, k2) in bucket_coefficients.items():
        print(f"  {bucket:>15}: k1={k1:.3f}, k2={k2:.3f}")
    if identity_buckets:
        print(f"Identity fallback buckets: {', '.join(sorted(identity_buckets))}")

    dimension_coefficients: dict[tuple[int, int], tuple[float, float]] = {}
    dimension_coeff_path: Path | None = None
    if args.search_dimensions:
        dim_report = search_exact_dimension_coefficients(
            str(train_dir),
            min_pairs=args.min_dim_pairs,
            search_pairs=args.dim_search_pairs,
            max_dimensions=args.max_dimensions,
        )
        coeff_path = (
            Path(args.dimension_coeffs_path).expanduser().resolve()
            if args.dimension_coeffs_path
            else output_dir / "dimension_coefficients.json"
        )
        ensure_dir(coeff_path.parent)
        coeff_path.write_text(json.dumps(dim_report, indent=2))
        dimension_coefficients = load_dimension_coefficients(coeff_path)
        dimension_coeff_path = coeff_path
        print(
            f"Saved dimension coefficients: {coeff_path} ({len(dimension_coefficients)} dimensions)"
        )
    elif args.dimension_coeffs_path:
        coeff_path = Path(args.dimension_coeffs_path).expanduser().resolve()
        if coeff_path.exists():
            dimension_coefficients = load_dimension_coefficients(coeff_path)
            dimension_coeff_path = coeff_path
            print(
                f"Loaded dimension coefficients: {coeff_path} ({len(dimension_coefficients)} dimensions)"
            )
        else:
            print(f"WARNING: dimension coefficients file not found: {coeff_path}")

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
    if args.limit:
        test_files = test_files[: args.limit]

    print(f"\nTest images: {len(test_files)}")

    artifact_tag = args.artifact_tag.strip()
    if artifact_tag:
        corrected_dir_name = f"corrected_dim_bucket_{artifact_tag}"
        zip_name = f"submission_dim_bucket_microgrid_{artifact_tag}.zip"
        manifest_name = f"submission_dim_bucket_microgrid_{artifact_tag}.json"
    else:
        corrected_dir_name = "corrected_dim_bucket"
        zip_name = "submission_dim_bucket_microgrid.zip"
        manifest_name = "submission_dim_bucket_microgrid_manifest.json"

    corrected_dir = ensure_dir(output_dir / corrected_dir_name)

    bucket_counts = Counter()
    identity_counts = Counter()
    source_counts = Counter()

    t0 = time.time()
    for i, fname in enumerate(test_files):
        img = cv2.imread(str(test_dir / fname))
        if img is None:
            print(f"  WARNING: Could not read {fname}")
            continue

        h, w = img.shape[:2]
        bucket = classify_image(h, w)
        coeff_source = "bucket"
        if (h, w) in dimension_coefficients:
            k1, k2 = dimension_coefficients[(h, w)]
            coeff_source = "dimension"
        else:
            k1, k2 = bucket_coefficients[bucket]
            if args.safe_fallback and bucket in RISKY_BUCKETS:
                k1, k2 = 0.0, 0.0
                coeff_source = "safe_fallback"
        bucket_counts[bucket] += 1
        source_counts[coeff_source] += 1

        if bucket in identity_buckets:
            corrected = img
            identity_counts[bucket] += 1
        else:
            corrected = apply_undistortion(img, k1, k2)

        out_path = corrected_dir / fname
        cv2.imwrite(
            str(out_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        )

        if (i + 1) % 100 == 0 or (i + 1) == len(test_files):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(test_files) - i - 1)
            print(
                f"  [{i+1}/{len(test_files)}] ETA: {eta:.0f}s | "
                f"{bucket}/{coeff_source} (k1={k1:.3f}, k2={k2:.3f})"
            )

    print("\n=== RESULTS ===")
    print(f"Total time: {time.time() - t0:.1f}s")
    print("\nBucket distribution:")
    for bucket, count in bucket_counts.most_common():
        k1, k2 = bucket_coefficients[bucket]
        pct = count / len(test_files) * 100
        print(f"  {bucket:>15}: {count} ({pct:.1f}%) — k1={k1:.3f}, k2={k2:.3f}")
    print("\nCoefficient source distribution:")
    for source, count in source_counts.most_common():
        pct = count / len(test_files) * 100
        print(f"  {source:>15}: {count} ({pct:.1f}%)")
    if identity_counts:
        print("\nIdentity fallback usage:")
        for bucket, count in identity_counts.items():
            print(f"  {bucket:>15}: {count}")

    # Create zip
    zip_path = output_dir / zip_name
    print(f"\nCreating zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fname in test_files:
            fpath = corrected_dir / fname
            if fpath.exists():
                zf.write(fpath, fname)

    manifest_path = output_dir / manifest_name
    manifest = {
        "timestamp_local": datetime.now().isoformat(),
        "artifact_tag": artifact_tag if artifact_tag else None,
        "test_count": len(test_files),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "corrected_dir": str(corrected_dir),
        "zip_path": str(zip_path),
        "jpeg_quality": args.jpeg_quality,
        "identity_buckets": sorted(identity_buckets),
        "safe_fallback": bool(args.safe_fallback),
        "dimension_coefficients_path": str(dimension_coeff_path)
        if dimension_coeff_path
        else None,
        "dimension_coefficients_count": len(dimension_coefficients),
        "coefficient_source_counts": dict(source_counts),
        "bucket_coefficients": {
            bucket: {"k1": float(k1), "k2": float(k2)}
            for bucket, (k1, k2) in bucket_coefficients.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Zip created: {zip_path} ({zip_size_mb:.1f} MB)")
    print(f"Manifest saved: {manifest_path}")
    print(f"\n>>> Upload {zip_path} to https://bounty.autohdr.com <<<")


if __name__ == "__main__":
    main()
