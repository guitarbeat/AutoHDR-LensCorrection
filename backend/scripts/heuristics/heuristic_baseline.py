#!/usr/bin/env python3
"""
Phase 1: Zero-ML Heuristic Baseline for Kaggle Automatic Lens Correction.

Strategy:
1. Grid-search Brown-Conrady coefficients (k1, k2) on a sample of training pairs.
2. Pick the (k1, k2) that minimizes MAE between undistorted original and generated (ground truth).
3. Apply those coefficients to all 1,000 test images.
4. Zip the corrected images for upload to bounty.autohdr.com.

Usage:
    python -m backend.scripts.heuristics.heuristic_baseline --phase search
    python -m backend.scripts.heuristics.heuristic_baseline --phase apply
    python -m backend.scripts.heuristics.heuristic_baseline --phase both
"""

import argparse
import cv2
import numpy as np
import os
import time
import zipfile
from pathlib import Path
from typing import Tuple

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.core.undistort_ops import undistort_via_maps

CFG = get_config()
DEFAULT_TRAIN_DIR = CFG.train_dir
DEFAULT_TEST_DIR = CFG.test_dir
DEFAULT_OUTPUT_DIR = CFG.output_root
NUM_SEARCH_PAIRS = 50  # Number of train pairs to evaluate during grid search
IMG_HEIGHT, IMG_WIDTH = 1367, 2048


def apply_undistortion(image: np.ndarray, k1: float, k2: float,
                       p1: float = 0.0, p2: float = 0.0, k3: float = 0.0) -> np.ndarray:
    """Apply Brown-Conrady undistortion using cached map+remap path."""
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    corrected, _ = undistort_via_maps(
        image,
        dist_coeffs=dist_coeffs,
        alpha=0.0,
        interpolation="linear",
        border_mode="constant",
    )
    return corrected


def calculate_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Absolute Error between two images."""
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))


def get_train_pairs(train_dir: str, n: int) -> list[Tuple[str, str]]:
    """Get n random (original, generated) file path pairs from training set."""
    all_files = sorted(os.listdir(train_dir))
    originals = [f for f in all_files if f.endswith("_original.jpg")]

    # Sample evenly across the dataset
    np.random.seed(42)
    indices = np.random.choice(len(originals), min(n, len(originals)), replace=False)

    pairs = []
    for idx in indices:
        orig_name = originals[idx]
        gen_name = orig_name.replace("_original.jpg", "_generated.jpg")
        orig_path = os.path.join(train_dir, orig_name)
        gen_path = os.path.join(train_dir, gen_name)
        if os.path.exists(gen_path):
            pairs.append((orig_path, gen_path))

    return pairs


def grid_search(pairs: list[Tuple[str, str]]) -> Tuple[float, float, float]:
    """
    Grid-search over k1, k2 to find coefficients that minimize MAE.
    Also tests identity (no correction) as baseline.
    """
    # Coarse grid first
    k1_values = np.arange(-0.5, 0.55, 0.05)
    k2_values = np.arange(-0.3, 0.35, 0.05)

    print(f"Grid search: {len(k1_values)} x {len(k2_values)} = {len(k1_values) * len(k2_values)} combinations")
    print(f"Evaluating on {len(pairs)} image pairs...")

    # Preload images (downsized for speed)
    scale = 0.25  # Process at 1/4 resolution for speed
    loaded_pairs = []
    for orig_path, gen_path in pairs:
        orig = cv2.imread(orig_path)
        gen = cv2.imread(gen_path)
        if orig is not None and gen is not None:
            orig_small = cv2.resize(orig, None, fx=scale, fy=scale)
            gen_small = cv2.resize(gen, None, fx=scale, fy=scale)
            loaded_pairs.append((orig_small, gen_small))

    print(f"Loaded {len(loaded_pairs)} pairs at {scale}x resolution")

    # Baseline: no correction (identity)
    baseline_mae = np.mean([calculate_mae(o, g) for o, g in loaded_pairs])
    print(f"\nBaseline (no correction): MAE = {baseline_mae:.4f}")

    best_k1, best_k2, best_mae = 0.0, 0.0, baseline_mae
    results = []

    total = len(k1_values) * len(k2_values)
    count = 0
    t0 = time.time()

    for k1 in k1_values:
        for k2 in k2_values:
            count += 1
            maes = []
            for orig, gen in loaded_pairs:
                corrected = apply_undistortion(orig, k1, k2)
                mae = calculate_mae(corrected, gen)
                maes.append(mae)
            mean_mae = np.mean(maes)
            results.append((k1, k2, mean_mae))

            if mean_mae < best_mae:
                best_mae = mean_mae
                best_k1, best_k2 = k1, k2

            if count % 50 == 0 or count == total:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"  [{count}/{total}] Best so far: k1={best_k1:.3f}, k2={best_k2:.3f}, MAE={best_mae:.4f} (ETA: {eta:.0f}s)")

    print(f"\n=== COARSE SEARCH COMPLETE ===")
    print(f"Best: k1={best_k1:.4f}, k2={best_k2:.4f}, MAE={best_mae:.4f}")
    print(f"Improvement over baseline: {baseline_mae - best_mae:.4f} ({(baseline_mae - best_mae)/baseline_mae*100:.1f}%)")

    # Fine grid around best
    print(f"\nRefining around k1={best_k1:.3f}, k2={best_k2:.3f}...")
    k1_fine = np.arange(best_k1 - 0.05, best_k1 + 0.055, 0.01)
    k2_fine = np.arange(best_k2 - 0.05, best_k2 + 0.055, 0.01)

    for k1 in k1_fine:
        for k2 in k2_fine:
            maes = []
            for orig, gen in loaded_pairs:
                corrected = apply_undistortion(orig, k1, k2)
                mae = calculate_mae(corrected, gen)
                maes.append(mae)
            mean_mae = np.mean(maes)
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_k1, best_k2 = k1, k2

    print(f"\n=== FINE SEARCH COMPLETE ===")
    print(f"Best: k1={best_k1:.4f}, k2={best_k2:.4f}, MAE={best_mae:.4f}")
    print(f"Total improvement over baseline: {baseline_mae - best_mae:.4f} ({(baseline_mae - best_mae)/baseline_mae*100:.1f}%)")

    return best_k1, best_k2, best_mae


def apply_to_test_set(k1: float, k2: float, test_dir: str, output_dir: str) -> str:
    """Apply undistortion to all test images and create submission zip."""
    output_path = ensure_dir(Path(output_dir))
    corrected_dir = ensure_dir(output_path / "corrected_heuristic")

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
    print(f"\nApplying k1={k1:.4f}, k2={k2:.4f} to {len(test_files)} test images...")

    t0 = time.time()
    for i, fname in enumerate(test_files):
        img = cv2.imread(os.path.join(test_dir, fname))
        if img is None:
            print(f"  WARNING: Could not read {fname}")
            continue

        corrected = apply_undistortion(img, k1, k2)

        # Ensure output is same size as input (competition likely requires this)
        if corrected.shape[:2] != img.shape[:2]:
            corrected = cv2.resize(corrected, (img.shape[1], img.shape[0]))

        # Save with high JPEG quality
        out_path = corrected_dir / fname
        cv2.imwrite(str(out_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(test_files) - i - 1)
            print(f"  [{i+1}/{len(test_files)}] ETA: {eta:.0f}s")

    # Create zip
    zip_path = output_path / "submission_heuristic.zip"
    print(f"\nCreating zip: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for fname in test_files:
            fpath = corrected_dir / fname
            if fpath.exists():
                zf.write(fpath, fname)  # Store with just filename, no directory prefix

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Zip created: {zip_path} ({zip_size_mb:.1f} MB)")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"\n>>> Upload {zip_path} to https://bounty.autohdr.com <<<")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Heuristic baseline for lens correction")
    parser.add_argument("--phase", choices=["search", "apply", "both"], default="both")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--k1", type=float, default=None, help="Preset k1 (skip search)")
    parser.add_argument("--k2", type=float, default=None, help="Preset k2 (skip search)")
    parser.add_argument("--num-pairs", type=int, default=NUM_SEARCH_PAIRS)
    args = parser.parse_args()

    train_dir = Path(args.train_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    if args.phase in ("search", "both"):
        require_existing_dir(train_dir, "Training images directory")
    if args.phase in ("apply", "both"):
        require_existing_dir(test_dir, "Test images directory")

    if args.phase in ("search", "both"):
        pairs = get_train_pairs(str(train_dir), args.num_pairs)
        best_k1, best_k2, best_mae = grid_search(pairs)

        # Save results
        with open(output_dir / "best_coefficients.txt", "w") as f:
            f.write(f"k1={best_k1}\nk2={best_k2}\nmae={best_mae}\n")
        print(f"\nCoefficients saved to {output_dir / 'best_coefficients.txt'}")
    else:
        best_k1 = args.k1 if args.k1 is not None else -0.1
        best_k2 = args.k2 if args.k2 is not None else 0.05

    if args.phase in ("apply", "both"):
        apply_to_test_set(best_k1, best_k2, str(test_dir), str(output_dir))


if __name__ == "__main__":
    main()
