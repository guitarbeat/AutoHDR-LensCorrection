#!/usr/bin/env python3
"""Build exact-dimension primary/safe coefficient table for CalibGuard-Dim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.core.undistort_ops import (
    resolve_border_mode,
    resolve_interpolation,
    undistort_via_maps,
)
from backend.evaluation import competition_proxy


HIGH_RISK_PARENTS = {"heavy_crop", "portrait_cropped"}
GLOBAL_FALLBACK = {"k1": -0.17, "k2": 0.35}

PARENT_PRIMARY_DEFAULTS: dict[str, dict[str, float]] = {
    "standard": {"k1": -0.170, "k2": 0.350},
    "near_standard_tall": {"k1": -0.170, "k2": 0.350},
    "near_standard_short": {"k1": -0.170, "k2": 0.350},
    "moderate_crop": {"k1": -0.080, "k2": 0.150},
    "heavy_crop": {"k1": -0.010, "k2": 0.010},
    "portrait_standard": {"k1": -0.020, "k2": 0.020},
    "portrait_cropped": {"k1": -0.005, "k2": 0.005},
}

PARENT_SAFE_DEFAULTS: dict[str, dict[str, float]] = {
    "standard": {"k1": -0.165, "k2": 0.340},
    "near_standard_tall": {"k1": -0.165, "k2": 0.340},
    "near_standard_short": {"k1": -0.160, "k2": 0.330},
    "moderate_crop": {"k1": -0.055, "k2": 0.100},
    "heavy_crop": {"k1": -0.002, "k2": 0.005},
    "portrait_standard": {"k1": -0.008, "k2": 0.015},
    "portrait_cropped": {"k1": 0.000, "k2": 0.000},
}


@dataclass
class TrainSample:
    original: np.ndarray
    target: np.ndarray
    target_gray: np.ndarray
    target_edges: np.ndarray
    target_line_hist: np.ndarray
    target_grad_hist: np.ndarray


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


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def apply_undistortion(
    image: np.ndarray,
    k1: float,
    k2: float,
    *,
    alpha: float = 0.0,
    interpolation: str = "linear",
    border_mode: str = "constant",
    model_type: str = "brown",
) -> np.ndarray:
    height, width = image.shape[:2]
    if model_type == "rational":
        dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)
    corrected, _ = undistort_via_maps(
        image,
        dist_coeffs=dist_coeffs,
        alpha=alpha,
        interpolation=interpolation,
        border_mode=border_mode,
    )
    if corrected.shape[:2] != (height, width):
        corrected = cv2.resize(
            corrected, (width, height), interpolation=cv2.INTER_LANCZOS4
        )
    return corrected


def edge_f1(pred_edges: np.ndarray, gt_edges: np.ndarray) -> float:
    pred_bool = pred_edges > 0
    gt_bool = gt_edges > 0
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    return float((2.0 * tp) / (2.0 * tp + fp + fn))


def line_angle_hist(edges: np.ndarray, bins: int = 18) -> np.ndarray:
    min_line = max(8, min(edges.shape[0], edges.shape[1]) // 6)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_line,
        maxLineGap=6,
    )
    if lines is None or len(lines) == 0:
        return np.zeros(bins, dtype=np.float32)

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = (angle + 180.0) % 180.0
        angles.append(angle)

    if not angles:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(np.array(angles), bins=bins, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float32)
    return hist / total


def line_angle_loss(pred_hist: np.ndarray, gt_hist: np.ndarray) -> float:
    pred_sum = float(pred_hist.sum())
    gt_sum = float(gt_hist.sum())
    if pred_sum == 0.0 and gt_sum == 0.0:
        return 0.0
    if pred_sum == 0.0 or gt_sum == 0.0:
        return 1.0
    return float(0.5 * np.abs(pred_hist - gt_hist).sum())


def grad_hist(gray: np.ndarray, bins: int = 36) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag = mag.astype(np.float32)
    threshold = float(np.percentile(mag, 60.0)) if mag.size else 0.0
    mask = mag > max(threshold, 1e-6)
    if not np.any(mask):
        return np.zeros(bins, dtype=np.float32)

    hist, _ = np.histogram(
        ang[mask],
        bins=bins,
        range=(0.0, 360.0),
        weights=mag[mask],
    )
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float32)
    return hist / total


def grad_hist_loss(pred_hist: np.ndarray, gt_hist: np.ndarray) -> float:
    pred_norm = float(np.linalg.norm(pred_hist))
    gt_norm = float(np.linalg.norm(gt_hist))
    if pred_norm == 0.0 and gt_norm == 0.0:
        return 0.0
    if pred_norm == 0.0 or gt_norm == 0.0:
        return 1.0
    cosine = float(np.dot(pred_hist, gt_hist) / (pred_norm * gt_norm + 1e-8))
    return float(np.clip(1.0 - cosine, 0.0, 1.0))


def ssim_gray(pred_gray: np.ndarray, gt_gray: np.ndarray) -> float:
    x = pred_gray.astype(np.float32)
    y = gt_gray.astype(np.float32)

    mu_x = float(x.mean())
    mu_y = float(y.mean())
    sigma_x = float(((x - mu_x) ** 2).mean())
    sigma_y = float(((y - mu_y) ** 2).mean())
    sigma_xy = float(((x - mu_x) * (y - mu_y)).mean())

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def norm_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(
        np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))) / 255.0
    )


def warp_penalty(width: int, height: int, k1: float, k2: float) -> float:
    r_max = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2) / max(width, height)
    r = np.linspace(0.0, r_max, 64, dtype=np.float32)
    radial = np.abs(k1 * (r**3) + k2 * (r**5))
    return float(np.clip(float(radial.max()) * 14.0, 0.0, 1.0))


def black_border_ratio(image: np.ndarray, threshold: int = 2) -> float:
    mask = np.all(image <= threshold, axis=2)
    return float(mask.mean())


def build_train_index(train_dir: Path) -> dict[str, dict[str, Any]]:
    files = sorted([f for f in os.listdir(train_dir) if f.endswith("_original.jpg")])
    index: dict[str, dict[str, Any]] = {}
    total = len(files)

    for idx, orig_name in enumerate(files, start=1):
        gen_name = orig_name.replace("_original.jpg", "_generated.jpg")
        orig_path = train_dir / orig_name
        gen_path = train_dir / gen_name
        if not gen_path.exists():
            continue

        image = cv2.imread(str(orig_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        dim_key = f"{width}x{height}"
        parent_class = classify_parent(height, width)

        dim_data = index.setdefault(
            dim_key,
            {
                "width": width,
                "height": height,
                "parent_class": parent_class,
                "pairs": [],
            },
        )
        dim_data["pairs"].append((orig_path, gen_path))

        if idx % 2000 == 0 or idx == total:
            print(f"  Indexed {idx}/{total} training originals")

    return index


def load_train_samples(
    pairs: list[tuple[Path, Path]],
    sample_count: int,
    rng: random.Random,
    scale: float,
) -> list[TrainSample]:
    if not pairs:
        return []

    selected = list(pairs)
    rng.shuffle(selected)
    selected = selected[: min(sample_count, len(selected))]

    samples: list[TrainSample] = []
    for orig_path, gen_path in selected:
        orig = cv2.imread(str(orig_path))
        gen = cv2.imread(str(gen_path))
        if orig is None or gen is None:
            continue

        if scale != 1.0:
            orig = cv2.resize(
                orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            gen = cv2.resize(
                gen, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        gen_gray = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
        gen_edges = cv2.Canny(gen_gray, 80, 160)

        samples.append(
            TrainSample(
                original=orig,
                target=gen,
                target_gray=gen_gray,
                target_edges=gen_edges,
                target_line_hist=line_angle_hist(gen_edges),
                target_grad_hist=grad_hist(gen_gray),
            )
        )

    return samples


def evaluate_candidate(
    samples: list[TrainSample],
    width: int,
    height: int,
    k1: float,
    k2: float,
    *,
    alpha: float,
    interpolation: str,
    border_mode: str,
    metric_profile: str,
    model_type: str,
) -> dict[str, float]:
    if not samples:
        return {
            "loss": float("inf"),
            "edge_f1": 0.0,
            "line_angle_loss": 1.0,
            "grad_hist_loss": 1.0,
            "ssim": 0.0,
            "norm_mae": 1.0,
            "warp_penalty": 1.0,
            "border_ratio": 1.0,
            "risk": 1.0,
        }

    edge_vals: list[float] = []
    line_vals: list[float] = []
    grad_vals: list[float] = []
    ssim_vals: list[float] = []
    mae_vals: list[float] = []
    border_vals: list[float] = []

    warp = warp_penalty(width, height, k1, k2)

    for sample in samples:
        corrected = apply_undistortion(
            sample.original,
            k1,
            k2,
            alpha=alpha,
            interpolation=interpolation,
            border_mode=border_mode,
            model_type=model_type,
        )
        if corrected.shape[:2] != sample.target.shape[:2]:
            corrected = cv2.resize(
                corrected,
                (sample.target.shape[1], sample.target.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        if metric_profile == "competition_lite":
            edge_vals.append(
                competition_proxy.edge_similarity_multiscale(corrected, sample.target)
            )
            line_vals.append(
                competition_proxy.line_orientation_loss(corrected, sample.target)
            )
            grad_vals.append(
                competition_proxy.gradient_orientation_loss(corrected, sample.target)
            )
            ssim_vals.append(competition_proxy.ssim_score(corrected, sample.target))
            mae_vals.append(competition_proxy.normalized_mae(corrected, sample.target))
        else:
            corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            corrected_edges = cv2.Canny(corrected_gray, 80, 160)

            edge_vals.append(edge_f1(corrected_edges, sample.target_edges))
            line_vals.append(
                line_angle_loss(
                    line_angle_hist(corrected_edges), sample.target_line_hist
                )
            )
            grad_vals.append(
                grad_hist_loss(grad_hist(corrected_gray), sample.target_grad_hist)
            )
            ssim_vals.append(ssim_gray(corrected_gray, sample.target_gray))
            mae_vals.append(norm_mae(corrected, sample.target))
        border_vals.append(black_border_ratio(corrected))

    edge_mean = float(np.mean(edge_vals))
    line_mean = float(np.mean(line_vals))
    grad_mean = float(np.mean(grad_vals))
    ssim_mean = float(np.mean(ssim_vals))
    mae_mean = float(np.mean(mae_vals))
    border_mean = float(np.mean(border_vals))

    loss = (
        0.40 * (1.0 - edge_mean)
        + 0.22 * line_mean
        + 0.18 * grad_mean
        + 0.15 * (1.0 - ssim_mean)
        + 0.05 * mae_mean
        + 0.05 * warp
    )

    border_scaled = float(np.clip(border_mean / 0.002, 0.0, 1.0))
    risk = 0.55 * border_scaled + 0.30 * warp + 0.15 * (1.0 - edge_mean)

    return {
        "loss": float(loss),
        "edge_f1": edge_mean,
        "line_angle_loss": line_mean,
        "grad_hist_loss": grad_mean,
        "ssim": ssim_mean,
        "norm_mae": mae_mean,
        "warp_penalty": warp,
        "border_ratio": border_mean,
        "risk": float(np.clip(risk, 0.0, 1.0)),
        "alpha": float(alpha),
    }


def _candidate_grid(
    base_k1: float,
    base_k2: float,
    k1_offsets: Iterable[float],
    k2_offsets: Iterable[float],
    alpha_values: Iterable[float],
) -> list[tuple[float, float, float]]:
    candidates: set[tuple[float, float, float]] = set()
    for dk1 in k1_offsets:
        for dk2 in k2_offsets:
            for alpha in alpha_values:
                candidates.add(
                    (
                        round(base_k1 + dk1, 6),
                        round(base_k2 + dk2, 6),
                        round(float(alpha), 6),
                    )
                )
    for alpha in alpha_values:
        candidates.add((round(base_k1, 6), round(base_k2, 6), round(float(alpha), 6)))
        candidates.add(
            (GLOBAL_FALLBACK["k1"], GLOBAL_FALLBACK["k2"], round(float(alpha), 6))
        )
    return sorted(candidates)


def parse_alpha_grid(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        values = [0.0]
    unique = sorted({round(v, 6) for v in values})
    return [float(v) for v in unique]


def _evaluate_entry(
    samples: list[TrainSample],
    width: int,
    height: int,
    k1: float,
    k2: float,
    alpha: float,
    interpolation: str,
    border_mode: str,
    metric_profile: str,
    model_type: str,
) -> dict[str, Any]:
    metrics = evaluate_candidate(
        samples=samples,
        width=width,
        height=height,
        k1=k1,
        k2=k2,
        alpha=alpha,
        interpolation=interpolation,
        border_mode=border_mode,
        metric_profile=metric_profile,
        model_type=model_type,
    )
    return {
        "k1": float(k1),
        "k2": float(k2),
        "alpha": float(alpha),
        "metrics": metrics,
    }


def _search_grid(
    samples: list[TrainSample],
    width: int,
    height: int,
    base_k1: float,
    base_k2: float,
    broad: bool,
    alpha_values: list[float],
    interpolation: str,
    border_mode: str,
    metric_profile: str,
    model_type: str,
) -> tuple[dict[str, Any], dict[str, Any], int, int, dict[str, Any]]:
    if broad:
        k1_offsets = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
        k2_offsets = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
    else:
        k1_offsets = [-0.01, -0.005, 0.0, 0.005, 0.01]
        k2_offsets = [-0.02, -0.01, 0.0, 0.01, 0.02]

    candidates = _candidate_grid(base_k1, base_k2, k1_offsets, k2_offsets, alpha_values)
    evals: list[dict[str, Any]] = []
    valid = 0
    for k1, k2, alpha in candidates:
        entry = _evaluate_entry(
            samples,
            width,
            height,
            k1,
            k2,
            alpha,
            interpolation,
            border_mode,
            metric_profile,
            model_type,
        )
        if entry["metrics"]["border_ratio"] <= 0.002:
            valid += 1
            evals.append(entry)

    if not evals:
        fallback_primary = _evaluate_entry(
            samples,
            width,
            height,
            base_k1,
            base_k2,
            alpha_values[0],
            interpolation,
            border_mode,
            metric_profile,
            model_type,
        )
        fallback_safe = fallback_primary.copy()
        return (
            fallback_primary,
            fallback_safe,
            len(candidates),
            valid,
            {"optimizer": "grid"},
        )

    primary = min(evals, key=lambda x: x["metrics"]["loss"])

    refine_candidates = _candidate_grid(
        primary["k1"],
        primary["k2"],
        k1_offsets=[-0.01, -0.005, 0.0, 0.005, 0.01],
        k2_offsets=[-0.02, -0.01, 0.0, 0.01, 0.02],
        alpha_values=alpha_values,
    )
    for k1, k2, alpha in refine_candidates:
        entry = _evaluate_entry(
            samples,
            width,
            height,
            k1,
            k2,
            alpha,
            interpolation,
            border_mode,
            metric_profile,
            model_type,
        )
        if entry["metrics"]["border_ratio"] <= 0.002:
            evals.append(entry)

    primary = min(evals, key=lambda x: x["metrics"]["loss"])
    safe = min(evals, key=lambda x: x["metrics"]["risk"])
    return (
        primary,
        safe,
        len(candidates) + len(refine_candidates),
        valid,
        {"optimizer": "grid"},
    )


def _search_de(
    samples: list[TrainSample],
    width: int,
    height: int,
    base_k1: float,
    base_k2: float,
    broad: bool,
    alpha_values: list[float],
    interpolation: str,
    border_mode: str,
    metric_profile: str,
    model_type: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], int, int, dict[str, Any]]:
    try:
        from backend.scripts.heuristics.optimizer_scipy import (
            differential_evolution_search,
        )
    except Exception:
        return _search_grid(
            samples,
            width,
            height,
            base_k1,
            base_k2,
            broad,
            alpha_values,
            interpolation,
            border_mode,
            metric_profile,
            model_type,
        )

    k1_span = 0.06 if broad else 0.02
    k2_span = 0.12 if broad else 0.04
    bounds: list[tuple[float, float]] = [
        (base_k1 - k1_span, base_k1 + k1_span),
        (base_k2 - k2_span, base_k2 + k2_span),
    ]
    optimize_alpha = len(alpha_values) > 1
    if optimize_alpha:
        bounds.append((min(alpha_values), max(alpha_values)))

    def objective(params: np.ndarray) -> float:
        k1 = float(params[0])
        k2 = float(params[1])
        alpha = float(params[2]) if optimize_alpha else float(alpha_values[0])
        metrics = evaluate_candidate(
            samples=samples,
            width=width,
            height=height,
            k1=k1,
            k2=k2,
            alpha=alpha,
            interpolation=interpolation,
            border_mode=border_mode,
            metric_profile=metric_profile,
            model_type=model_type,
        )
        border_excess = max(0.0, float(metrics["border_ratio"]) - 0.002)
        return float(metrics["loss"] + border_excess * 25.0)

    result = differential_evolution_search(
        objective=objective,
        bounds=bounds,
        seed=seed,
        maxiter=12 if broad else 8,
        popsize=8 if broad else 6,
        timeout_s=60.0 if broad else 35.0,
    )

    best_x = np.asarray(result["x"], dtype=np.float64)
    best_k1 = float(best_x[0])
    best_k2 = float(best_x[1])
    best_alpha = float(best_x[2]) if optimize_alpha else float(alpha_values[0])

    local_alphas = alpha_values if optimize_alpha else [best_alpha]
    refine_candidates = _candidate_grid(
        best_k1,
        best_k2,
        k1_offsets=[-0.01, -0.005, 0.0, 0.005, 0.01],
        k2_offsets=[-0.02, -0.01, 0.0, 0.01, 0.02],
        alpha_values=local_alphas,
    )

    evals: list[dict[str, Any]] = []
    valid = 0
    best_entry = _evaluate_entry(
        samples,
        width,
        height,
        best_k1,
        best_k2,
        best_alpha,
        interpolation,
        border_mode,
        metric_profile,
        model_type,
    )
    if best_entry["metrics"]["border_ratio"] <= 0.002:
        evals.append(best_entry)
        valid += 1

    for k1, k2, alpha in refine_candidates:
        entry = _evaluate_entry(
            samples,
            width,
            height,
            k1,
            k2,
            alpha,
            interpolation,
            border_mode,
            metric_profile,
            model_type,
        )
        if entry["metrics"]["border_ratio"] <= 0.002:
            evals.append(entry)
            valid += 1

    if not evals:
        fallback_primary = _evaluate_entry(
            samples,
            width,
            height,
            base_k1,
            base_k2,
            alpha_values[0],
            interpolation,
            border_mode,
            metric_profile,
            model_type,
        )
        fallback_safe = fallback_primary.copy()
        meta = {"optimizer": "de_fallback", "de": result}
        return (
            fallback_primary,
            fallback_safe,
            int(result["evaluations"]) + len(refine_candidates),
            valid,
            meta,
        )

    primary = min(evals, key=lambda x: x["metrics"]["loss"])
    safe = min(evals, key=lambda x: x["metrics"]["risk"])
    meta = {"optimizer": "de", "de": result}
    return (
        primary,
        safe,
        int(result["evaluations"]) + len(refine_candidates),
        valid,
        meta,
    )


def _search(
    samples: list[TrainSample],
    width: int,
    height: int,
    base_k1: float,
    base_k2: float,
    broad: bool,
    alpha_values: list[float],
    interpolation: str,
    border_mode: str,
    metric_profile: str,
    optimizer: str,
    model_type: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], int, int, dict[str, Any]]:
    if optimizer == "de":
        return _search_de(
            samples=samples,
            width=width,
            height=height,
            base_k1=base_k1,
            base_k2=base_k2,
            broad=broad,
            alpha_values=alpha_values,
            interpolation=interpolation,
            border_mode=border_mode,
            metric_profile=metric_profile,
            model_type=model_type,
            seed=seed,
        )
    return _search_grid(
        samples=samples,
        width=width,
        height=height,
        base_k1=base_k1,
        base_k2=base_k2,
        broad=broad,
        alpha_values=alpha_values,
        interpolation=interpolation,
        border_mode=border_mode,
        metric_profile=metric_profile,
        model_type=model_type,
    )


def scored_dim_stats(
    score_csv: Path, test_dir: Path
) -> tuple[dict[str, int], dict[str, float]]:
    if not score_csv.exists():
        return {}, {}

    counts: dict[str, int] = {}
    zeros: dict[str, int] = {}

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

            image = cv2.imread(str(test_dir / f"{image_id}.jpg"))
            if image is None:
                continue
            h, w = image.shape[:2]
            dim_key = f"{w}x{h}"

            counts[dim_key] = counts.get(dim_key, 0) + 1
            if score == 0.0:
                zeros[dim_key] = zeros.get(dim_key, 0) + 1

    zero_rate: dict[str, float] = {}
    for dim_key, count in counts.items():
        z = zeros.get(dim_key, 0)
        zero_rate[dim_key] = (z / max(count, 1)) * 100.0
    return counts, zero_rate


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": "calibguard_dim_manifest.v1",
            "builds": [],
            "runs": [],
        }
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("version", "calibguard_dim_manifest.v1")
    data.setdefault("builds", [])
    data.setdefault("runs", [])
    return data


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Build CalibGuard-Dim coefficient table"
    )
    parser.add_argument("--train-dir", default=str(cfg.train_dir))
    parser.add_argument("--sample-per-dim", type=int, default=300)
    parser.add_argument("--min-dim-support", type=int, default=30)
    parser.add_argument(
        "--out-table",
        default=str(
            cfg.repo_root / "backend/scripts/heuristics/calibguard_dim_table.json"
        ),
    )
    parser.add_argument(
        "--out-manifest",
        default=str(
            cfg.repo_root / "backend/scripts/heuristics/calibguard_dim_manifest.json"
        ),
    )
    parser.add_argument("--score-csv", default=str(cfg.repo_root / "submission.csv"))
    parser.add_argument("--test-dir", default=str(cfg.test_dir))
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dims", type=int, default=None)
    parser.add_argument(
        "--interp", choices=["linear", "cubic", "lanczos4"], default="linear"
    )
    parser.add_argument(
        "--border-mode",
        choices=["constant", "reflect", "replicate"],
        default="constant",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0.0",
        help="Comma-separated alpha candidates for getOptimalNewCameraMatrix (e.g. 0.0,0.1,0.2).",
    )
    parser.add_argument("--optimizer", choices=["grid", "de"], default="grid")
    parser.add_argument("--enable-rational", action="store_true")
    parser.add_argument(
        "--metric-profile", choices=["legacy", "competition_lite"], default="legacy"
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    out_table = Path(args.out_table).expanduser().resolve()
    out_manifest = Path(args.out_manifest).expanduser().resolve()
    score_csv = Path(args.score_csv).expanduser().resolve()

    require_existing_dir(train_dir, "Training images directory")
    require_existing_dir(test_dir, "Test images directory")
    ensure_dir(out_table.parent)
    ensure_dir(out_manifest.parent)

    interp_name, _ = resolve_interpolation(args.interp)
    border_name, _ = resolve_border_mode(args.border_mode)
    alpha_values = parse_alpha_grid(args.alpha_grid)
    model_type = "rational" if args.enable_rational else "brown"

    rng = random.Random(args.seed)

    print("Building train index by exact dimension...")
    index = build_train_index(train_dir)
    dim_items = sorted(index.items(), key=lambda kv: len(kv[1]["pairs"]), reverse=True)
    if args.max_dims is not None:
        dim_items = dim_items[: max(0, args.max_dims)]

    test_counts, test_zero_rates = scored_dim_stats(score_csv, test_dir)

    table: dict[str, Any] = {
        "version": "calibguard_dim_table.v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "method": "CalibGuard-Dim",
        "train_dir": str(train_dir),
        "sample_per_dim": int(args.sample_per_dim),
        "min_dim_support": int(args.min_dim_support),
        "scale": float(args.scale),
        "global_fallback": GLOBAL_FALLBACK,
        "model_defaults": {"type": model_type},
        "render_defaults": {
            "interp": interp_name,
            "border_mode": border_name,
            "alpha_grid": alpha_values,
        },
        "search_defaults": {
            "optimizer": args.optimizer,
            "metric_profile": args.metric_profile,
        },
        "parent_classes": {},
        "dimensions": {},
        "metadata": {
            "git_commit": _git_commit(),
            "score_csv": str(score_csv),
            "test_dir": str(test_dir),
        },
    }

    for parent in sorted(PARENT_PRIMARY_DEFAULTS):
        table["parent_classes"][parent] = {
            "primary": PARENT_PRIMARY_DEFAULTS[parent],
            "safe": PARENT_SAFE_DEFAULTS[parent],
        }

    print(f"Tuning {len(dim_items)} dimension groups...")
    for idx, (dim_key, dim_info) in enumerate(dim_items, start=1):
        support = len(dim_info["pairs"])
        width = int(dim_info["width"])
        height = int(dim_info["height"])
        parent = str(dim_info["parent_class"])

        print(f"  [{idx}/{len(dim_items)}] {dim_key} support={support} parent={parent}")
        samples = load_train_samples(
            dim_info["pairs"], args.sample_per_dim, rng, args.scale
        )
        if not samples:
            print("    No valid samples loaded; using parent defaults")
            primary = {
                "k1": float(PARENT_PRIMARY_DEFAULTS[parent]["k1"]),
                "k2": float(PARENT_PRIMARY_DEFAULTS[parent]["k2"]),
                "alpha": float(alpha_values[0]),
                "metrics": {},
            }
            safe = {
                "k1": float(PARENT_SAFE_DEFAULTS[parent]["k1"]),
                "k2": float(PARENT_SAFE_DEFAULTS[parent]["k2"]),
                "alpha": float(alpha_values[0]),
                "metrics": {},
            }
            evaluated = 0
            valid = 0
            search_meta: dict[str, Any] = {"optimizer": args.optimizer}
        else:
            base = PARENT_PRIMARY_DEFAULTS[parent]
            broad = support >= args.min_dim_support
            primary, safe, evaluated, valid, search_meta = _search(
                samples,
                width=width,
                height=height,
                base_k1=float(base["k1"]),
                base_k2=float(base["k2"]),
                broad=broad,
                alpha_values=alpha_values,
                interpolation=interp_name,
                border_mode=border_name,
                metric_profile=args.metric_profile,
                optimizer=args.optimizer,
                model_type=model_type,
                seed=int(args.seed),
            )

        reasons: list[str] = []
        zero_rate = float(test_zero_rates.get(dim_key, 0.0))
        test_count = int(test_counts.get(dim_key, 0))
        if zero_rate >= 20.0:
            reasons.append(f"zero_rate_ge_20pct({zero_rate:.2f})")
        if 0 < test_count < 10:
            reasons.append(f"test_count_lt_10({test_count})")
        if parent in HIGH_RISK_PARENTS:
            reasons.append(f"high_risk_parent({parent})")

        table["dimensions"][dim_key] = {
            "width": width,
            "height": height,
            "parent_class": parent,
            "support": support,
            "sampled_pairs": len(samples),
            "search": {
                "broad": support >= args.min_dim_support,
                "evaluated_candidates": evaluated,
                "valid_candidates": valid,
                "black_border_max": 0.002,
                "optimizer": args.optimizer,
                "metric_profile": args.metric_profile,
                "meta": search_meta,
            },
            "primary": {
                "k1": primary["k1"],
                "k2": primary["k2"],
                "alpha": float(primary.get("alpha", alpha_values[0])),
                "metrics": primary.get("metrics", {}),
            },
            "safe": {
                "k1": safe["k1"],
                "k2": safe["k2"],
                "alpha": float(safe.get("alpha", alpha_values[0])),
                "metrics": safe.get("metrics", {}),
            },
            "model": {
                "type": model_type,
            },
            "render": {
                "interp": interp_name,
                "border_mode": border_name,
                "alpha": float(primary.get("alpha", alpha_values[0])),
            },
            "guardrails": {
                "force_safe": len(reasons) > 0,
                "reasons": reasons,
                "test_count": test_count,
                "latest_zero_rate": zero_rate,
            },
        }

    with out_table.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, sort_keys=True)

    table_hash = _sha256_file(out_table)

    manifest = load_manifest(out_manifest)
    manifest["builds"].append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_commit": _git_commit(),
            "args": {
                "train_dir": str(train_dir),
                "sample_per_dim": int(args.sample_per_dim),
                "min_dim_support": int(args.min_dim_support),
                "scale": float(args.scale),
                "seed": int(args.seed),
                "max_dims": args.max_dims,
                "interp": interp_name,
                "border_mode": border_name,
                "alpha_grid": alpha_values,
                "optimizer": args.optimizer,
                "enable_rational": bool(args.enable_rational),
                "metric_profile": args.metric_profile,
            },
            "table_path": str(out_table),
            "table_sha256": table_hash,
            "dimensions_tuned": len(table["dimensions"]),
        }
    )

    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("\n=== CalibGuard-Dim Table Build Complete ===")
    print(f"Table: {out_table}")
    print(f"Table SHA256: {table_hash}")
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
