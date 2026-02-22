#!/usr/bin/env python3
"""Flow-Residual CalibGuard heuristic with per-dimension residual maps and gating."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.evaluation import competition_proxy
from backend.scripts.heuristics.heuristic_calibguard_dim import (
    choose_coeffs,
    choose_model_type,
    classify_parent,
)
from backend.scripts.heuristics.zip_repair import (
    ZipFinalizeStats,
    finalize_zip_with_verification,
    safe_read_original,
    safe_read_path,
)


FLOW_WORK_SCALE = 0.25
FLOW_SMOOTH_SIGMA = 1.2
FLOW_STABILITY_VAR_MAX = 3.5
BORDER_RATIO_MAX = 0.002
WARP_RISK_MAX = 0.35


@dataclass(frozen=True)
class DimensionFlowModel:
    """Calibrated residual-flow model for one exact dimension."""

    dim_key: str
    width: int
    height: int
    support: int
    parent_class: str
    base_coeffs: tuple[float, float]
    coeff_source: str
    model_type: str
    lambda_value: float
    accepted: bool
    reason: str
    proxy_gain: float
    base_loss: float
    best_loss: float
    border_ratio: float
    flow_var: float
    flow_p99: float
    flow_clip_max: float
    warp_risk: float
    flow_map: np.ndarray | None = None


def parse_lambda_grid(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        values = [0.0]
    return sorted({round(v, 6) for v in values})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_table(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_dim_key(dim_key: str) -> tuple[int, int]:
    w_s, h_s = dim_key.lower().split("x", 1)
    return int(w_s), int(h_s)


def load_train_index(train_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Map exact dimensions to training pair paths."""
    return load_train_index_filtered(train_dir)


def load_train_index_filtered(
    train_dir: Path,
    *,
    target_dims: set[str] | None = None,
    per_dim_cap: int | None = None,
) -> dict[str, list[tuple[Path, Path]]]:
    """Map exact dimensions to training pair paths with optional target/cap limits."""
    pairs_by_dim: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    originals = sorted([p for p in train_dir.iterdir() if p.name.endswith("_original.jpg")])
    remaining_dims = set(target_dims) if target_dims else None

    for idx, orig_path in enumerate(originals, start=1):
        if remaining_dims is not None and not remaining_dims:
            break

        gen_path = train_dir / orig_path.name.replace("_original.jpg", "_generated.jpg")
        if not gen_path.exists():
            continue
        image = cv2.imread(str(orig_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        dim_key = f"{w}x{h}"

        if target_dims is not None and dim_key not in target_dims:
            continue

        if per_dim_cap is not None and len(pairs_by_dim[dim_key]) >= per_dim_cap:
            if remaining_dims is not None and dim_key in remaining_dims:
                remaining_dims.discard(dim_key)
            continue

        pairs_by_dim[dim_key].append((orig_path, gen_path))
        if per_dim_cap is not None and remaining_dims is not None:
            if len(pairs_by_dim[dim_key]) >= per_dim_cap:
                remaining_dims.discard(dim_key)

        if idx % 4000 == 0:
            print(f"  Indexed {idx}/{len(originals)} train originals...")

    if target_dims:
        covered = sum(1 for dim in target_dims if dim in pairs_by_dim and pairs_by_dim[dim])
        print(
            f"  Indexed target dimensions with data: {covered}/{len(target_dims)} "
            f"(per_dim_cap={per_dim_cap})"
        )

    return pairs_by_dim


def apply_base_undistortion(
    image: np.ndarray,
    k1: float,
    k2: float,
    *,
    model_type: str,
) -> np.ndarray:
    """Apply base CalibGuard undistortion with alpha fixed to 0."""
    h, w = image.shape[:2]
    focal = float(np.sqrt(w * w + h * h))
    camera = np.array(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    if model_type == "rational":
        dist = np.array([k1, k2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

    new_camera, _ = cv2.getOptimalNewCameraMatrix(camera, dist, (w, h), 0.0, (w, h))
    corrected = cv2.undistort(image, camera, dist, None, new_camera)
    if corrected.shape[:2] != (h, w):
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return corrected


def compute_dense_flow(src_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        src_gray,
        tgt_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=17,
        iterations=4,
        poly_n=7,
        poly_sigma=1.4,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    return flow.astype(np.float32)


def clip_flow_magnitude(flow: np.ndarray, clip_max: float) -> np.ndarray:
    if clip_max <= 0.0:
        return np.zeros_like(flow)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    safe_mag = np.maximum(mag, 1e-6)
    scale = np.minimum(1.0, clip_max / safe_mag)
    clipped = flow * scale[..., None]
    return clipped.astype(np.float32)


def aggregate_residual_flow(
    flow_stack: list[np.ndarray],
    *,
    flow_max_mag: float,
    smooth_sigma: float = FLOW_SMOOTH_SIGMA,
) -> tuple[np.ndarray, float, float, float]:
    """Aggregate per-pair flows using robust median and magnitude clipping."""
    if not flow_stack:
        raise ValueError("flow_stack must not be empty")

    stack = np.stack(flow_stack, axis=0).astype(np.float32)
    median_flow = np.median(stack, axis=0).astype(np.float32)

    med_mag = np.sqrt(median_flow[..., 0] ** 2 + median_flow[..., 1] ** 2)
    p99 = float(np.percentile(med_mag, 99.0)) if med_mag.size else 0.0
    clip_max = float(min(max(p99, 0.0), max(flow_max_mag, 0.0)))

    clipped = clip_flow_magnitude(median_flow, clip_max)
    smoothed = cv2.GaussianBlur(clipped, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)

    centered = stack - stack.mean(axis=0, keepdims=True)
    var_scalar = float(np.mean(centered[..., 0] ** 2 + centered[..., 1] ** 2))

    return smoothed.astype(np.float32), var_scalar, p99, clip_max


def resize_flow_to_image(flow: np.ndarray, height: int, width: int) -> np.ndarray:
    fh, fw = flow.shape[:2]
    resized = cv2.resize(flow, (width, height), interpolation=cv2.INTER_CUBIC)
    resized[..., 0] *= float(width) / float(fw)
    resized[..., 1] *= float(height) / float(fh)
    return resized.astype(np.float32)


def apply_residual_flow(image: np.ndarray, flow: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0.0:
        return image.copy()

    h, w = image.shape[:2]
    flow_scaled = resize_flow_to_image(flow, h, w) * float(lam)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs - flow_scaled[..., 0]
    map_y = ys - flow_scaled[..., 1]

    remapped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return remapped


def compute_black_border_ratio(image: np.ndarray, threshold: int = 2) -> float:
    mask = np.all(image <= threshold, axis=2)
    return float(mask.mean())


def residual_warp_risk(flow: np.ndarray, lam: float) -> float:
    if lam <= 0.0:
        return 0.0
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    if mag.size == 0:
        return 0.0
    p99 = float(np.percentile(mag, 99.0))
    return float(abs(lam) * p99)


def competition_lite_loss(pred: np.ndarray, target: np.ndarray) -> float:
    edge = competition_proxy.edge_similarity_multiscale(pred, target)
    line = competition_proxy.line_orientation_loss(pred, target)
    grad = competition_proxy.gradient_orientation_loss(pred, target)
    ssim = competition_proxy.ssim_score(pred, target)
    mae = competition_proxy.normalized_mae(pred, target)
    return float(
        0.40 * (1.0 - edge)
        + 0.22 * line
        + 0.18 * grad
        + 0.15 * (1.0 - ssim)
        + 0.05 * mae
    )


def should_accept_dimension(
    *,
    proxy_gain: float,
    border_ratio: float,
    warp_risk: float,
    flow_var: float,
    min_proxy_gain: float,
    max_border_ratio: float,
    max_warp_risk: float,
    max_flow_var: float,
) -> tuple[bool, str]:
    if flow_var > max_flow_var:
        return False, "reject_unstable_flow"
    if proxy_gain < min_proxy_gain:
        return False, "reject_low_proxy_gain"
    if border_ratio > max_border_ratio:
        return False, "reject_border_ratio"
    if warp_risk > max_warp_risk:
        return False, "reject_warp_risk"
    return True, "accepted"


def evaluate_lambdas(
    holdout_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    k1: float,
    k2: float,
    model_type: str,
    flow_map: np.ndarray,
    lambda_grid: list[float],
) -> tuple[float, float, float, float]:
    if not holdout_pairs:
        return 0.0, float("inf"), float("inf"), 1.0

    stats: dict[float, tuple[float, float]] = {}

    for lam in lambda_grid:
        losses: list[float] = []
        borders: list[float] = []

        for orig, target in holdout_pairs:
            base = apply_base_undistortion(orig, k1, k2, model_type=model_type)
            pred = apply_residual_flow(base, flow_map, lam)
            losses.append(competition_lite_loss(pred, target))
            borders.append(compute_black_border_ratio(pred))

        stats[lam] = (float(np.mean(losses)), float(np.mean(borders)))

    base_loss = stats.get(0.0, (float("inf"), 1.0))[0]
    best_lam = min(stats, key=lambda x: stats[x][0])
    best_loss, best_border = stats[best_lam]

    return float(best_lam), float(base_loss), float(best_loss), float(best_border)


def build_dimension_model(
    *,
    dim_key: str,
    pairs: list[tuple[Path, Path]],
    support_total: int,
    table: dict[str, Any],
    profile: str,
    train_sample_per_dim: int,
    holdout_ratio: float,
    flow_max_mag: float,
    lambda_grid: list[float],
    accept_min_proxy_gain: float,
    rng: random.Random,
) -> DimensionFlowModel:
    w, h = parse_dim_key(dim_key)
    parent = classify_parent(h, w)

    k1, k2, _alpha, coeff_source = choose_coeffs(table, dim_key, parent, profile)
    model_type = choose_model_type(table, dim_key)

    sampled = list(pairs)
    rng.shuffle(sampled)
    sampled = sampled[: min(len(sampled), train_sample_per_dim)]

    if len(sampled) < 4:
        return DimensionFlowModel(
            dim_key=dim_key,
            width=w,
            height=h,
            support=support_total,
            parent_class=parent,
            base_coeffs=(k1, k2),
            coeff_source=coeff_source,
            model_type=model_type,
            lambda_value=0.0,
            accepted=False,
            reason="reject_not_enough_samples",
            proxy_gain=0.0,
            base_loss=float("inf"),
            best_loss=float("inf"),
            border_ratio=1.0,
            flow_var=float("inf"),
            flow_p99=0.0,
            flow_clip_max=0.0,
            warp_risk=0.0,
            flow_map=None,
        )

    split_idx = max(1, int(round(len(sampled) * (1.0 - holdout_ratio))))
    split_idx = min(split_idx, len(sampled) - 1)

    train_subset = sampled[:split_idx]
    holdout_subset = sampled[split_idx:]

    flow_stack: list[np.ndarray] = []
    for orig_path, gen_path in train_subset:
        orig = cv2.imread(str(orig_path))
        target = cv2.imread(str(gen_path))
        if orig is None or target is None:
            continue

        base = apply_base_undistortion(orig, k1, k2, model_type=model_type)

        if FLOW_WORK_SCALE != 1.0:
            base = cv2.resize(base, None, fx=FLOW_WORK_SCALE, fy=FLOW_WORK_SCALE, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)

        flow_stack.append(compute_dense_flow(base, target))

    if not flow_stack:
        return DimensionFlowModel(
            dim_key=dim_key,
            width=w,
            height=h,
            support=support_total,
            parent_class=parent,
            base_coeffs=(k1, k2),
            coeff_source=coeff_source,
            model_type=model_type,
            lambda_value=0.0,
            accepted=False,
            reason="reject_no_flow_pairs",
            proxy_gain=0.0,
            base_loss=float("inf"),
            best_loss=float("inf"),
            border_ratio=1.0,
            flow_var=float("inf"),
            flow_p99=0.0,
            flow_clip_max=0.0,
            warp_risk=0.0,
            flow_map=None,
        )

    flow_map, flow_var, flow_p99, flow_clip_max = aggregate_residual_flow(
        flow_stack,
        flow_max_mag=flow_max_mag,
    )

    holdout_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for orig_path, gen_path in holdout_subset:
        orig = cv2.imread(str(orig_path))
        target = cv2.imread(str(gen_path))
        if orig is None or target is None:
            continue
        holdout_pairs.append((orig, target))

    best_lam, base_loss, best_loss, best_border = evaluate_lambdas(
        holdout_pairs,
        k1=k1,
        k2=k2,
        model_type=model_type,
        flow_map=flow_map,
        lambda_grid=lambda_grid,
    )

    proxy_gain = float(base_loss - best_loss)
    warp_risk = residual_warp_risk(flow_map, best_lam)
    accepted, reason = should_accept_dimension(
        proxy_gain=proxy_gain,
        border_ratio=best_border,
        warp_risk=warp_risk,
        flow_var=flow_var,
        min_proxy_gain=accept_min_proxy_gain,
        max_border_ratio=BORDER_RATIO_MAX,
        max_warp_risk=WARP_RISK_MAX,
        max_flow_var=FLOW_STABILITY_VAR_MAX,
    )

    if not accepted:
        best_lam = 0.0

    return DimensionFlowModel(
        dim_key=dim_key,
        width=w,
        height=h,
        support=support_total,
        parent_class=parent,
        base_coeffs=(k1, k2),
        coeff_source=coeff_source,
        model_type=model_type,
        lambda_value=float(best_lam),
        accepted=accepted,
        reason=reason,
        proxy_gain=proxy_gain,
        base_loss=base_loss,
        best_loss=best_loss,
        border_ratio=best_border,
        flow_var=flow_var,
        flow_p99=flow_p99,
        flow_clip_max=flow_clip_max,
        warp_risk=warp_risk,
        flow_map=flow_map if accepted else None,
    )


def build_zip_from_render_dirs(
    *,
    output_zip: Path,
    expected_names: list[str],
    primary_dir: Path,
    base_dir: Path,
    test_original_dir: Path,
    repair_mode: str,
) -> ZipFinalizeStats:
    return finalize_zip_with_verification(
        output_zip=output_zip,
        expected_names=expected_names,
        primary_reader=lambda name: safe_read_path(primary_dir / name, source="primary"),
        base_reader=lambda name: safe_read_path(base_dir / name, source="base"),
        original_reader=lambda name: safe_read_original(test_original_dir, name),
        repair_mode=repair_mode,
        expected_count=len(expected_names),
    )


def write_ids(path: Path, ids: list[str]) -> None:
    body = "\n".join(sorted(ids))
    if body:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Flow-residual exact-dimension heuristic")
    parser.add_argument("--train-dir", default=str(cfg.train_dir))
    parser.add_argument("--test-dir", default=str(cfg.test_dir))
    parser.add_argument("--output-dir", default=str(cfg.output_root))
    parser.add_argument(
        "--dim-table-path",
        default=str(cfg.repo_root / "backend/scripts/heuristics/calibguard_dim_table.json"),
    )
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--train-sample-per-dim", type=int, default=400)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--flow-max-mag", type=float, default=40.0)
    parser.add_argument("--lambda-grid", default="0,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--accept-min-proxy-gain", type=float, default=0.002)
    parser.add_argument(
        "--profile", choices=["safe", "balanced", "aggressive"], default="balanced"
    )
    parser.add_argument(
        "--max-dimensions",
        type=int,
        default=None,
        help="Optional cap on number of exact dimensions to calibrate (highest support first).",
    )
    parser.add_argument("--artifact-tag", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--repair-mode",
        choices=["best_effort", "strict"],
        default="best_effort",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    table_path = Path(args.dim_table_path).expanduser().resolve()

    require_existing_dir(train_dir, "Training images directory")
    require_existing_dir(test_dir, "Test images directory")
    if not table_path.exists():
        raise FileNotFoundError(f"Dimension table not found: {table_path}")

    if not (0.0 < args.holdout_ratio < 1.0):
        raise ValueError("--holdout-ratio must be between 0 and 1")

    lambda_grid = parse_lambda_grid(args.lambda_grid)
    if 0.0 not in lambda_grid:
        lambda_grid = [0.0] + lambda_grid

    rng = random.Random(args.seed)
    table = load_table(table_path)

    print("=== Flow-Residual CalibGuard (Exact-Dimension) ===")
    print(f"Train dir: {train_dir}")
    print(f"Test dir: {test_dir}")
    print(f"Table: {table_path}")
    print(f"Profile: {args.profile}")

    dim_entries = table.get("dimensions", {}) if isinstance(table.get("dimensions"), dict) else {}

    scored_dims: list[tuple[str, int]] = []
    for dim_key, entry in dim_entries.items():
        if not isinstance(entry, dict):
            continue
        support = int(entry.get("support", 0))
        if support < args.min_support:
            continue
        scored_dims.append((dim_key, support))

    scored_dims.sort(key=lambda item: item[1], reverse=True)
    if args.max_dimensions is not None:
        scored_dims = scored_dims[: max(args.max_dimensions, 0)]

    support_by_dim = {dim_key: support for dim_key, support in scored_dims}
    eligible_dims = [dim for dim, _ in scored_dims]

    per_dim_cap = max(args.train_sample_per_dim, 8)
    train_index = load_train_index_filtered(
        train_dir,
        target_dims=set(eligible_dims),
        per_dim_cap=per_dim_cap,
    )
    eligible_dims = [dim for dim in eligible_dims if dim in train_index and train_index[dim]]

    eligible_dims = sorted(eligible_dims)
    print(f"Eligible dimensions: {len(eligible_dims)}")

    models: dict[str, DimensionFlowModel] = {}
    start = time.time()

    for idx, dim_key in enumerate(eligible_dims, start=1):
        model = build_dimension_model(
            dim_key=dim_key,
            pairs=train_index[dim_key],
            support_total=int(support_by_dim.get(dim_key, len(train_index[dim_key]))),
            table=table,
            profile=args.profile,
            train_sample_per_dim=args.train_sample_per_dim,
            holdout_ratio=args.holdout_ratio,
            flow_max_mag=args.flow_max_mag,
            lambda_grid=lambda_grid,
            accept_min_proxy_gain=args.accept_min_proxy_gain,
            rng=rng,
        )
        models[dim_key] = model

        print(
            f"  [{idx}/{len(eligible_dims)}] {dim_key} | support={model.support} | "
            f"lambda={model.lambda_value:.3f} | gain={model.proxy_gain:.5f} | "
            f"accepted={model.accepted} ({model.reason})"
        )

    accepted_count = sum(1 for m in models.values() if m.accepted)
    print(f"Accepted dimensions: {accepted_count}/{len(models)}")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.artifact_tag}_{timestamp}" if args.artifact_tag else timestamp

    candidate_dir = ensure_dir(output_dir / f"corrected_flow_residual_dim_v1_{suffix}")
    baseline_dir = ensure_dir(output_dir / f"corrected_flow_residual_dim_base_{suffix}")

    test_files = sorted([p.name for p in test_dir.iterdir() if p.name.lower().endswith(".jpg")])
    if args.limit is not None:
        test_files = test_files[: max(args.limit, 0)]
    elif len(test_files) != 1000:
        raise RuntimeError(
            f"Expected 1000 test JPGs when --limit is not set, found {len(test_files)} in {test_dir}"
        )

    source_counts = Counter()
    parent_counts = Counter()
    total = len(test_files)
    unreadable_files: list[str] = []
    write_failures: list[str] = []

    print(f"Rendering {total} test images...")

    for idx, name in enumerate(test_files, start=1):
        image = cv2.imread(str(test_dir / name))
        if image is None:
            unreadable_files.append(name)
            source_counts["input_unreadable"] += 1
            continue

        h, w = image.shape[:2]
        dim_key = f"{w}x{h}"
        parent = classify_parent(h, w)
        parent_counts[parent] += 1

        k1, k2, _alpha, coeff_source = choose_coeffs(table, dim_key, parent, args.profile)
        model_type = choose_model_type(table, dim_key)

        base = apply_base_undistortion(image, k1, k2, model_type=model_type)

        base_write_ok = cv2.imwrite(
            str(baseline_dir / name),
            base,
            [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
        )
        if not base_write_ok:
            write_failures.append(f"base:{name}")

        dim_model = models.get(dim_key)
        if dim_model is not None and dim_model.accepted and dim_model.flow_map is not None:
            pred = apply_residual_flow(base, dim_model.flow_map, dim_model.lambda_value)
            source_counts["base_plus_residual"] += 1
        else:
            pred = base
            if dim_model is None:
                source_counts["base_only_unsupported_dim"] += 1
            else:
                source_counts[f"base_only_{dim_model.reason}"] += 1

        candidate_write_ok = cv2.imwrite(
            str(candidate_dir / name),
            pred,
            [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
        )
        if not candidate_write_ok:
            write_failures.append(f"candidate:{name}")

        if idx % 100 == 0 or idx == total:
            elapsed = time.time() - start
            print(f"  [{idx}/{total}] elapsed={elapsed:.1f}s coeff_source={coeff_source}")

    rendered_base_count = sum(1 for name in test_files if (baseline_dir / name).exists())
    rendered_candidate_count = sum(1 for name in test_files if (candidate_dir / name).exists())
    render_verification = {
        "expected": len(test_files),
        "rendered_base": rendered_base_count,
        "rendered_candidate": rendered_candidate_count,
        "input_unreadable_count": len(unreadable_files),
        "input_unreadable_sample": unreadable_files[:10],
        "write_failure_count": len(write_failures),
        "write_failure_sample": write_failures[:10],
    }

    if args.repair_mode == "strict":
        if rendered_base_count != len(test_files) or rendered_candidate_count != len(test_files):
            raise RuntimeError(
                "Strict mode requires complete rendered outputs before zipping "
                f"(expected={len(test_files)} rendered_base={rendered_base_count} "
                f"rendered_candidate={rendered_candidate_count} unreadable={len(unreadable_files)} "
                f"write_failures={len(write_failures)})"
            )
    else:
        if rendered_base_count != len(test_files) or rendered_candidate_count != len(test_files):
            print(
                "WARNING: Rendered output count mismatch; best_effort repair will backfill "
                f"(expected={len(test_files)} rendered_base={rendered_base_count} "
                f"rendered_candidate={rendered_candidate_count})"
            )

    candidate_zip = output_dir / f"submission_flow_residual_dim_v1_{suffix}.zip"
    base_zip = output_dir / f"submission_flow_residual_dim_base_{suffix}.zip"

    candidate_stats = build_zip_from_render_dirs(
        output_zip=candidate_zip,
        expected_names=test_files,
        primary_dir=candidate_dir,
        base_dir=baseline_dir,
        test_original_dir=test_dir,
        repair_mode=args.repair_mode,
    )
    base_stats = build_zip_from_render_dirs(
        output_zip=base_zip,
        expected_names=test_files,
        primary_dir=baseline_dir,
        base_dir=baseline_dir,
        test_original_dir=test_dir,
        repair_mode=args.repair_mode,
    )

    candidate_repaired_ids_path = output_dir / f"id_list_flow_residual_dim_v1_repaired_{suffix}.txt"
    base_repaired_ids_path = output_dir / f"id_list_flow_residual_dim_base_repaired_{suffix}.txt"
    write_ids(candidate_repaired_ids_path, candidate_stats.repaired_ids)
    write_ids(base_repaired_ids_path, base_stats.repaired_ids)

    manifest_path = output_dir / f"submission_flow_residual_dim_v1_{suffix}_manifest.json"

    dim_manifest: dict[str, Any] = {}
    for dim_key, model in models.items():
        dim_manifest[dim_key] = {
            "support": model.support,
            "parent_class": model.parent_class,
            "base_coeffs": {
                "k1": model.base_coeffs[0],
                "k2": model.base_coeffs[1],
                "source": model.coeff_source,
                "model_type": model.model_type,
            },
            "lambda": model.lambda_value,
            "proxy_gain": model.proxy_gain,
            "base_loss": model.base_loss,
            "best_loss": model.best_loss,
            "border_ratio": model.border_ratio,
            "flow_var": model.flow_var,
            "flow_p99": model.flow_p99,
            "flow_clip_max": model.flow_clip_max,
            "warp_risk": model.warp_risk,
            "accepted": model.accepted,
            "reason": model.reason,
        }

    manifest = {
        "method": "flow_residual_dim_v1",
        "base_profile": args.profile,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "repair_mode": args.repair_mode,
        "repair_count": int(candidate_stats.repair_count),
        "repair_by_source": {
            "base": int(candidate_stats.repair_by_source.get("base", 0)),
            "original": int(candidate_stats.repair_by_source.get("original", 0)),
        },
        "repair_failures": candidate_stats.repair_failures,
        "repaired_ids_path": str(candidate_repaired_ids_path),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "dim_table_path": str(table_path),
        "global_thresholds": {
            "min_support": int(args.min_support),
            "train_sample_per_dim": int(args.train_sample_per_dim),
            "holdout_ratio": float(args.holdout_ratio),
            "flow_max_mag": float(args.flow_max_mag),
            "lambda_grid": [float(v) for v in lambda_grid],
            "accept_min_proxy_gain": float(args.accept_min_proxy_gain),
            "border_ratio_max": float(BORDER_RATIO_MAX),
            "warp_risk_max": float(WARP_RISK_MAX),
            "flow_stability_var_max": float(FLOW_STABILITY_VAR_MAX),
            "alpha_forced": 0.0,
        },
        "dimensions": dim_manifest,
        "artifacts": {
            "candidate_zip": str(candidate_zip),
            "candidate_sha256": sha256_file(candidate_zip),
            "candidate_zip_size_bytes": candidate_zip.stat().st_size,
            "base_zip": str(base_zip),
            "base_sha256": sha256_file(base_zip),
            "base_zip_size_bytes": base_zip.stat().st_size,
            "candidate_dir": str(candidate_dir),
            "base_dir": str(baseline_dir),
            "candidate_repaired_ids_path": str(candidate_repaired_ids_path),
            "base_repaired_ids_path": str(base_repaired_ids_path),
        },
        "candidate_repair": {
            "repair_mode": args.repair_mode,
            "repair_count": int(candidate_stats.repair_count),
            "repair_by_source": {
                "base": int(candidate_stats.repair_by_source.get("base", 0)),
                "original": int(candidate_stats.repair_by_source.get("original", 0)),
            },
            "repair_failures": candidate_stats.repair_failures,
            "repaired_ids_path": str(candidate_repaired_ids_path),
            "zip_written_count": int(candidate_stats.written_count),
        },
        "base_repair": {
            "repair_mode": args.repair_mode,
            "repair_count": int(base_stats.repair_count),
            "repair_by_source": {
                "base": int(base_stats.repair_by_source.get("base", 0)),
                "original": int(base_stats.repair_by_source.get("original", 0)),
            },
            "repair_failures": base_stats.repair_failures,
            "repaired_ids_path": str(base_repaired_ids_path),
            "zip_written_count": int(base_stats.written_count),
        },
        "counts": {
            "test_images": len(test_files),
            "eligible_dimensions": len(eligible_dims),
            "accepted_dimensions": accepted_count,
            "parent_distribution": dict(parent_counts),
            "inference_source_distribution": dict(source_counts),
            "render_verification": render_verification,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.time() - start
    print("\n=== COMPLETE ===")
    print(f"Candidate zip: {candidate_zip}")
    print(f"Base zip: {base_zip}")
    print(f"Manifest: {manifest_path}")
    print(
        "Candidate repair: "
        f"mode={args.repair_mode} count={candidate_stats.repair_count} "
        f"by_source={candidate_stats.repair_by_source}"
    )
    print(
        "Base repair: "
        f"mode={args.repair_mode} count={base_stats.repair_count} "
        f"by_source={base_stats.repair_by_source}"
    )
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
