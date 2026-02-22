#!/usr/bin/env python3
"""Run CalibGuard-Dim inference with safe/balanced/aggressive guardrail profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.core.undistort_ops import (
    resolve_border_mode,
    resolve_interpolation,
    undistort_via_maps,
)


HIGH_RISK_PARENTS = {"heavy_crop", "portrait_cropped"}


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


def _coeff_pair(source: dict[str, Any], key: str) -> tuple[float, float] | None:
    node = source.get(key)
    if not isinstance(node, dict):
        return None
    if "k1" not in node or "k2" not in node:
        return None
    return float(node["k1"]), float(node["k2"])


def _coeff_triplet(
    source: dict[str, Any], key: str, default_alpha: float = 0.0
) -> tuple[float, float, float] | None:
    node = source.get(key)
    if not isinstance(node, dict):
        return None
    if "k1" not in node or "k2" not in node:
        return None
    alpha = float(node.get("alpha", default_alpha))
    return float(node["k1"]), float(node["k2"]), alpha


def choose_coeffs(
    table: dict[str, Any],
    dim_key: str,
    parent_class: str,
    profile: str,
) -> tuple[float, float, float, str]:
    dimensions = table.get("dimensions", {})
    parent_classes = table.get("parent_classes", {})
    global_fallback = table.get("global_fallback", {"k1": -0.17, "k2": 0.35})

    entry = dimensions.get(dim_key)
    parent_entry = parent_classes.get(parent_class, {})
    default_alpha = 0.0

    def fallback_safe() -> tuple[float, float, float, str]:
        parent_safe = _coeff_triplet(parent_entry, "safe", default_alpha=default_alpha)
        if parent_safe is not None:
            return parent_safe[0], parent_safe[1], parent_safe[2], "parent_safe"
        return (
            float(global_fallback["k1"]),
            float(global_fallback["k2"]),
            default_alpha,
            "global_fallback",
        )

    def fallback_primary() -> tuple[float, float, float, str]:
        parent_primary = _coeff_triplet(
            parent_entry, "primary", default_alpha=default_alpha
        )
        if parent_primary is not None:
            return (
                parent_primary[0],
                parent_primary[1],
                parent_primary[2],
                "parent_primary",
            )
        return (
            float(global_fallback["k1"]),
            float(global_fallback["k2"]),
            default_alpha,
            "global_fallback",
        )

    if entry is None:
        if profile == "aggressive":
            return fallback_safe()
        if profile == "balanced" and parent_class in HIGH_RISK_PARENTS:
            return fallback_safe()
        if profile == "safe":
            return fallback_safe()
        return fallback_primary()

    primary = _coeff_triplet(entry, "primary", default_alpha=default_alpha)
    safe = _coeff_triplet(entry, "safe", default_alpha=default_alpha)
    support = int(entry.get("support", 0))
    guardrails = (
        entry.get("guardrails", {}) if isinstance(entry.get("guardrails"), dict) else {}
    )
    force_safe = bool(guardrails.get("force_safe", False))

    if safe is None:
        safe_fallback = fallback_safe()
        safe = (safe_fallback[0], safe_fallback[1], safe_fallback[2])
    if primary is None:
        primary = safe

    if profile == "safe":
        return safe[0], safe[1], safe[2], "dim_safe"

    if profile == "balanced":
        if force_safe:
            return safe[0], safe[1], safe[2], "dim_safe_guardrail"
        return primary[0], primary[1], primary[2], "dim_primary"

    # aggressive profile
    if force_safe:
        return safe[0], safe[1], safe[2], "dim_safe_guardrail"
    if support >= 100:
        return primary[0], primary[1], primary[2], "dim_primary"
    return safe[0], safe[1], safe[2], "dim_safe_low_support"


def choose_model_type(table: dict[str, Any], dim_key: str) -> str:
    dimensions = table.get("dimensions", {})
    entry = dimensions.get(dim_key, {}) if isinstance(dimensions, dict) else {}
    if isinstance(entry, dict):
        model_node = entry.get("model", {})
        if isinstance(model_node, dict):
            model_type = str(model_node.get("type", "")).strip().lower()
            if model_type in {"brown", "rational"}:
                return model_type
    defaults = table.get("model_defaults", {})
    if isinstance(defaults, dict):
        model_type = str(defaults.get("type", "")).strip().lower()
        if model_type in {"brown", "rational"}:
            return model_type
    return "brown"


def choose_render_settings(
    table: dict[str, Any],
    dim_key: str,
    cli_interp: str | None,
    cli_border_mode: str | None,
) -> tuple[str, str]:
    table_defaults = (
        table.get("render_defaults", {})
        if isinstance(table.get("render_defaults"), dict)
        else {}
    )
    interp = str(table_defaults.get("interp", "linear")).strip().lower()
    border_mode = str(table_defaults.get("border_mode", "constant")).strip().lower()

    dimensions = table.get("dimensions", {})
    entry = dimensions.get(dim_key, {}) if isinstance(dimensions, dict) else {}
    if isinstance(entry, dict):
        render = entry.get("render", {})
        if isinstance(render, dict):
            interp = str(render.get("interp", interp)).strip().lower()
            border_mode = str(render.get("border_mode", border_mode)).strip().lower()

    if cli_interp is not None:
        interp = cli_interp
    if cli_border_mode is not None:
        border_mode = cli_border_mode
    interp = resolve_interpolation(interp)[0]
    border_mode = resolve_border_mode(border_mode)[0]
    return interp, border_mode


def main() -> None:
    cfg = get_config()
    default_table = (
        cfg.repo_root / "backend/scripts/heuristics/calibguard_dim_table.json"
    )
    default_manifest = (
        cfg.repo_root / "backend/scripts/heuristics/calibguard_dim_manifest.json"
    )

    parser = argparse.ArgumentParser(description="CalibGuard-Dim heuristic inference")
    parser.add_argument("--test-dir", default=str(cfg.test_dir))
    parser.add_argument("--table", default=str(default_table))
    parser.add_argument(
        "--profile", choices=["safe", "balanced", "aggressive"], default="safe"
    )
    parser.add_argument("--output-dir", default=str(cfg.output_root))
    parser.add_argument("--artifact-tag", default="calibguard_dim")
    parser.add_argument("--manifest", default=str(default_manifest))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--interp",
        choices=["linear", "cubic", "lanczos4"],
        default=None,
        help="Optional remap interpolation override (otherwise use table/defaults).",
    )
    parser.add_argument(
        "--border-mode",
        choices=["constant", "reflect", "replicate"],
        default=None,
        help="Optional remap border mode override (otherwise use table/defaults).",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    table_path = Path(args.table).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    require_existing_dir(test_dir, "Test images directory")
    if not table_path.exists():
        raise FileNotFoundError(f"CalibGuard table not found: {table_path}")

    with table_path.open("r", encoding="utf-8") as f:
        table = json.load(f)

    table_hash = _sha256_file(table_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
    if args.limit:
        test_files = test_files[: args.limit]

    corrected_dir = ensure_dir(
        output_dir / f"corrected_calibguard_dim_{args.profile}_{timestamp}"
    )

    route_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    dim_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    interp_counts: Counter[str] = Counter()
    border_counts: Counter[str] = Counter()

    t0 = time.time()
    print(f"=== CalibGuard-Dim ({args.profile}) ===")
    print(f"Table: {table_path}")
    print(f"Table SHA256: {table_hash}")
    print(f"Test images: {len(test_files)}")

    for i, fname in enumerate(test_files, start=1):
        image = cv2.imread(str(test_dir / fname))
        if image is None:
            print(f"  WARNING: Could not read {fname}")
            continue

        h, w = image.shape[:2]
        dim_key = f"{w}x{h}"
        parent = classify_parent(h, w)

        k1, k2, alpha, route = choose_coeffs(table, dim_key, parent, args.profile)
        model_type = choose_model_type(table, dim_key)
        interp_name, border_name = choose_render_settings(
            table,
            dim_key,
            cli_interp=args.interp,
            cli_border_mode=args.border_mode,
        )
        corrected = apply_undistortion(
            image,
            k1,
            k2,
            alpha=alpha,
            interpolation=interp_name,
            border_mode=border_name,
            model_type=model_type,
        )
        if corrected.shape[:2] != (h, w):
            corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(
            str(corrected_dir / fname), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        route_counts[route] += 1
        parent_counts[parent] += 1
        dim_counts[dim_key] += 1
        model_counts[model_type] += 1
        interp_counts[interp_name] += 1
        border_counts[border_name] += 1

        if i % 100 == 0 or i == len(test_files):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(test_files) - i)
            print(
                f"  [{i}/{len(test_files)}] ETA: {eta:.0f}s | "
                f"dim={dim_key} parent={parent} route={route} model={model_type}"
            )

    zip_name = f"submission_{args.artifact_tag}_{args.profile}_{timestamp}.zip"
    zip_path = output_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fname in test_files:
            fpath = corrected_dir / fname
            if fpath.exists():
                zf.write(fpath, fname)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0

    print("\n=== RESULTS ===")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Zip: {zip_path} ({zip_size_mb:.1f} MB)")
    print("Route distribution:")
    for route, count in route_counts.most_common():
        pct = count / max(len(test_files), 1) * 100
        print(f"  {route:>20}: {count} ({pct:.1f}%)")

    print("Parent class distribution:")
    for parent, count in parent_counts.most_common():
        pct = count / max(len(test_files), 1) * 100
        print(f"  {parent:>20}: {count} ({pct:.1f}%)")
    print("Render distribution:")
    for interp, count in interp_counts.most_common():
        pct = count / max(len(test_files), 1) * 100
        print(f"  interp={interp:>10}: {count} ({pct:.1f}%)")
    for border, count in border_counts.most_common():
        pct = count / max(len(test_files), 1) * 100
        print(f"  border={border:>10}: {count} ({pct:.1f}%)")
    print("Model distribution:")
    for model, count in model_counts.most_common():
        pct = count / max(len(test_files), 1) * 100
        print(f"  {model:>20}: {count} ({pct:.1f}%)")

    ensure_dir(manifest_path.parent)
    manifest = load_manifest(manifest_path)
    manifest["runs"].append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "profile": args.profile,
            "table_path": str(table_path),
            "table_sha256": table_hash,
            "artifact_path": str(zip_path),
            "artifact_tag": args.artifact_tag,
            "test_images": len(test_files),
            "route_counts": dict(route_counts),
            "parent_counts": dict(parent_counts),
            "model_counts": dict(model_counts),
            "interp_counts": dict(interp_counts),
            "border_mode_counts": dict(border_counts),
            "cli_interp_override": args.interp,
            "cli_border_mode_override": args.border_mode,
        }
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Manifest updated: {manifest_path}")
    print(f"\n>>> Upload {zip_path} to https://bounty.autohdr.com <<<")


if __name__ == "__main__":
    main()
