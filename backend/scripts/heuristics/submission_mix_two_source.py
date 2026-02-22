#!/usr/bin/env python3
"""Build a two-source patch zip using score deltas between base and candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from backend.config import get_config, require_existing_dir
from backend.scripts.heuristics.zip_repair import (
    ReadResult,
    build_expected_name_set,
    finalize_zip_with_verification,
    safe_read_member_from_zip,
    safe_read_original,
)


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Compose patched submission from base and candidate artifacts."
    )
    parser.add_argument("--base-zip", required=True, type=Path)
    parser.add_argument("--candidate-zip", required=True, type=Path)
    parser.add_argument("--base-score-csv", required=True, type=Path)
    parser.add_argument("--candidate-score-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--test-original-dir",
        type=Path,
        default=Path(str(cfg.test_dir)),
        help=f"Fallback test JPG directory (default: {cfg.test_dir})",
    )
    parser.add_argument(
        "--repair-mode",
        choices=["best_effort", "strict"],
        default="best_effort",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--manifest-out", default="")
    return parser.parse_args()


def resolve_existing_path(raw: Path, label: str) -> Path:
    path = raw.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_scores(csv_path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or row.get("id") or "").strip()
            score_raw = row.get("score")
            if not image_id or score_raw is None:
                continue
            try:
                scores[image_id] = float(score_raw)
            except ValueError:
                continue
    if not scores:
        raise RuntimeError(f"No score rows loaded from: {csv_path}")
    return scores


def jpg_members(zf: zipfile.ZipFile) -> list[str]:
    names = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir() and info.filename.lower().endswith(".jpg")
    ]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate JPG names detected in zip")
    return names


def validate_zip_member_set(zip_path: Path, expected: set[str], label: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(jpg_members(zf))
    if names != expected:
        missing = sorted(expected - names)[:10]
        extra = sorted(names - expected)[:10]
        raise RuntimeError(
            f"{label} member mismatch. missing={len(expected - names)} extra={len(names - expected)} "
            f"missing_sample={missing} extra_sample={extra}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def mean_for_scores(ids: list[str], scores: dict[str, float]) -> float:
    return sum(scores[i] for i in ids) / max(len(ids), 1)


def main() -> None:
    args = parse_args()

    base_zip = resolve_existing_path(args.base_zip, "base zip")
    candidate_zip = resolve_existing_path(args.candidate_zip, "candidate zip")
    base_score_csv = resolve_existing_path(args.base_score_csv, "base score CSV")
    candidate_score_csv = resolve_existing_path(
        args.candidate_score_csv, "candidate score CSV"
    )
    test_original_dir = require_existing_dir(
        args.test_original_dir.expanduser().resolve(),
        "test originals directory",
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag.strip()
    if not tag:
        raise RuntimeError("--tag must be non-empty")

    base_scores = load_scores(base_score_csv)
    candidate_scores = load_scores(candidate_score_csv)

    base_names, expected_member_set = build_expected_name_set(base_zip)
    if len(base_names) != 1000:
        raise RuntimeError(f"Base zip must contain exactly 1000 JPG files, found {len(base_names)}")

    validate_zip_member_set(candidate_zip, expected_member_set, "candidate zip")

    ids = sorted(Path(name).stem for name in base_names)
    if (set(ids) - set(base_scores)) or (set(ids) - set(candidate_scores)):
        raise RuntimeError("Score CSVs must contain all 1000 image IDs from zips")

    replacements: dict[str, bool] = {}
    replacement_ids: list[str] = []
    for image_id in ids:
        use_candidate = candidate_scores[image_id] > (base_scores[image_id] + args.margin)
        replacements[image_id] = use_candidate
        if use_candidate:
            replacement_ids.append(image_id)

    output_name = args.output_name.strip() or f"submission_flow_residual_dim_patch_v1_{tag}.zip"
    out_zip = output_dir / output_name

    with (
        zipfile.ZipFile(base_zip, "r") as base_zf,
        zipfile.ZipFile(candidate_zip, "r") as cand_zf,
    ):
        def primary_reader(name: str) -> ReadResult:
            image_id = Path(name).stem
            if replacements.get(image_id, False):
                return safe_read_member_from_zip(cand_zf, name, source="candidate")
            return safe_read_member_from_zip(base_zf, name, source="base")

        def base_reader(name: str) -> ReadResult:
            return safe_read_member_from_zip(base_zf, name, source="base")

        def original_reader(name: str) -> ReadResult:
            return safe_read_original(test_original_dir, name)

        repair_stats = finalize_zip_with_verification(
            output_zip=out_zip,
            expected_names=base_names,
            primary_reader=primary_reader,
            base_reader=base_reader,
            original_reader=original_reader,
            repair_mode=args.repair_mode,
            expected_count=len(base_names),
        )

    mixed_mean = 0.0
    for image_id in ids:
        mixed_mean += candidate_scores[image_id] if replacements[image_id] else base_scores[image_id]
    mixed_mean /= max(len(ids), 1)

    base_mean = mean_for_scores(ids, base_scores)
    candidate_mean = mean_for_scores(ids, candidate_scores)

    manifest_path = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out.strip()
        else output_dir / f"submission_flow_residual_dim_patch_v1_{tag}_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    repaired_ids_path = output_dir / f"id_list_flow_residual_patch_repaired_{tag}.txt"
    repaired_ids_path.write_text("\n".join(repair_stats.repaired_ids) + ("\n" if repair_stats.repaired_ids else ""), encoding="utf-8")

    manifest = {
        "method": "submission_mix_two_source.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "margin": float(args.margin),
        "repair_mode": args.repair_mode,
        "repair_count": int(repair_stats.repair_count),
        "repair_by_source": {
            "base": int(repair_stats.repair_by_source.get("base", 0)),
            "original": int(repair_stats.repair_by_source.get("original", 0)),
        },
        "repair_failures": repair_stats.repair_failures,
        "repaired_ids_path": str(repaired_ids_path.resolve()),
        "inputs": {
            "base_zip": str(base_zip),
            "candidate_zip": str(candidate_zip),
            "base_score_csv": str(base_score_csv),
            "candidate_score_csv": str(candidate_score_csv),
            "test_original_dir": str(test_original_dir),
        },
        "outputs": {
            "zip": str(out_zip),
            "sha256": sha256_file(out_zip),
            "zip_size_bytes": out_zip.stat().st_size,
            "manifest": str(manifest_path),
        },
        "counts": {
            "image_count": len(ids),
            "replacement_count": len(replacement_ids),
            "replacement_rate": float(len(replacement_ids) / max(len(ids), 1)),
            "zip_written_count": int(repair_stats.written_count),
        },
        "predicted_means": {
            "base": float(base_mean),
            "candidate": float(candidate_mean),
            "mixed": float(mixed_mean),
            "uplift_vs_base": float(mixed_mean - base_mean),
        },
        "replacement_ids_path": str(
            (output_dir / f"id_list_flow_residual_patch_{tag}.txt").resolve()
        ),
    }

    replacement_ids_path = Path(manifest["replacement_ids_path"])
    replacement_ids_path.write_text("\n".join(replacement_ids) + "\n", encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Output zip: {out_zip}")
    print(f"Manifest: {manifest_path}")
    print(f"Replacements: {len(replacement_ids)}/{len(ids)}")
    print(
        "Repair: "
        f"mode={args.repair_mode} count={repair_stats.repair_count} "
        f"by_source={repair_stats.repair_by_source}"
    )
    print(
        "Predicted means: "
        f"base={base_mean:.6f} candidate={candidate_mean:.6f} mixed={mixed_mean:.6f}"
    )


if __name__ == "__main__":
    main()
