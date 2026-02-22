#!/usr/bin/env python3
"""Build deterministic V4 mix-batch submission candidates from scored artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from backend.config import get_config
from backend.scripts.heuristics.zip_repair import (
    ReadResult,
    ZipFinalizeStats,
    build_expected_name_set,
    finalize_zip_with_verification,
    safe_read_member_from_zip,
    safe_read_original,
)

SOURCE_BASE = "base"
SOURCE_ZERO = "zero"
SOURCE_T5 = "t5"
SOURCE_ORIGINAL = "original"


@dataclass(frozen=True)
class CandidateDef:
    """Candidate zip recipe."""

    name: str
    output_name: str
    description: str
    rule_expression: str
    patches: dict[str, str]  # image_id -> source name


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Compose deterministic V4 mix-batch candidates from base/zero/t5 scored artifacts."
    )
    parser.add_argument("--base-zip", required=True, type=Path)
    parser.add_argument("--zero-zip", required=True, type=Path)
    parser.add_argument("--t5-zip", required=True, type=Path)
    parser.add_argument("--base-score-csv", required=True, type=Path)
    parser.add_argument("--zero-score-csv", required=True, type=Path)
    parser.add_argument("--t5-score-csv", required=True, type=Path)
    parser.add_argument(
        "--test-original-dir",
        type=Path,
        default=Path(str(cfg.test_dir)),
        help=f"Fallback test JPG directory (default: {cfg.test_dir})",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="Batch tag used in output names.")
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument(
        "--ledger-out",
        type=Path,
        default=None,
        help="Optional path for candidate run ledger JSON.",
    )
    parser.add_argument(
        "--zpos-delta-thresholds",
        default="0.0",
        help="Comma-separated zero-over-base delta thresholds (e.g. 0.0,0.1,0.25,0.5).",
    )
    parser.add_argument(
        "--orig-fallback-thresholds",
        default="0.0,1.0",
        help="Comma-separated base-score thresholds for original-image fallback.",
    )
    parser.add_argument(
        "--t5-margin-thresholds",
        default="0.0",
        help="Comma-separated margins for t5 > max(base,zero) + margin.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional cap on number of emitted candidates after deterministic ordering (0 = no cap).",
    )
    parser.add_argument(
        "--include-legacy-candidates",
        dest="include_legacy_candidates",
        action="store_true",
        default=True,
        help="Include legacy named candidates (zpos, zpos_t5plus, zle0, zle1, znonneg).",
    )
    parser.add_argument(
        "--no-legacy-candidates",
        dest="include_legacy_candidates",
        action="store_false",
        help="Disable legacy named candidates.",
    )
    parser.add_argument(
        "--repair-mode",
        choices=["best_effort", "strict"],
        default="best_effort",
    )
    return parser.parse_args()


def parse_thresholds(raw: str, label: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise RuntimeError(f"Invalid {label} threshold '{item}'") from exc
    if not values:
        raise RuntimeError(f"{label} thresholds cannot be empty")
    return sorted(set(values))


def threshold_slug(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text.replace("-", "neg").replace(".", "p")


def resolve_existing_path(raw: Path, label: str) -> Path:
    path = raw.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_existing_dir(raw: Path, label: str) -> Path:
    path = resolve_existing_path(raw, label)
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
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


def _jpg_members(zf: zipfile.ZipFile) -> list[str]:
    names = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir() and info.filename.lower().endswith(".jpg")
    ]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate JPG names detected in zip.")
    return names


def load_base_members(base_zip: Path) -> list[str]:
    names, _ = build_expected_name_set(base_zip)
    if len(names) != 1000:
        raise RuntimeError(f"Base zip must contain exactly 1000 JPGs, found {len(names)}")
    return names


def validate_zip_member_set(zip_path: Path, expected: set[str], label: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(_jpg_members(zf))
    if names != expected:
        missing = sorted(expected - names)[:10]
        extra = sorted(names - expected)[:10]
        raise RuntimeError(
            f"{label} member mismatch. missing={len(expected - names)} extra={len(names - expected)} "
            f"missing_sample={missing} extra_sample={extra}"
        )


def detect_replacement_ids(
    zero_zip: Path,
    test_original_dir: Path,
    expected_names: Iterable[str],
) -> set[str]:
    replaced: set[str] = set()
    with zipfile.ZipFile(zero_zip, "r") as zf:
        for name in expected_names:
            original_path = test_original_dir / name
            if not original_path.exists():
                raise FileNotFoundError(f"Missing test original referenced by zip: {original_path}")
            if zf.read(name) == original_path.read_bytes():
                replaced.add(Path(name).stem)
    return replaced


def write_id_list(output_dir: Path, tag: str, list_name: str, ids: Iterable[str]) -> Path:
    path = output_dir / f"id_list_{list_name}_{tag}.txt"
    sorted_ids = sorted(ids)
    body = "\n".join(sorted_ids)
    if body:
        body += "\n"
    path.write_text(body, encoding="utf-8")
    return path


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_candidate_zip(
    *,
    candidate: CandidateDef,
    output_path: Path,
    base_zip: Path,
    zero_zip: Path,
    t5_zip: Path,
    test_original_dir: Path,
    base_names: list[str],
    repair_mode: str,
) -> ZipFinalizeStats:
    with (
        zipfile.ZipFile(base_zip, "r") as base_zf,
        zipfile.ZipFile(zero_zip, "r") as zero_zf,
        zipfile.ZipFile(t5_zip, "r") as t5_zf,
    ):
        def primary_reader(name: str) -> ReadResult:
            image_id = Path(name).stem
            source = candidate.patches.get(image_id, SOURCE_BASE)
            if source == SOURCE_ZERO:
                return safe_read_member_from_zip(zero_zf, name, source="zero")
            if source == SOURCE_T5:
                return safe_read_member_from_zip(t5_zf, name, source="t5")
            if source == SOURCE_ORIGINAL:
                return safe_read_original(test_original_dir, name)
            if source == SOURCE_BASE:
                return safe_read_member_from_zip(base_zf, name, source="base")
            return ReadResult(
                payload=None,
                source="primary",
                error=f"unsupported source '{source}' in candidate {candidate.name}",
            )

        def base_reader(name: str) -> ReadResult:
            return safe_read_member_from_zip(base_zf, name, source="base")

        def original_reader(name: str) -> ReadResult:
            return safe_read_original(test_original_dir, name)

        return finalize_zip_with_verification(
            output_zip=output_path,
            expected_names=base_names,
            primary_reader=primary_reader,
            base_reader=base_reader,
            original_reader=original_reader,
            repair_mode=repair_mode,
            expected_count=len(base_names),
        )


def predicted_mean_for_candidate(
    *,
    candidate: CandidateDef,
    ordered_ids: list[str],
    base_scores: dict[str, float],
    zero_scores: dict[str, float],
    t5_scores: dict[str, float],
) -> float:
    total = 0.0
    for image_id in ordered_ids:
        source = candidate.patches.get(image_id, SOURCE_BASE)
        if source in {SOURCE_ZERO, SOURCE_ORIGINAL}:
            total += zero_scores[image_id]
        elif source == SOURCE_T5:
            total += t5_scores[image_id]
        else:
            total += base_scores[image_id]
    return total / max(len(ordered_ids), 1)


def compose_patch_map(
    *,
    delta_ids: set[str] | None = None,
    orig_ids: set[str] | None = None,
    t5_ids: set[str] | None = None,
) -> dict[str, str]:
    patches: dict[str, str] = {}

    # Original fallback is a hard guardrail and has highest priority.
    for image_id in sorted(orig_ids or set()):
        patches[image_id] = SOURCE_ORIGINAL

    for image_id in sorted(delta_ids or set()):
        if patches.get(image_id) != SOURCE_ORIGINAL:
            patches[image_id] = SOURCE_ZERO

    for image_id in sorted(t5_ids or set()):
        if patches.get(image_id) != SOURCE_ORIGINAL:
            patches[image_id] = SOURCE_T5

    return patches


def default_ledger_path(manifest_path: Path) -> Path:
    stem = manifest_path.stem
    if stem.endswith("_manifest"):
        stem = stem[: -len("_manifest")]
    return manifest_path.with_name(f"{stem}_ledger.json")


def main() -> None:
    args = parse_args()

    base_zip = resolve_existing_path(args.base_zip, "base zip")
    zero_zip = resolve_existing_path(args.zero_zip, "zero zip")
    t5_zip = resolve_existing_path(args.t5_zip, "t5 zip")
    base_score_csv = resolve_existing_path(args.base_score_csv, "base score CSV")
    zero_score_csv = resolve_existing_path(args.zero_score_csv, "zero score CSV")
    t5_score_csv = resolve_existing_path(args.t5_score_csv, "t5 score CSV")
    test_original_dir = resolve_existing_dir(args.test_original_dir, "test originals directory")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_out.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = (
        args.ledger_out.expanduser().resolve()
        if args.ledger_out is not None
        else default_ledger_path(manifest_path)
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    tag = args.tag.strip()
    if not tag:
        raise RuntimeError("--tag must be non-empty")

    zpos_thresholds = parse_thresholds(args.zpos_delta_thresholds, "zpos-delta")
    orig_thresholds = parse_thresholds(args.orig_fallback_thresholds, "orig-fallback")
    t5_thresholds = parse_thresholds(args.t5_margin_thresholds, "t5-margin")

    base_scores = load_scores(base_score_csv)
    zero_scores = load_scores(zero_score_csv)
    t5_scores = load_scores(t5_score_csv)

    base_names = load_base_members(base_zip)
    base_name_set = set(base_names)

    validate_zip_member_set(zero_zip, base_name_set, "zero zip")
    validate_zip_member_set(t5_zip, base_name_set, "t5 zip")

    base_ids = {Path(name).stem for name in base_names}
    score_ids = set(base_scores) & set(zero_scores) & set(t5_scores)
    missing_score_ids = sorted(base_ids - score_ids)
    if missing_score_ids:
        raise RuntimeError(
            f"Scores missing for {len(missing_score_ids)} image IDs. sample={missing_score_ids[:10]}"
        )
    ordered_ids = sorted(base_ids)

    replacement_ids = detect_replacement_ids(zero_zip, test_original_dir, base_names)

    delta_ids_map = {
        t: {
            image_id
            for image_id in ordered_ids
            if (zero_scores[image_id] - base_scores[image_id]) >= t
        }
        for t in zpos_thresholds
    }
    orig_ids_map = {
        t: {image_id for image_id in replacement_ids if base_scores[image_id] <= t}
        for t in orig_thresholds
    }
    t5_ids_map = {
        m: {
            image_id
            for image_id in ordered_ids
            if t5_scores[image_id] > (max(base_scores[image_id], zero_scores[image_id]) + m)
        }
        for m in t5_thresholds
    }
    z_nonneg_ids = {
        image_id
        for image_id in replacement_ids
        if zero_scores[image_id] >= base_scores[image_id]
    }

    id_list_paths: dict[str, str] = {
        "replacement_ids": str(write_id_list(output_dir, tag, "replacement_ids", replacement_ids)),
        "z_nonneg_replacement_ids": str(
            write_id_list(output_dir, tag, "z_nonneg_replacement_ids", z_nonneg_ids)
        ),
    }
    for t, ids in delta_ids_map.items():
        id_list_paths[f"zero_over_base_delta_ge_{threshold_slug(t)}"] = str(
            write_id_list(output_dir, tag, f"zero_over_base_delta_ge_{threshold_slug(t)}", ids)
        )
    for t, ids in orig_ids_map.items():
        id_list_paths[f"orig_fallback_base_le_{threshold_slug(t)}"] = str(
            write_id_list(output_dir, tag, f"orig_fallback_base_le_{threshold_slug(t)}", ids)
        )
    for m, ids in t5_ids_map.items():
        id_list_paths[f"t5_margin_gt_{threshold_slug(m)}"] = str(
            write_id_list(output_dir, tag, f"t5_margin_gt_{threshold_slug(m)}", ids)
        )

    candidates: list[CandidateDef] = []

    def add_candidate(
        name: str,
        description: str,
        rule_expression: str,
        patches: dict[str, str],
    ) -> None:
        candidates.append(
            CandidateDef(
                name=name,
                output_name=f"submission_v4_mix_{name}_{tag}.zip",
                description=description,
                rule_expression=rule_expression,
                patches=patches,
            )
        )

    if args.include_legacy_candidates:
        legacy_delta = delta_ids_map.get(0.0, set())
        legacy_t5 = t5_ids_map.get(0.0, set())
        add_candidate(
            "zpos",
            "Legacy: patch IDs where zero_over_base_delta >= 0.0, source=zero",
            "zero_over_base_delta >= 0.0",
            compose_patch_map(delta_ids=legacy_delta),
        )
        add_candidate(
            "zpos_t5plus",
            "Legacy: zpos baseline with t5 overrides where t5 > max(base,zero) + 0.0",
            "zero_over_base_delta >= 0.0; t5 > max(base,zero) + 0.0",
            compose_patch_map(delta_ids=legacy_delta, t5_ids=legacy_t5),
        )
        add_candidate(
            "zle0",
            "Legacy: replacement IDs where base <= 0.0 fallback to original",
            "replacement_id and base <= 0.0",
            compose_patch_map(orig_ids=orig_ids_map.get(0.0, set())),
        )
        add_candidate(
            "zle1",
            "Legacy: replacement IDs where base <= 1.0 fallback to original",
            "replacement_id and base <= 1.0",
            compose_patch_map(orig_ids=orig_ids_map.get(1.0, set())),
        )
        add_candidate(
            "znonneg",
            "Legacy: replacement IDs where zero score is non-regressive, source=zero",
            "replacement_id and zero >= base",
            compose_patch_map(delta_ids=z_nonneg_ids),
        )

    for t in zpos_thresholds:
        add_candidate(
            f"delta_ge_{threshold_slug(t)}",
            f"Patch IDs where zero_over_base_delta >= {t}, source=zero",
            f"zero_over_base_delta >= {t}",
            compose_patch_map(delta_ids=delta_ids_map[t]),
        )

    for t in zpos_thresholds:
        for m in t5_thresholds:
            add_candidate(
                f"delta_ge_{threshold_slug(t)}__t5m_gt_{threshold_slug(m)}",
                f"delta>= {t} with t5 margin override > {m}",
                f"zero_over_base_delta >= {t}; t5 > max(base,zero) + {m}",
                compose_patch_map(delta_ids=delta_ids_map[t], t5_ids=t5_ids_map[m]),
            )

    for t in orig_thresholds:
        add_candidate(
            f"orig_le_{threshold_slug(t)}",
            f"Fallback to original for replacement IDs where base <= {t}",
            f"replacement_id and base <= {t}",
            compose_patch_map(orig_ids=orig_ids_map[t]),
        )

    for o in orig_thresholds:
        for d in zpos_thresholds:
            add_candidate(
                f"orig_le_{threshold_slug(o)}__delta_ge_{threshold_slug(d)}",
                f"orig<= {o} fallback plus delta>= {d} zero patches",
                f"replacement_id and base <= {o}; zero_over_base_delta >= {d}",
                compose_patch_map(orig_ids=orig_ids_map[o], delta_ids=delta_ids_map[d]),
            )

    for o in orig_thresholds:
        for d in zpos_thresholds:
            for m in t5_thresholds:
                add_candidate(
                    f"orig_le_{threshold_slug(o)}__delta_ge_{threshold_slug(d)}__t5m_gt_{threshold_slug(m)}",
                    f"orig<= {o}, delta>= {d}, and t5 margin override > {m}",
                    f"replacement_id and base <= {o}; zero_over_base_delta >= {d}; "
                    f"t5 > max(base,zero) + {m}",
                    compose_patch_map(
                        orig_ids=orig_ids_map[o],
                        delta_ids=delta_ids_map[d],
                        t5_ids=t5_ids_map[m],
                    ),
                )

    unique_candidates: list[CandidateDef] = []
    seen_patch_keys: set[tuple[tuple[str, str], ...]] = set()
    for candidate in candidates:
        key = tuple(sorted(candidate.patches.items()))
        if key in seen_patch_keys:
            continue
        seen_patch_keys.add(key)
        unique_candidates.append(candidate)

    unique_candidates.sort(key=lambda c: (len(c.patches), c.name))
    if args.max_candidates > 0:
        unique_candidates = unique_candidates[: args.max_candidates]

    candidate_artifacts: dict[str, str] = {}
    predicted_means: dict[str, float] = {}
    sha256: dict[str, str] = {}
    zip_size_bytes: dict[str, int] = {}
    replacement_counts: dict[str, int] = {}
    repair_counts: dict[str, int] = {}
    repair_by_source: dict[str, dict[str, int]] = {}
    repair_failures: dict[str, list[str]] = {}
    repaired_ids_paths: dict[str, str] = {}
    candidate_defs_out: list[dict[str, object]] = []
    ledger_entries: list[dict[str, object]] = []

    parent_artifacts = {
        "base_zip": str(base_zip),
        "zero_zip": str(zero_zip),
        "t5_zip": str(t5_zip),
        "base_score_csv": str(base_score_csv),
        "zero_score_csv": str(zero_score_csv),
        "t5_score_csv": str(t5_score_csv),
        "test_original_dir": str(test_original_dir),
    }

    for candidate in unique_candidates:
        output_path = output_dir / candidate.output_name
        expected_replacements = len(candidate.patches)
        repair_stats = _build_candidate_zip(
            candidate=candidate,
            output_path=output_path,
            base_zip=base_zip,
            zero_zip=zero_zip,
            t5_zip=t5_zip,
            test_original_dir=test_original_dir,
            base_names=base_names,
            repair_mode=args.repair_mode,
        )

        predicted_mean = predicted_mean_for_candidate(
            candidate=candidate,
            ordered_ids=ordered_ids,
            base_scores=base_scores,
            zero_scores=zero_scores,
            t5_scores=t5_scores,
        )
        output_hash = compute_sha256(output_path)

        candidate_artifacts[candidate.name] = str(output_path)
        predicted_means[candidate.name] = round(predicted_mean, 8)
        sha256[candidate.name] = output_hash
        zip_size_bytes[candidate.name] = output_path.stat().st_size
        replacement_counts[candidate.name] = expected_replacements
        repair_counts[candidate.name] = int(repair_stats.repair_count)
        repair_by_source[candidate.name] = {
            "base": int(repair_stats.repair_by_source.get("base", 0)),
            "original": int(repair_stats.repair_by_source.get("original", 0)),
        }
        repair_failures[candidate.name] = list(repair_stats.repair_failures)
        repaired_ids_path = write_id_list(
            output_dir,
            tag,
            f"{candidate.name}_repaired_ids",
            repair_stats.repaired_ids,
        )
        repaired_ids_paths[candidate.name] = str(repaired_ids_path)

        per_source_counts: dict[str, int] = {}
        for source in candidate.patches.values():
            per_source_counts[source] = per_source_counts.get(source, 0) + 1

        candidate_defs_out.append(
            {
                "name": candidate.name,
                "output_name": candidate.output_name,
                "description": candidate.description,
                "rule_expression": candidate.rule_expression,
                "replacement_count": expected_replacements,
                "replacement_sources": per_source_counts,
                "repair_count": int(repair_stats.repair_count),
                "repair_by_source": {
                    "base": int(repair_stats.repair_by_source.get("base", 0)),
                    "original": int(repair_stats.repair_by_source.get("original", 0)),
                },
            }
        )
        ledger_entries.append(
            {
                "candidate_name": candidate.name,
                "artifact_path": str(output_path),
                "parent_artifacts": parent_artifacts,
                "rule_expression": candidate.rule_expression,
                "replacement_count": expected_replacements,
                "repair_mode": args.repair_mode,
                "repair_count": int(repair_stats.repair_count),
                "repair_by_source": {
                    "base": int(repair_stats.repair_by_source.get("base", 0)),
                    "original": int(repair_stats.repair_by_source.get("original", 0)),
                },
                "sha256": output_hash,
                "bounty_request_id": None,
                "kaggle_submission_timestamp": None,
                "kaggle_public_score": None,
            }
        )

        print(
            f"[{candidate.name}] replacements={expected_replacements} "
            f"predicted_mean={predicted_mean:.6f} repairs={repair_stats.repair_count} "
            f"sha256={output_hash[:12]}..."
        )

    manifest = {
        "batch_tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repair_mode": args.repair_mode,
        "repair_count": repair_counts,
        "repair_by_source": repair_by_source,
        "repair_failures": repair_failures,
        "repaired_ids_path": repaired_ids_paths,
        "inputs": parent_artifacts,
        "thresholds": {
            "zpos_delta_thresholds": zpos_thresholds,
            "orig_fallback_thresholds": orig_thresholds,
            "t5_margin_thresholds": t5_thresholds,
        },
        "candidate_defs": candidate_defs_out,
        "candidate_artifacts": candidate_artifacts,
        "predicted_means": predicted_means,
        "sha256": sha256,
        "zip_size_bytes": zip_size_bytes,
        "replacement_counts": replacement_counts,
        "id_list_paths": id_list_paths,
        "diagnostics": {
            "base_member_count": len(base_names),
            "replacement_ids_count": len(replacement_ids),
            "emitted_candidates": len(unique_candidates),
            "z_nonneg_replacement_ids_count": len(z_nonneg_ids),
            "delta_set_sizes": {str(k): len(v) for k, v in delta_ids_map.items()},
            "orig_set_sizes": {str(k): len(v) for k, v in orig_ids_map.items()},
            "t5_set_sizes": {str(k): len(v) for k, v in t5_ids_map.items()},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")

    ledger = {
        "batch_tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parent_artifacts": parent_artifacts,
        "entries": ledger_entries,
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Ledger written: {ledger_path}")


if __name__ == "__main__":
    main()
