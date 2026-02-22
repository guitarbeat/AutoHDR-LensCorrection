#!/usr/bin/env python3
"""Build mixed V4 submission zip candidates from scored fallback artifacts."""

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
    patches: dict[str, str]  # image_id -> source name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose V4 mix-batch submission candidates from base/zero/t5 artifacts."
    )
    parser.add_argument("--base-zip", required=True, type=Path)
    parser.add_argument("--zero-zip", required=True, type=Path)
    parser.add_argument("--t5-zip", required=True, type=Path)
    parser.add_argument("--base-score-csv", required=True, type=Path)
    parser.add_argument("--zero-score-csv", required=True, type=Path)
    parser.add_argument("--t5-score-csv", required=True, type=Path)
    parser.add_argument("--test-original-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="Batch tag used in output names.")
    parser.add_argument("--manifest-out", required=True, type=Path)
    return parser.parse_args()


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
    with zipfile.ZipFile(base_zip, "r") as zf:
        names = _jpg_members(zf)
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
                raise FileNotFoundError(
                    f"Missing test original referenced by zip: {original_path}"
                )
            if zf.read(name) == original_path.read_bytes():
                replaced.add(Path(name).stem)
    return replaced


def write_id_list(output_dir: Path, tag: str, list_name: str, ids: Iterable[str]) -> Path:
    path = output_dir / f"id_list_{list_name}_{tag}.txt"
    path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
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
) -> int:
    used_replacements = 0
    with (
        zipfile.ZipFile(base_zip, "r") as base_zf,
        zipfile.ZipFile(zero_zip, "r") as zero_zf,
        zipfile.ZipFile(t5_zip, "r") as t5_zf,
        zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as out_zf,
    ):
        for name in base_names:
            image_id = Path(name).stem
            source = candidate.patches.get(image_id, SOURCE_BASE)
            if source == SOURCE_ZERO:
                payload = zero_zf.read(name)
                used_replacements += 1
            elif source == SOURCE_T5:
                payload = t5_zf.read(name)
                used_replacements += 1
            elif source == SOURCE_ORIGINAL:
                payload = (test_original_dir / name).read_bytes()
                used_replacements += 1
            elif source == SOURCE_BASE:
                payload = base_zf.read(name)
            else:
                raise RuntimeError(
                    f"Unsupported source '{source}' in candidate {candidate.name}"
                )

            out_zf.writestr(name, payload)
    return used_replacements


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
        if source == SOURCE_ZERO or source == SOURCE_ORIGINAL:
            total += zero_scores[image_id]
        elif source == SOURCE_T5:
            total += t5_scores[image_id]
        else:
            total += base_scores[image_id]
    return total / max(len(ordered_ids), 1)


def main() -> None:
    args = parse_args()

    base_zip = resolve_existing_path(args.base_zip, "base zip")
    zero_zip = resolve_existing_path(args.zero_zip, "zero zip")
    t5_zip = resolve_existing_path(args.t5_zip, "t5 zip")
    base_score_csv = resolve_existing_path(args.base_score_csv, "base score CSV")
    zero_score_csv = resolve_existing_path(args.zero_score_csv, "zero score CSV")
    t5_score_csv = resolve_existing_path(args.t5_score_csv, "t5 score CSV")
    test_original_dir = resolve_existing_dir(
        args.test_original_dir, "test originals directory"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_out.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tag = args.tag.strip()
    if not tag:
        raise RuntimeError("--tag must be non-empty")

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

    z_pos = sorted(
        image_id for image_id in ordered_ids if zero_scores[image_id] > base_scores[image_id]
    )
    z_nonneg = sorted(
        image_id
        for image_id in ordered_ids
        if image_id in replacement_ids and zero_scores[image_id] >= base_scores[image_id]
    )
    z_le0 = sorted(
        image_id
        for image_id in ordered_ids
        if image_id in replacement_ids and base_scores[image_id] <= 0.0
    )
    z_le1 = sorted(
        image_id
        for image_id in ordered_ids
        if image_id in replacement_ids and base_scores[image_id] <= 1.0
    )
    t5_beats_both = sorted(
        image_id
        for image_id in ordered_ids
        if t5_scores[image_id] > max(base_scores[image_id], zero_scores[image_id])
    )

    id_list_paths = {
        "replacement_ids": str(
            write_id_list(output_dir, tag, "replacement_ids", replacement_ids)
        ),
        "z_pos": str(write_id_list(output_dir, tag, "z_pos", z_pos)),
        "z_nonneg": str(write_id_list(output_dir, tag, "z_nonneg", z_nonneg)),
        "z_le0": str(write_id_list(output_dir, tag, "z_le0", z_le0)),
        "z_le1": str(write_id_list(output_dir, tag, "z_le1", z_le1)),
        "t5_beats_both": str(
            write_id_list(output_dir, tag, "t5_beats_both", t5_beats_both)
        ),
    }

    z_pos_patches = {image_id: SOURCE_ZERO for image_id in z_pos}
    z_pos_t5plus_patches = dict(z_pos_patches)
    for image_id in t5_beats_both:
        z_pos_t5plus_patches[image_id] = SOURCE_T5

    candidates = [
        CandidateDef(
            name="zpos",
            output_name=f"submission_v4_mix_zpos_{tag}.zip",
            description="Patch IDs where zero score beats base score, source=zero zip",
            patches=z_pos_patches,
        ),
        CandidateDef(
            name="zpos_t5plus",
            output_name=f"submission_v4_mix_zpos_t5plus_{tag}.zip",
            description="zpos baseline + IDs where t5 beats both base and zero, source=t5 zip override",
            patches=z_pos_t5plus_patches,
        ),
        CandidateDef(
            name="zle0",
            output_name=f"submission_v4_mix_zle0_{tag}.zip",
            description="Patch replacement IDs where base score <= 0, source=test originals",
            patches={image_id: SOURCE_ORIGINAL for image_id in z_le0},
        ),
        CandidateDef(
            name="zle1",
            output_name=f"submission_v4_mix_zle1_{tag}.zip",
            description="Patch replacement IDs where base score <= 1, source=test originals",
            patches={image_id: SOURCE_ORIGINAL for image_id in z_le1},
        ),
        CandidateDef(
            name="znonneg",
            output_name=f"submission_v4_mix_znonneg_{tag}.zip",
            description="Patch replacement IDs where zero score is non-regressive, source=zero zip",
            patches={image_id: SOURCE_ZERO for image_id in z_nonneg},
        ),
    ]

    candidate_artifacts: dict[str, str] = {}
    predicted_means: dict[str, float] = {}
    sha256: dict[str, str] = {}
    zip_size_bytes: dict[str, int] = {}
    replacement_counts: dict[str, int] = {}
    candidate_defs_out: list[dict[str, object]] = []

    for candidate in candidates:
        output_path = output_dir / candidate.output_name
        expected_replacements = len(candidate.patches)
        used_replacements = _build_candidate_zip(
            candidate=candidate,
            output_path=output_path,
            base_zip=base_zip,
            zero_zip=zero_zip,
            t5_zip=t5_zip,
            test_original_dir=test_original_dir,
            base_names=base_names,
        )
        if used_replacements != expected_replacements:
            raise RuntimeError(
                f"Replacement count mismatch for {candidate.name}: expected={expected_replacements} actual={used_replacements}"
            )

        validate_zip_member_set(output_path, base_name_set, f"candidate {candidate.name}")

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

        per_source_counts: dict[str, int] = {}
        for source in candidate.patches.values():
            per_source_counts[source] = per_source_counts.get(source, 0) + 1

        candidate_defs_out.append(
            {
                "name": candidate.name,
                "output_name": candidate.output_name,
                "description": candidate.description,
                "replacement_count": expected_replacements,
                "replacement_sources": per_source_counts,
            }
        )

        print(
            f"[{candidate.name}] replacements={expected_replacements} "
            f"predicted_mean={predicted_mean:.6f} sha256={output_hash[:12]}..."
        )

    manifest = {
        "batch_tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "base_zip": str(base_zip),
            "zero_zip": str(zero_zip),
            "t5_zip": str(t5_zip),
            "base_score_csv": str(base_score_csv),
            "zero_score_csv": str(zero_score_csv),
            "t5_score_csv": str(t5_score_csv),
            "test_original_dir": str(test_original_dir),
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
            "t5_beats_both_count": len(t5_beats_both),
            "z_pos_count": len(z_pos),
            "z_nonneg_count": len(z_nonneg),
            "z_le0_count": len(z_le0),
            "z_le1_count": len(z_le1),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
