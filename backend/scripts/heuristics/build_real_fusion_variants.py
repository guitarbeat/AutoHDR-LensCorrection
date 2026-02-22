#!/usr/bin/env python3
"""Build legit image-backed fusion variants from scored CSV + ZIP inputs.

This script creates output ZIP files containing 1000 JPG images by selecting one
source image per image_id using score-based rules. It never writes score-space
CSV submissions directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXPECTED_IMAGE_COUNT = 1000


@dataclass
class InputSource:
    score_csv_name: str
    score_csv_path: Path
    zip_path: Path
    scores: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real image-backed fusion ZIP variants with deterministic rule sets."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path(
            "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/"
            "submission_v4_oracle_valid_allzip_20260222_175058_manifest.json"
        ),
        help="Manifest containing score_csv + zip inputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Volumes/Love SSD/AutoHDR_Submissions"),
        help="Output directory for generated zip artifacts.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle"),
        help="Directory for variant manifests.",
    )
    parser.add_argument(
        "--tag",
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Tag suffix used in output file names.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="submission_realfusion",
        help="Prefix used for generated ZIP artifact names.",
    )
    parser.add_argument(
        "--safe-margins",
        default="0.2,0.5",
        help="Comma-separated margins for cycle1_safe guard vs lp30 (default: 0.2,0.5).",
    )
    parser.add_argument(
        "--aggr-margins",
        default="0.5,0.8",
        help="Comma-separated margins for cycle2_aggressive guard vs cycle2_t0 (default: 0.5,0.8).",
    )
    return parser.parse_args()


def parse_margin_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        values.append(float(cleaned))
    if not values:
        raise ValueError("Margin list cannot be empty.")
    return values


def load_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_id" not in reader.fieldnames or "score" not in reader.fieldnames:
            raise RuntimeError(f"Invalid scored CSV schema: {path}")
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            score_raw = (row.get("score") or "").strip()
            if not image_id:
                raise RuntimeError(f"Blank image_id in {path}")
            if not score_raw:
                raise RuntimeError(f"Blank score in {path}")
            score = float(score_raw)
            scores[image_id] = score
    return scores


def load_inputs(manifest_path: Path) -> list[InputSource]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeError(f"Manifest has no inputs list: {manifest_path}")

    sources: list[InputSource] = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        score_csv = Path(str(item["score_csv"])).expanduser().resolve()
        zip_path = Path(str(item["zip"])).expanduser().resolve()
        if not score_csv.exists():
            invalid_candidate = score_csv.with_name(f"INVALID_{score_csv.name}")
            if invalid_candidate.exists():
                score_csv = invalid_candidate
            else:
                raise RuntimeError(f"Missing score CSV: {score_csv}")
        if not zip_path.exists():
            raise RuntimeError(f"Missing ZIP: {zip_path}")
        scores = load_scores(score_csv)
        sources.append(
            InputSource(
                score_csv_name=score_csv.name,
                score_csv_path=score_csv,
                zip_path=zip_path,
                scores=scores,
            )
        )

    if not sources:
        raise RuntimeError("No valid sources loaded from manifest.")

    id_sets = [set(source.scores.keys()) for source in sources]
    common_ids = set.intersection(*id_sets)
    if len(common_ids) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            "Input score CSVs do not share the required 1000 IDs "
            f"(shared={len(common_ids)})"
        )
    for source in sources:
        if len(source.scores) != EXPECTED_IMAGE_COUNT:
            raise RuntimeError(
                f"Score CSV must have 1000 rows: {source.score_csv_path} has {len(source.scores)}"
            )
    return sources


def find_source_name(sources: list[InputSource], needle: str) -> str:
    for source in sources:
        if needle in source.score_csv_name:
            return source.score_csv_name
    raise RuntimeError(f"Could not find source containing '{needle}'")


def build_rankings(sources: list[InputSource]) -> dict[str, list[tuple[str, float]]]:
    all_ids = sorted(sources[0].scores.keys())
    rankings: dict[str, list[tuple[str, float]]] = {}
    for image_id in all_ids:
        pairs = [(source.score_csv_name, source.scores[image_id]) for source in sources]
        pairs.sort(key=lambda item: item[1], reverse=True)
        rankings[image_id] = pairs
    return rankings


def select_base_max(rankings: dict[str, list[tuple[str, float]]]) -> dict[str, str]:
    return {image_id: pairs[0][0] for image_id, pairs in rankings.items()}


def pick_best_excluding(
    pairs: list[tuple[str, float]],
    excluded_source: str,
) -> tuple[str, float]:
    for source_name, score in pairs:
        if source_name != excluded_source:
            return source_name, score
    raise RuntimeError(f"No alternative source available when excluding {excluded_source}")


def apply_safe_guard(
    *,
    selection: dict[str, str],
    rankings: dict[str, list[tuple[str, float]]],
    safe_source: str,
    lp30_source: str,
    margin: float,
) -> dict[str, str]:
    out = dict(selection)
    for image_id, chosen_source in selection.items():
        if chosen_source != safe_source:
            continue
        score_map = dict(rankings[image_id])
        safe_score = score_map[safe_source]
        lp30_score = score_map[lp30_source]
        if safe_score <= lp30_score + margin:
            out[image_id] = lp30_source
    return out


def apply_aggr_guard(
    *,
    selection: dict[str, str],
    rankings: dict[str, list[tuple[str, float]]],
    aggr_source: str,
    cycle2_t0_source: str,
    margin: float,
) -> dict[str, str]:
    out = dict(selection)
    for image_id, chosen_source in selection.items():
        if chosen_source != aggr_source:
            continue
        score_map = dict(rankings[image_id])
        aggr_score = score_map[aggr_source]
        cycle2_score = score_map[cycle2_t0_source]
        if aggr_score <= cycle2_score + margin:
            alternative_source, _ = pick_best_excluding(rankings[image_id], aggr_source)
            out[image_id] = alternative_source
    return out


def selection_stats(
    selection: dict[str, str],
    rankings: dict[str, list[tuple[str, float]]],
) -> tuple[float, dict[str, int]]:
    source_counts = Counter(selection.values())
    score_sum = 0.0
    for image_id, source_name in selection.items():
        score_map = dict(rankings[image_id])
        score_sum += score_map[source_name]
    return score_sum / len(selection), dict(source_counts)


def build_zip_member_maps(sources: list[InputSource]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for source in sources:
        member_map: dict[str, str] = {}
        with ZipFile(source.zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".jpg"):
                    continue
                stem = Path(name).stem
                member_map[stem] = name
        maps[source.score_csv_name] = member_map
    return maps


def write_variant_zip(
    *,
    variant_name: str,
    selection: dict[str, str],
    source_lookup: dict[str, InputSource],
    zip_member_maps: dict[str, dict[str, str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    open_zips: dict[str, ZipFile] = {}
    try:
        for source_name in source_lookup:
            open_zips[source_name] = ZipFile(source_lookup[source_name].zip_path, "r")

        with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as out_zip:
            for image_id in sorted(selection.keys()):
                source_name = selection[image_id]
                source_zip = open_zips[source_name]
                member_name = zip_member_maps[source_name].get(image_id)
                if member_name is None:
                    raise RuntimeError(
                        f"[{variant_name}] Missing image_id '{image_id}' in source ZIP {source_lookup[source_name].zip_path}"
                    )
                data = source_zip.read(member_name)
                out_zip.writestr(f"{image_id}.jpg", data)
    finally:
        for zf in open_zips.values():
            zf.close()


def write_id_source_map(path: Path, selection: dict[str, str], rankings: dict[str, list[tuple[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "selected_source", "selected_score"])
        for image_id in sorted(selection.keys()):
            source_name = selection[image_id]
            selected_score = dict(rankings[image_id])[source_name]
            writer.writerow([image_id, source_name, f"{selected_score:.6f}"])


def main() -> None:
    args = parse_args()
    safe_margins = parse_margin_list(args.safe_margins)
    aggr_margins = parse_margin_list(args.aggr_margins)
    sources = load_inputs(args.input_manifest.expanduser().resolve())
    rankings = build_rankings(sources)
    source_lookup = {source.score_csv_name: source for source in sources}

    safe_source = find_source_name(sources, "submission_calibguard_cycle1_safe_scored.csv")
    lp30_source = find_source_name(sources, "submission_v4_fallback_learned_pos30_20260222_082633_scored.csv")
    aggr_source = find_source_name(sources, "submission_calibguard_cycle2_aggressive_20260222_135105_scored.csv")
    cycle2_t0_source = find_source_name(sources, "submission_v4_fallback_cycle2_t0_20260222_080059_scored.csv")

    base_selection = select_base_max(rankings)
    variants: list[tuple[str, dict[str, str], dict[str, float | str]]] = []
    variants.append(
        (
            "max_refresh",
            base_selection,
            {"rule": "max_per_image_across_inputs"},
        )
    )

    for margin in safe_margins:
        selection = apply_safe_guard(
            selection=base_selection,
            rankings=rankings,
            safe_source=safe_source,
            lp30_source=lp30_source,
            margin=margin,
        )
        variants.append(
            (
                f"safe_guard_m{str(margin).replace('.', 'p')}",
                selection,
                {"rule": "safe_guard_vs_lp30", "safe_margin_vs_lp30": margin},
            )
        )

    for margin in aggr_margins:
        selection = apply_aggr_guard(
            selection=base_selection,
            rankings=rankings,
            aggr_source=aggr_source,
            cycle2_t0_source=cycle2_t0_source,
            margin=margin,
        )
        variants.append(
            (
                f"aggr_guard_m{str(margin).replace('.', 'p')}",
                selection,
                {"rule": "aggr_guard_vs_cycle2_t0", "aggr_margin_vs_cycle2_t0": margin},
            )
        )

    if safe_margins and aggr_margins:
        combo_selection = apply_safe_guard(
            selection=base_selection,
            rankings=rankings,
            safe_source=safe_source,
            lp30_source=lp30_source,
            margin=safe_margins[-1],
        )
        combo_selection = apply_aggr_guard(
            selection=combo_selection,
            rankings=rankings,
            aggr_source=aggr_source,
            cycle2_t0_source=cycle2_t0_source,
            margin=aggr_margins[-1],
        )
        variants.append(
            (
                f"combo_safe{str(safe_margins[-1]).replace('.', 'p')}_aggr{str(aggr_margins[-1]).replace('.', 'p')}",
                combo_selection,
                {
                    "rule": "combo_safe_and_aggr_guard",
                    "safe_margin_vs_lp30": safe_margins[-1],
                    "aggr_margin_vs_cycle2_t0": aggr_margins[-1],
                },
            )
        )

    zip_member_maps = build_zip_member_maps(sources)

    generation_records: list[dict[str, object]] = []
    for short_name, selection, rule_meta in variants:
        artifact_name = f"{args.artifact_prefix}_{short_name}_{args.tag}"
        out_zip = args.out_dir.expanduser().resolve() / f"{artifact_name}.zip"
        id_source_map = args.out_dir.expanduser().resolve() / f"id_list_realfusion_{short_name}_{args.tag}.csv"
        predicted_mean, source_counts = selection_stats(selection, rankings)

        write_variant_zip(
            variant_name=short_name,
            selection=selection,
            source_lookup=source_lookup,
            zip_member_maps=zip_member_maps,
            out_path=out_zip,
        )
        write_id_source_map(id_source_map, selection, rankings)

        record = {
            "artifact_name": artifact_name,
            "zip": str(out_zip),
            "id_source_map_csv": str(id_source_map),
            "predicted_mean": predicted_mean,
            "source_counts": source_counts,
        }
        record.update(rule_meta)
        generation_records.append(record)
        print(
            f"[variant] {artifact_name} predicted_mean={predicted_mean:.6f} "
            f"zip={out_zip}"
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "build_real_fusion_variants.v1",
        "input_manifest": str(args.input_manifest.expanduser().resolve()),
        "tag": args.tag,
        "artifact_prefix": args.artifact_prefix,
        "sources": [
            {
                "score_csv_name": source.score_csv_name,
                "score_csv_path": str(source.score_csv_path),
                "zip_path": str(source.zip_path),
            }
            for source in sources
        ],
        "variants": generation_records,
    }
    manifest_path = args.manifest_dir.expanduser().resolve() / f"real_fusion_variants_{args.tag}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
