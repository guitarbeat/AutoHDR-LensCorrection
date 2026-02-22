#!/usr/bin/env python3
"""Organize canonical submission artifacts for the real scoring pipeline.

Canonical pipeline:
  images ZIP (1000 JPGs) -> bounty scoring -> *_scored.csv -> Kaggle submit
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import ensure_dir, get_config, require_existing_dir


@dataclass(frozen=True)
class KaggleSubmission:
    file_name: str
    date: str
    description: str
    status: str
    public_score: float | None
    private_score: float | None


@dataclass(frozen=True)
class ChainRecord:
    stem: str
    score_csv: Path
    score_rows: int
    score_unique_ids: int
    score_mean: float
    zip_path: Path | None
    zip_jpg_count: int
    reasons: list[str]
    kaggle_submissions: list[KaggleSubmission]

    @property
    def is_real_chain(self) -> bool:
        if self.zip_path is None:
            return False
        return self.score_rows == 1000 and self.score_unique_ids == 1000 and self.zip_jpg_count == 1000

    @property
    def best_kaggle_public_score(self) -> float | None:
        scores = [s.public_score for s in self.kaggle_submissions if s.public_score is not None]
        return max(scores) if scores else None


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Build a canonical artifact index for real submission chains."
    )
    parser.add_argument(
        "--zip-root",
        default=str(cfg.output_root),
        help=f"Directory containing submission ZIP files (default: {cfg.output_root})",
    )
    parser.add_argument(
        "--bounty-dir",
        default=str(cfg.repo_root / "backend" / "outputs" / "bounty"),
        help="Directory containing bounty score CSV files.",
    )
    parser.add_argument(
        "--kaggle-dir",
        default=str(cfg.repo_root / "backend" / "outputs" / "kaggle"),
        help="Directory containing local Kaggle submission CSV files.",
    )
    parser.add_argument(
        "--organize-dir",
        default=str(cfg.repo_root / "backend" / "outputs" / "kaggle" / "real_chain"),
        help="Output directory for canonical links + manifest.",
    )
    parser.add_argument(
        "--competition",
        default="automatic-lens-correction",
        help="Kaggle competition slug for submission lookup.",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip kaggle CLI lookup and build manifest from local artifacts only.",
    )
    parser.add_argument(
        "--kaggle-page-size",
        type=int,
        default=200,
        help="Page size for kaggle submissions fetch (max 200).",
    )
    parser.add_argument(
        "--score-dirs",
        action="append",
        default=None,
        help=(
            "Repeatable scored CSV directories. Defaults to --bounty-dir and --kaggle-dir "
            "when omitted."
        ),
    )
    return parser.parse_args()


SCORED_TS_RE = re.compile(r"^(?P<stem>submission.+)_scored_\d{8}_\d{6}\.csv$")
RESCORED_TS_RE = re.compile(r"^(?P<stem>submission.+)_rescored_\d{8}_\d{6}\.csv$")


def stem_from_score_csv_name(name: str) -> str | None:
    if not name.lower().endswith(".csv"):
        return None

    if name == "submission_v4.csv":
        return "submission_v4"
    match_scored_ts = SCORED_TS_RE.match(name)
    if match_scored_ts:
        return match_scored_ts.group("stem")
    match_rescored_ts = RESCORED_TS_RE.match(name)
    if match_rescored_ts:
        return match_rescored_ts.group("stem")
    if name.endswith("_scored.csv"):
        return name.removesuffix("_scored.csv")
    if name.endswith("_rescored.csv"):
        return name.removesuffix("_rescored.csv")
    if name == "submission_v4_rescore.csv":
        return "submission_v4"
    return None


def resolve_kaggle_dir(raw: Path, *, skip_kaggle: bool) -> Path:
    kaggle_dir = raw.expanduser().resolve()
    if skip_kaggle:
        return kaggle_dir
    return require_existing_dir(kaggle_dir, "Kaggle outputs directory")


def load_score_rows(score_csv: Path) -> tuple[int, int, float]:
    rows = 0
    ids: set[str] = set()
    score_sum = 0.0

    with score_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or row.get("id") or "").strip()
            score_raw = row.get("score")
            if not image_id or score_raw is None:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                continue
            rows += 1
            ids.add(image_id)
            score_sum += score

    mean = score_sum / rows if rows else 0.0
    return rows, len(ids), float(mean)


def find_zip_for_stem(stem: str, zip_root: Path) -> Path | None:
    exact = zip_root / f"{stem}.zip"
    if exact.exists():
        return exact.resolve()

    prefix_matches = sorted(zip_root.glob(f"{stem}_*.zip"))
    if prefix_matches:
        # Timestamped naming often uses suffixes; prefer most recent lexicographic file name.
        return prefix_matches[-1].resolve()

    # Handle historical alias for cycle1 safe scored CSV.
    if stem == "submission_calibguard_cycle1_safe":
        alias_matches = sorted(zip_root.glob("submission_calibguard_cycle1_safe_*.zip"))
        if alias_matches:
            return alias_matches[-1].resolve()

    return None


def count_zip_jpgs(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return sum(
            1
            for info in zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".jpg")
        )


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    raw = value.strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_kaggle_submissions(
    competition: str, *, page_size: int
) -> tuple[list[KaggleSubmission], bool]:
    if page_size <= 0 or page_size > 200:
        raise ValueError(f"--kaggle-page-size must be in [1, 200], got {page_size}")
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submissions",
            "-c",
            competition,
            "-v",
            "--page-size",
            str(page_size),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(result.stdout.splitlines())
    submissions: list[KaggleSubmission] = []
    for row in reader:
        submissions.append(
            KaggleSubmission(
                file_name=(row.get("fileName") or "").strip(),
                date=(row.get("date") or "").strip(),
                description=(row.get("description") or "").strip(),
                status=(row.get("status") or "").strip(),
                public_score=parse_optional_float(row.get("publicScore")),
                private_score=parse_optional_float(row.get("privateScore")),
            )
        )
    # The Kaggle CLI does not expose a next-page token in CSV output.
    # If this page is full, assume there may be additional history not fetched.
    maybe_truncated = len(submissions) >= page_size
    return submissions, maybe_truncated


def submission_matches_chain(sub: KaggleSubmission, stem: str, score_name: str) -> bool:
    if sub.file_name == score_name:
        return True

    raw_submit_name = f"{stem}.csv"
    if sub.file_name == raw_submit_name:
        return True

    description = sub.description.lower()
    if stem.startswith("submission_calibguard_cycle1_safe"):
        return sub.file_name == "submission.csv" and "cycle1" in description and "safe" in description
    if stem.startswith("submission_calibguard_cycle2_balanced"):
        return (
            sub.file_name == "submission.csv"
            and "cycle2" in description
            and "balanced" in description
        )
    if stem.startswith("submission_calibguard_cycle2_aggressive"):
        return (
            sub.file_name == "submission.csv"
            and "cycle2" in description
            and "aggressive" in description
        )
    return False


def ensure_clean_link_dir(path: Path) -> None:
    ensure_dir(path)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()


def write_link(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)


def write_readme(path: Path) -> None:
    text = (
        "Canonical pipeline:\n"
        "  images ZIP (1000 JPGs) -> bounty scoring -> *_scored.csv -> Kaggle submit\n\n"
        "This directory is generated by backend/scripts/heuristics/organize_real_chain.py\n"
        "and contains symlinks only (original files remain in place).\n"
    )
    path.write_text(text, encoding="utf-8")


def collect_score_csvs(scored_dirs: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for scored_dir in scored_dirs:
        if not scored_dir.exists() or not scored_dir.is_dir():
            continue
        for score_csv in scored_dir.glob("submission*.csv"):
            resolved = score_csv.resolve()
            unique[str(resolved)] = resolved
    return sorted(unique.values(), key=lambda p: (p.name, str(p)))


def dedupe_score_csvs_by_stem(score_csvs: list[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    selected: dict[str, Path] = {}
    duplicates_skipped: list[dict[str, Any]] = []

    for score_csv in sorted(score_csvs, key=lambda p: str(p)):
        stem = stem_from_score_csv_name(score_csv.name)
        if stem is None:
            continue

        current = selected.get(stem)
        if current is None:
            selected[stem] = score_csv
            continue

        current_mtime = current.stat().st_mtime_ns
        candidate_mtime = score_csv.stat().st_mtime_ns

        # Keep newer mtime; if tie, keep lexicographically later path for determinism.
        keep_candidate = (candidate_mtime, str(score_csv)) > (current_mtime, str(current))
        if keep_candidate:
            duplicates_skipped.append(
                {
                    "stem": stem,
                    "skipped_path": str(current),
                    "kept_path": str(score_csv),
                    "reason": "older_mtime_or_tie_break",
                    "skipped_mtime_ns": current_mtime,
                    "kept_mtime_ns": candidate_mtime,
                }
            )
            selected[stem] = score_csv
        else:
            duplicates_skipped.append(
                {
                    "stem": stem,
                    "skipped_path": str(score_csv),
                    "kept_path": str(current),
                    "reason": "older_mtime_or_tie_break",
                    "skipped_mtime_ns": candidate_mtime,
                    "kept_mtime_ns": current_mtime,
                }
            )

    selected_paths = sorted(selected.values(), key=lambda p: (p.name, str(p)))
    duplicates_skipped.sort(
        key=lambda item: (str(item.get("stem", "")), str(item.get("skipped_path", "")))
    )
    return selected_paths, duplicates_skipped


def sanitize_link_token(value: str) -> str:
    token = "".join(c if (c.isalnum() or c in {"_", "-", "."}) else "_" for c in value.strip())
    return token or "na"


def unique_link_path(directory: Path, base_name: str) -> Path:
    candidate = directory / base_name
    if not candidate.exists() and not candidate.is_symlink():
        return candidate

    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    idx = 2
    while True:
        trial = directory / f"{stem}__{idx}{suffix}"
        if not trial.exists() and not trial.is_symlink():
            return trial
        idx += 1


def build_chain_records(
    scored_dirs: list[Path],
    zip_root: Path,
    kaggle_submissions: list[KaggleSubmission],
) -> tuple[list[ChainRecord], list[dict[str, Any]]]:
    records: list[ChainRecord] = []
    score_csvs = collect_score_csvs(scored_dirs)
    selected_score_csvs, duplicates_skipped = dedupe_score_csvs_by_stem(score_csvs)

    for score_csv in selected_score_csvs:
        stem = stem_from_score_csv_name(score_csv.name)
        if stem is None:
            continue

        score_rows, score_unique_ids, score_mean = load_score_rows(score_csv)
        zip_path = find_zip_for_stem(stem, zip_root)
        zip_jpg_count = 0
        reasons: list[str] = []

        if zip_path is None:
            reasons.append("zip_not_found")
        else:
            zip_jpg_count = count_zip_jpgs(zip_path)
            if zip_jpg_count != 1000:
                reasons.append(f"zip_jpg_count_{zip_jpg_count}")

        if score_rows != 1000:
            reasons.append(f"score_rows_{score_rows}")
        if score_unique_ids != 1000:
            reasons.append(f"score_unique_ids_{score_unique_ids}")

        matches = [
            sub
            for sub in kaggle_submissions
            if submission_matches_chain(sub, stem=stem, score_name=score_csv.name)
        ]

        records.append(
            ChainRecord(
                stem=stem,
                score_csv=score_csv.resolve(),
                score_rows=score_rows,
                score_unique_ids=score_unique_ids,
                score_mean=score_mean,
                zip_path=zip_path,
                zip_jpg_count=zip_jpg_count,
                reasons=reasons,
                kaggle_submissions=matches,
            )
        )

    # Highest Kaggle score first, then highest bounty mean.
    records.sort(
        key=lambda r: (
            r.best_kaggle_public_score if r.best_kaggle_public_score is not None else -1.0,
            r.score_mean,
        ),
        reverse=True,
    )
    return records, duplicates_skipped


def chain_to_json(record: ChainRecord) -> dict[str, Any]:
    return {
        "stem": record.stem,
        "score_csv": str(record.score_csv),
        "score_rows": record.score_rows,
        "score_unique_ids": record.score_unique_ids,
        "score_mean": record.score_mean,
        "zip_path": str(record.zip_path) if record.zip_path else None,
        "zip_jpg_count": record.zip_jpg_count,
        "is_real_chain": record.is_real_chain,
        "reasons": record.reasons,
        "kaggle_submissions": [
            {
                "file_name": s.file_name,
                "date": s.date,
                "description": s.description,
                "status": s.status,
                "public_score": s.public_score,
                "private_score": s.private_score,
            }
            for s in record.kaggle_submissions
        ],
        "best_kaggle_public_score": record.best_kaggle_public_score,
    }


def main() -> None:
    args = parse_args()

    zip_root = require_existing_dir(Path(args.zip_root).expanduser().resolve(), "ZIP root")
    bounty_dir = require_existing_dir(
        Path(args.bounty_dir).expanduser().resolve(), "Bounty outputs directory"
    )
    kaggle_dir = resolve_kaggle_dir(Path(args.kaggle_dir), skip_kaggle=args.skip_kaggle)
    organize_dir = ensure_dir(Path(args.organize_dir).expanduser().resolve())
    if args.score_dirs:
        score_dirs = [Path(raw).expanduser().resolve() for raw in args.score_dirs]
    else:
        score_dirs = [bounty_dir, kaggle_dir]

    kaggle_rows: list[KaggleSubmission] = []
    kaggle_error: str | None = None
    kaggle_maybe_truncated = False
    if not args.skip_kaggle:
        try:
            kaggle_rows, kaggle_maybe_truncated = load_kaggle_submissions(
                args.competition,
                page_size=args.kaggle_page_size,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            kaggle_error = str(exc)

    records, duplicates_skipped = build_chain_records(
        scored_dirs=score_dirs,
        zip_root=zip_root,
        kaggle_submissions=kaggle_rows,
    )

    zipped_link_dir = ensure_dir(organize_dir / "zips")
    scored_link_dir = ensure_dir(organize_dir / "scored_csv")
    kaggle_link_dir = ensure_dir(organize_dir / "kaggle_submit_csv")
    ensure_clean_link_dir(zipped_link_dir)
    ensure_clean_link_dir(scored_link_dir)
    ensure_clean_link_dir(kaggle_link_dir)

    for record in records:
        write_link(record.score_csv, scored_link_dir / record.score_csv.name)
        if record.zip_path is not None:
            write_link(record.zip_path, zipped_link_dir / record.zip_path.name)
        for sub in record.kaggle_submissions:
            stem_token = sanitize_link_token(record.stem)
            file_token = sanitize_link_token(sub.file_name)
            link_name = f"{stem_token}__{file_token}"
            link_path = unique_link_path(kaggle_link_dir, link_name)

            if sub.file_name == "submission.csv":
                # submission.csv is reused across many runs; link directly to this chain's scored CSV.
                write_link(record.score_csv, link_path)
                continue

            local_candidates = [
                kaggle_dir / sub.file_name,
                bounty_dir / sub.file_name,
            ]
            if sub.file_name == record.score_csv.name:
                local_candidates.append(record.score_csv)

            linked = False
            for local_csv in local_candidates:
                if local_csv.exists():
                    write_link(local_csv.resolve(), link_path)
                    linked = True
                    break
            if not linked and sub.file_name == record.score_csv.name:
                write_link(record.score_csv, link_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest = {
        "method": "organize_real_chain.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_pipeline": "images ZIP (1000 JPGs) -> bounty scoring -> *_scored.csv -> Kaggle submit",
        "paths": {
            "zip_root": str(zip_root),
            "bounty_dir": str(bounty_dir),
            "kaggle_dir": str(kaggle_dir),
            "score_dirs": [str(path) for path in score_dirs],
            "organize_dir": str(organize_dir),
        },
        "kaggle_lookup": {
            "competition": args.competition,
            "skipped": bool(args.skip_kaggle),
            "error": kaggle_error,
            "page_size": args.kaggle_page_size,
            "row_count": len(kaggle_rows),
            "possible_truncation": kaggle_maybe_truncated,
        },
        "counts": {
            "chains_total": len(records),
            "chains_real_ready": sum(1 for r in records if r.is_real_chain),
            "chains_missing_requirements": sum(1 for r in records if not r.is_real_chain),
            "duplicates_skipped": len(duplicates_skipped),
        },
        "duplicates_skipped": duplicates_skipped,
        "chains": [chain_to_json(record) for record in records],
    }

    manifest_latest = organize_dir / "manifest_latest.json"
    manifest_snapshot = organize_dir / f"manifest_{timestamp}.json"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_latest.write_text(manifest_text, encoding="utf-8")
    manifest_snapshot.write_text(manifest_text, encoding="utf-8")

    rows_tsv = [
        "stem\tis_real_chain\tzip_name\tzip_jpg_count\tscore_csv\tscore_rows\tscore_unique_ids\tscore_mean\tbest_kaggle_public_score\tkaggle_submission_files"
    ]
    for record in records:
        zip_name = record.zip_path.name if record.zip_path else ""
        unique_kaggle_files: list[str] = []
        seen_names: set[str] = set()
        for sub in record.kaggle_submissions:
            if sub.file_name in seen_names:
                continue
            seen_names.add(sub.file_name)
            unique_kaggle_files.append(sub.file_name)
        kaggle_files = ",".join(unique_kaggle_files)
        best_kaggle = (
            f"{record.best_kaggle_public_score:.5f}"
            if record.best_kaggle_public_score is not None
            else ""
        )
        rows_tsv.append(
            "\t".join(
                [
                    record.stem,
                    "yes" if record.is_real_chain else "no",
                    zip_name,
                    str(record.zip_jpg_count),
                    record.score_csv.name,
                    str(record.score_rows),
                    str(record.score_unique_ids),
                    f"{record.score_mean:.5f}",
                    best_kaggle,
                    kaggle_files,
                ]
            )
        )
    table_latest = organize_dir / "table_latest.tsv"
    table_latest.write_text("\n".join(rows_tsv) + "\n", encoding="utf-8")

    write_readme(organize_dir / "README.txt")

    print(f"Organize dir: {organize_dir}")
    print(f"Manifest: {manifest_latest}")
    print(f"Table: {table_latest}")
    print(
        "Counts: "
        f"total={manifest['counts']['chains_total']} "
        f"real_ready={manifest['counts']['chains_real_ready']} "
        f"missing={manifest['counts']['chains_missing_requirements']}"
    )


if __name__ == "__main__":
    main()
