#!/usr/bin/env python3
"""
List and prune generated submission artifacts under AUTOHDR_OUTPUT_ROOT.

Examples:
  python -m backend.scripts.heuristics.prune_submission_artifacts --list
  python -m backend.scripts.heuristics.prune_submission_artifacts \
    --zip-name submission_dim_bucket_microgrid_zero_guard_v2_20260222_030614.zip \
    --zip-name submission_dim_bucket_microgrid_zero_guard_v3_20260222_030901.zip
  python -m backend.scripts.heuristics.prune_submission_artifacts \
    --glob 'submission_dim_bucket_microgrid_zero_guard_*.zip' --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from backend.config import get_config, require_existing_dir


CALIBGUARD_STEM_RE = re.compile(
    r"^submission_(?P<tag>.+)_(?P<profile>safe|balanced|aggressive)_(?P<ts>\d{8}_\d{6})$"
)
SAFE_HEURISTIC_STEM_RE = re.compile(r"^submission_safe_heuristic_(?P<ts>\d{8}_\d{6})$")
HYBRID_CANDIDATE_STEM_RE = re.compile(
    r"^submission_hybrid_candidate_(?P<ts>\d{8}_\d{6})$"
)


@dataclass(frozen=True)
class ArtifactBundle:
    zip_path: Path
    companions: list[Path]

    @property
    def all_paths(self) -> list[Path]:
        return [self.zip_path, *self.companions]


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _safe_resolve(path: Path, output_root: Path) -> Path:
    resolved = path.resolve()
    if resolved == output_root or output_root in resolved.parents:
        return resolved
    raise ValueError(f"Refusing to operate outside output root: {resolved}")


def _infer_companions(
    zip_path: Path, output_root: Path, include_json: bool, include_corrected: bool
) -> list[Path]:
    companions: list[Path] = []

    if include_json:
        json_path = zip_path.with_suffix(".json")
        if json_path.exists():
            companions.append(json_path)

    if not include_corrected:
        return companions

    stem = zip_path.stem
    corrected_dir: Path | None = None

    if stem == "submission_dim_bucket_microgrid":
        corrected_dir = output_root / "corrected_dim_bucket"
    elif stem.startswith("submission_dim_bucket_microgrid_"):
        tag = stem.removeprefix("submission_dim_bucket_microgrid_")
        corrected_dir = output_root / f"corrected_dim_bucket_{tag}"
    else:
        match = CALIBGUARD_STEM_RE.match(stem)
        if match:
            profile = match.group("profile")
            ts = match.group("ts")
            corrected_dir = output_root / f"corrected_calibguard_dim_{profile}_{ts}"
        else:
            match_safe = SAFE_HEURISTIC_STEM_RE.match(stem)
            if match_safe:
                ts = match_safe.group("ts")
                corrected_dir = output_root / f"corrected_safe_heuristic_{ts}"
            else:
                match_hybrid = HYBRID_CANDIDATE_STEM_RE.match(stem)
                if match_hybrid:
                    ts = match_hybrid.group("ts")
                    corrected_dir = output_root / f"corrected_hybrid_candidate_{ts}"

    if corrected_dir and corrected_dir.exists():
        companions.append(corrected_dir)

    return companions


def _build_bundle(
    zip_path: Path, output_root: Path, include_json: bool, include_corrected: bool
) -> ArtifactBundle:
    companions = _infer_companions(
        zip_path, output_root, include_json, include_corrected
    )
    return ArtifactBundle(zip_path=zip_path, companions=companions)


def _list_zips(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("submission*.zip"))


def _resolve_selected_zips(
    output_root: Path, zip_names: list[str], globs: list[str]
) -> list[Path]:
    selected: dict[str, Path] = {}

    for name in zip_names:
        candidate = output_root / name
        if not candidate.exists():
            print(f"WARNING: zip not found: {candidate}")
            continue
        if not candidate.is_file():
            print(f"WARNING: not a file: {candidate}")
            continue
        selected[str(candidate.resolve())] = candidate

    for pattern in globs:
        for candidate in output_root.glob(pattern):
            if candidate.is_file() and candidate.suffix.lower() == ".zip":
                selected[str(candidate.resolve())] = candidate

    return sorted(selected.values())


def _print_bundle(bundle: ArtifactBundle, output_root: Path) -> int:
    total_bytes = 0
    for path in bundle.all_paths:
        resolved = _safe_resolve(path, output_root)
        if resolved.is_file():
            total_bytes += resolved.stat().st_size
        elif resolved.is_dir():
            total_bytes += sum(
                p.stat().st_size for p in resolved.rglob("*") if p.is_file()
            )

    print(f"- {bundle.zip_path.name} ({_format_size(total_bytes)})")
    for path in bundle.companions:
        rel = (
            path.relative_to(output_root)
            if output_root in path.resolve().parents
            else path.name
        )
        kind = "dir" if path.is_dir() else "file"
        print(f"    companion[{kind}]: {rel}")
    return total_bytes


def _delete_path(path: Path, output_root: Path) -> None:
    resolved = _safe_resolve(path, output_root)
    if not resolved.exists():
        return
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="List and prune generated submission artifacts."
    )
    parser.add_argument(
        "--output-dir",
        default=str(cfg.output_root),
        help=f"Submission output root (default: {cfg.output_root})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available submission zip artifacts and inferred companion files/directories.",
    )
    parser.add_argument(
        "--zip-name",
        action="append",
        default=[],
        help="Exact zip filename under --output-dir (repeatable).",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern under --output-dir to select zip files (repeatable).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not include sidecar .json files in deletion set.",
    )
    parser.add_argument(
        "--no-corrected",
        action="store_true",
        help="Do not include inferred corrected_* directories in deletion set.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete selected files. Without this flag the command is dry-run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = require_existing_dir(
        Path(args.output_dir).expanduser().resolve(), "Output directory"
    )
    include_json = not args.no_json
    include_corrected = not args.no_corrected

    if args.list:
        zips = _list_zips(output_root)
        print(f"Output root: {output_root}")
        print(f"Found {len(zips)} submission zip(s)")
        total_bytes = 0
        for zip_path in zips:
            bundle = _build_bundle(
                zip_path,
                output_root,
                include_json=include_json,
                include_corrected=include_corrected,
            )
            total_bytes += _print_bundle(bundle, output_root)
        print(f"Estimated total footprint: {_format_size(total_bytes)}")
        if not args.zip_name and not args.glob:
            return

    selected = _resolve_selected_zips(output_root, args.zip_name, args.glob)
    if not selected:
        print("No zip artifacts selected. Use --list or provide --zip-name/--glob.")
        return

    bundles = [
        _build_bundle(
            zip_path,
            output_root,
            include_json=include_json,
            include_corrected=include_corrected,
        )
        for zip_path in selected
    ]

    print("\nSelected bundles:")
    total_bytes = 0
    all_paths: dict[str, Path] = {}
    for bundle in bundles:
        total_bytes += _print_bundle(bundle, output_root)
        for path in bundle.all_paths:
            all_paths[str(path.resolve())] = path

    print(f"Total selected footprint: {_format_size(total_bytes)}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete selected artifacts.")
        return

    print("\nDeleting selected artifacts...")
    deleted = 0
    for path in sorted(all_paths.values(), key=lambda p: len(str(p)), reverse=True):
        if path.exists():
            _delete_path(path, output_root)
            deleted += 1
            print(f"  deleted: {path}")
    print(f"Deletion complete. Removed {deleted} path(s).")


if __name__ == "__main__":
    main()
