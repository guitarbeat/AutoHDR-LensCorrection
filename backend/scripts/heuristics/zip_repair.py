#!/usr/bin/env python3
"""Shared ZIP repair helpers for submission artifact builders."""

from __future__ import annotations

import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ReadResult:
    payload: bytes | None
    source: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.payload is not None


@dataclass(frozen=True)
class ResolvedPayload:
    payload: bytes | None
    source: str | None
    repaired: bool
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.payload is not None


@dataclass(frozen=True)
class ZipFinalizeStats:
    expected_count: int
    written_count: int
    repair_count: int
    repair_by_source: dict[str, int]
    repair_failures: list[str]
    repaired_ids: list[str]


def build_expected_name_set(base_zip: Path) -> tuple[list[str], set[str]]:
    """Return deterministic JPG member ordering and exact expected set."""
    with zipfile.ZipFile(base_zip, "r") as zf:
        members = [
            info.filename
            for info in zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".jpg")
        ]

    if len(members) != len(set(members)):
        raise RuntimeError(f"Duplicate JPG names detected in base zip: {base_zip}")

    ordered = sorted(members)
    return ordered, set(ordered)


def safe_read_member(zip_path: Path, member_name: str) -> ReadResult:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return safe_read_member_from_zip(zf, member_name, source="member")
    except Exception as exc:  # pragma: no cover - simple pass-through guard
        return ReadResult(
            payload=None,
            source="member",
            error=f"open zip failed ({zip_path}): {type(exc).__name__}: {exc}",
        )


def safe_read_member_from_zip(
    zf: zipfile.ZipFile,
    member_name: str,
    *,
    source: str,
) -> ReadResult:
    try:
        payload = zf.read(member_name)
        return ReadResult(payload=payload, source=source, error=None)
    except KeyError:
        return ReadResult(payload=None, source=source, error=f"missing member: {member_name}")
    except Exception as exc:
        return ReadResult(
            payload=None,
            source=source,
            error=f"read member failed ({member_name}): {type(exc).__name__}: {exc}",
        )


def safe_read_path(path: Path, *, source: str) -> ReadResult:
    try:
        payload = path.read_bytes()
        return ReadResult(payload=payload, source=source, error=None)
    except Exception as exc:
        return ReadResult(
            payload=None,
            source=source,
            error=f"read path failed ({path}): {type(exc).__name__}: {exc}",
        )


def safe_read_original(test_dir: Path, member_name: str) -> ReadResult:
    path = test_dir / Path(member_name).name
    return safe_read_path(path, source="original")


def resolve_payload(
    primary_reader: Callable[[], ReadResult],
    base_reader: Callable[[], ReadResult],
    original_reader: Callable[[], ReadResult],
) -> ResolvedPayload:
    """Resolve payload via strict fallback order: Primary -> Base -> Original."""
    errors: list[str] = []

    primary = primary_reader()
    if primary.ok:
        return ResolvedPayload(
            payload=primary.payload,
            source=primary.source,
            repaired=False,
            errors=errors,
        )
    if primary.error:
        errors.append(f"primary:{primary.error}")

    base = base_reader()
    if base.ok:
        return ResolvedPayload(
            payload=base.payload,
            source=base.source,
            repaired=True,
            errors=errors,
        )
    if base.error:
        errors.append(f"base:{base.error}")

    original = original_reader()
    if original.ok:
        return ResolvedPayload(
            payload=original.payload,
            source=original.source,
            repaired=True,
            errors=errors,
        )
    if original.error:
        errors.append(f"original:{original.error}")

    return ResolvedPayload(payload=None, source=None, repaired=False, errors=errors)


def _verify_zip_matches_expected(
    zip_path: Path,
    expected_names: list[str],
    expected_set: set[str],
    expected_count: int,
) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            info.filename
            for info in zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".jpg")
        ]

    if len(names) != len(set(names)):
        raise RuntimeError(f"Output ZIP has duplicate JPG names: {zip_path}")

    found_set = set(names)
    if found_set != expected_set:
        missing = sorted(expected_set - found_set)[:10]
        extra = sorted(found_set - expected_set)[:10]
        raise RuntimeError(
            f"Output ZIP member mismatch for {zip_path}: "
            f"missing={len(expected_set - found_set)} sample_missing={missing} "
            f"extra={len(found_set - expected_set)} sample_extra={extra}"
        )

    if len(names) != expected_count:
        raise RuntimeError(
            f"Output ZIP count mismatch for {zip_path}: "
            f"expected={expected_count} actual={len(names)}"
        )

    if sorted(names) != expected_names:
        # Deterministic check: write order must match expected deterministic ordering.
        raise RuntimeError(
            f"Output ZIP order mismatch for {zip_path}; expected deterministic ordering."
        )

    return len(names)


def finalize_zip_with_verification(
    *,
    output_zip: Path,
    expected_names: list[str],
    primary_reader: Callable[[str], ReadResult],
    base_reader: Callable[[str], ReadResult],
    original_reader: Callable[[str], ReadResult],
    repair_mode: str = "best_effort",
    expected_count: int | None = None,
) -> ZipFinalizeStats:
    if repair_mode not in {"best_effort", "strict"}:
        raise ValueError(f"Unsupported repair mode: {repair_mode}")

    if len(expected_names) != len(set(expected_names)):
        raise RuntimeError("Expected member list contains duplicates.")

    expected_set = set(expected_names)
    target_count = expected_count if expected_count is not None else len(expected_names)

    repaired_ids: list[str] = []
    repair_by_source_counter: Counter[str] = Counter()
    repair_failures: list[str] = []

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as out_zf:
        for member_name in expected_names:
            if repair_mode == "strict":
                primary = primary_reader(member_name)
                if not primary.ok:
                    failure = primary.error or "primary read failed"
                    repair_failures.append(f"{member_name}: {failure}")
                    continue
                out_zf.writestr(member_name, primary.payload)
                continue

            resolved = resolve_payload(
                lambda: primary_reader(member_name),
                lambda: base_reader(member_name),
                lambda: original_reader(member_name),
            )
            if not resolved.ok:
                error_text = "; ".join(resolved.errors) if resolved.errors else "unknown read failure"
                repair_failures.append(f"{member_name}: {error_text}")
                continue

            out_zf.writestr(member_name, resolved.payload)
            if resolved.repaired:
                repaired_ids.append(Path(member_name).stem)
                source = resolved.source or "unknown"
                repair_by_source_counter[source] += 1

    if repair_failures:
        sample = repair_failures[:10]
        raise RuntimeError(
            "ZIP finalize failed due unrecoverable payload reads "
            f"(count={len(repair_failures)} sample={sample})"
        )

    written_count = _verify_zip_matches_expected(
        zip_path=output_zip,
        expected_names=expected_names,
        expected_set=expected_set,
        expected_count=target_count,
    )

    repair_by_source = {
        "base": int(repair_by_source_counter.get("base", 0)),
        "original": int(repair_by_source_counter.get("original", 0)),
    }

    return ZipFinalizeStats(
        expected_count=target_count,
        written_count=written_count,
        repair_count=len(repaired_ids),
        repair_by_source=repair_by_source,
        repair_failures=repair_failures,
        repaired_ids=sorted(repaired_ids),
    )
