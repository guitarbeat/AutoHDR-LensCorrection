from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from backend.scripts.heuristics.zip_repair import ReadResult, finalize_zip_with_verification


def _zip_member_bytes(path: Path, name: str) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.read(name)


def test_finalize_uses_base_fallback_when_primary_fails(tmp_path: Path) -> None:
    out_zip = tmp_path / "out.zip"
    expected = ["a.jpg", "b.jpg"]
    payload_primary = b"primary-b"
    payload_base = b"base-a"

    def primary_reader(name: str) -> ReadResult:
        if name == "b.jpg":
            return ReadResult(payload=payload_primary, source="primary")
        return ReadResult(payload=None, source="primary", error="missing primary")

    def base_reader(name: str) -> ReadResult:
        if name == "a.jpg":
            return ReadResult(payload=payload_base, source="base")
        return ReadResult(payload=b"base-b", source="base")

    def original_reader(name: str) -> ReadResult:
        return ReadResult(payload=b"orig", source="original")

    stats = finalize_zip_with_verification(
        output_zip=out_zip,
        expected_names=expected,
        primary_reader=primary_reader,
        base_reader=base_reader,
        original_reader=original_reader,
        repair_mode="best_effort",
        expected_count=2,
    )

    assert stats.repair_count == 1
    assert stats.repair_by_source["base"] == 1
    assert stats.repair_by_source["original"] == 0
    assert _zip_member_bytes(out_zip, "a.jpg") == payload_base
    assert _zip_member_bytes(out_zip, "b.jpg") == payload_primary


def test_finalize_uses_original_when_primary_and_base_fail(tmp_path: Path) -> None:
    out_zip = tmp_path / "out.zip"
    expected = ["a.jpg"]
    payload_original = b"original-a"

    stats = finalize_zip_with_verification(
        output_zip=out_zip,
        expected_names=expected,
        primary_reader=lambda _: ReadResult(payload=None, source="primary", error="primary fail"),
        base_reader=lambda _: ReadResult(payload=None, source="base", error="base fail"),
        original_reader=lambda _: ReadResult(payload=payload_original, source="original"),
        repair_mode="best_effort",
        expected_count=1,
    )

    assert stats.repair_count == 1
    assert stats.repair_by_source["base"] == 0
    assert stats.repair_by_source["original"] == 1
    assert _zip_member_bytes(out_zip, "a.jpg") == payload_original


def test_finalize_raises_on_unrecoverable_payload(tmp_path: Path) -> None:
    out_zip = tmp_path / "out.zip"
    expected = ["a.jpg"]

    with pytest.raises(RuntimeError):
        finalize_zip_with_verification(
            output_zip=out_zip,
            expected_names=expected,
            primary_reader=lambda _: ReadResult(payload=None, source="primary", error="primary fail"),
            base_reader=lambda _: ReadResult(payload=None, source="base", error="base fail"),
            original_reader=lambda _: ReadResult(payload=None, source="original", error="orig fail"),
            repair_mode="best_effort",
            expected_count=1,
        )


def test_finalize_rejects_duplicate_expected_names(tmp_path: Path) -> None:
    out_zip = tmp_path / "out.zip"
    expected = ["a.jpg", "a.jpg"]

    with pytest.raises(RuntimeError):
        finalize_zip_with_verification(
            output_zip=out_zip,
            expected_names=expected,
            primary_reader=lambda _: ReadResult(payload=b"x", source="primary"),
            base_reader=lambda _: ReadResult(payload=b"x", source="base"),
            original_reader=lambda _: ReadResult(payload=b"x", source="original"),
            repair_mode="best_effort",
            expected_count=2,
        )
