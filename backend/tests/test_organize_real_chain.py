from pathlib import Path

import pytest

from backend.scripts.heuristics.organize_real_chain import (
    collect_score_csvs,
    dedupe_score_csvs_by_stem,
    resolve_kaggle_dir,
    stem_from_score_csv_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("submission_v4.csv", "submission_v4"),
        ("submission_v4_mix_zpos_scored.csv", "submission_v4_mix_zpos"),
        (
            "submission_v4_oracle_valid_allzip_20260222_175058_scored_20260222_121503.csv",
            "submission_v4_oracle_valid_allzip_20260222_175058",
        ),
        (
            "submission_v4_oracle_allbest_20260222_145359_rescored_20260222_121503.csv",
            "submission_v4_oracle_allbest_20260222_145359",
        ),
        ("submission_v4_rescore.csv", "submission_v4"),
        ("not_a_submission.txt", None),
    ],
)
def test_stem_from_score_csv_name_supports_timestamped_variants(
    name: str,
    expected: str | None,
) -> None:
    assert stem_from_score_csv_name(name) == expected


def test_resolve_kaggle_dir_allows_missing_path_when_skipping_kaggle(tmp_path: Path) -> None:
    missing = tmp_path / "kaggle_outputs_missing"
    resolved = resolve_kaggle_dir(missing, skip_kaggle=True)
    assert resolved == missing.resolve()


def test_resolve_kaggle_dir_requires_existing_path_when_not_skipping(tmp_path: Path) -> None:
    missing = tmp_path / "kaggle_outputs_missing"
    with pytest.raises(FileNotFoundError):
        resolve_kaggle_dir(missing, skip_kaggle=False)


def test_collect_score_csvs_merges_multiple_dirs_without_duplicates(tmp_path: Path) -> None:
    bounty_dir = tmp_path / "bounty"
    kaggle_dir = tmp_path / "kaggle"
    bounty_dir.mkdir()
    kaggle_dir.mkdir()

    # Same filename in both dirs should still yield two unique paths
    (bounty_dir / "submission_a_scored.csv").write_text("image_id,score\nx,1\n", encoding="utf-8")
    (kaggle_dir / "submission_a_scored.csv").write_text("image_id,score\nx,1\n", encoding="utf-8")
    (kaggle_dir / "submission_b_scored_20260222_121503.csv").write_text(
        "image_id,score\nx,1\n",
        encoding="utf-8",
    )

    collected = collect_score_csvs([bounty_dir, kaggle_dir])
    collected_set = {str(path) for path in collected}

    assert str((bounty_dir / "submission_a_scored.csv").resolve()) in collected_set
    assert str((kaggle_dir / "submission_a_scored.csv").resolve()) in collected_set
    assert str((kaggle_dir / "submission_b_scored_20260222_121503.csv").resolve()) in collected_set
    assert len(collected_set) == 3


def test_dedupe_score_csvs_by_stem_keeps_latest_mtime(tmp_path: Path) -> None:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()

    older = score_dir / "submission_x_scored.csv"
    newer = score_dir / "submission_x_scored_20260222_121503.csv"
    older.write_text("image_id,score\nx,1\n", encoding="utf-8")
    newer.write_text("image_id,score\nx,1\n", encoding="utf-8")

    # Force deterministic mtime ordering.
    older_mtime = 1_700_000_000
    newer_mtime = older_mtime + 10
    older.touch()
    newer.touch()
    older.chmod(0o644)
    newer.chmod(0o644)
    import os

    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    selected, duplicates = dedupe_score_csvs_by_stem([older, newer])

    assert len(selected) == 1
    assert selected[0].resolve() == newer.resolve()
    assert len(duplicates) == 1
    assert duplicates[0]["stem"] == "submission_x"
    assert duplicates[0]["skipped_path"] == str(older.resolve())
    assert duplicates[0]["kept_path"] == str(newer.resolve())
