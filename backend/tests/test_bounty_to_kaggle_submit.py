from backend.scripts.bounty_to_kaggle_submit import has_submitted_duplicate


def test_duplicate_guard_ignores_bounty_only_entry() -> None:
    entries = [
        {
            "candidate_name": "zpos",
            "sha256": "abc123",
            "bounty_request_id": "req-001",
        }
    ]
    assert has_submitted_duplicate(entries, "abc123") is None


def test_duplicate_guard_matches_when_kaggle_message_exists() -> None:
    entries = [
        {
            "candidate_name": "zpos",
            "sha256": "abc123",
            "kaggle_submission_message": "real zip run",
        }
    ]
    assert has_submitted_duplicate(entries, "abc123") == entries[0]


def test_duplicate_guard_matches_when_public_score_recorded_even_zero() -> None:
    entries = [
        {
            "candidate_name": "zpos",
            "sha256": "abc123",
            "kaggle_public_score": 0.0,
        }
    ]
    assert has_submitted_duplicate(entries, "abc123") == entries[0]
