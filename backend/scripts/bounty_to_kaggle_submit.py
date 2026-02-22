#!/usr/bin/env python3
"""Upload zip to bounty, save scored CSV, then submit to Kaggle competition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional at runtime
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNTY_SUBMIT_SCRIPT = Path(__file__).resolve().with_name("bounty_submit.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounty scoring for a zip artifact and submit resulting CSV to Kaggle."
    )
    parser.add_argument(
        "--zip-file",
        required=True,
        type=Path,
        help="Path to submission zip (1000 corrected JPGs).",
    )
    parser.add_argument(
        "--team-name",
        default=os.getenv("AUTOHDR_BOUNTY_TEAM_NAME", "Aaron"),
        help="Bounty team name.",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("AUTOHDR_BOUNTY_EMAIL", "aaronlorenzowoods@gmail.com"),
        help="Bounty contact email.",
    )
    parser.add_argument(
        "--kaggle-username",
        default=os.getenv("AUTOHDR_BOUNTY_KAGGLE_USERNAME", "alwoods"),
        help="Kaggle username for bounty form.",
    )
    parser.add_argument(
        "--github-repo",
        default=os.getenv("AUTOHDR_BOUNTY_GITHUB_REPO", ""),
        help="Optional GitHub repo URL for bounty form.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "submission.csv",
        help="Where to write scored CSV (default: repo-root submission.csv).",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON output path.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=3.0,
        help="Bounty polling interval in seconds.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=660.0,
        help="Bounty scoring timeout in seconds.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--competition",
        default="automatic-lens-correction",
        help="Kaggle competition slug (default: automatic-lens-correction).",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Kaggle submission message.",
    )
    parser.add_argument(
        "--skip-kaggle-submit",
        action="store_true",
        help="Only run bounty scoring; do not submit to Kaggle.",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help="Optional ledger JSON path for idempotency and run tracking.",
    )
    parser.add_argument(
        "--candidate-name",
        default=None,
        help="Candidate name used to update ledger entry (defaults to zip stem).",
    )
    parser.add_argument(
        "--force-duplicate",
        action="store_true",
        help="Allow re-submitting a SHA that is already recorded as submitted.",
    )
    parser.add_argument(
        "--kaggle-poll-attempts",
        type=int,
        default=18,
        help="Number of polls for matching Kaggle submission metadata.",
    )
    parser.add_argument(
        "--kaggle-poll-interval-sec",
        type=float,
        default=10.0,
        help="Polling interval for Kaggle submission metadata.",
    )
    return parser.parse_args()


def run_checked(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(shlex.quote(part) for part in cmd)}")
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def build_default_message(zip_path: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Auto submission from {zip_path.name} at {ts}"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def find_entry_for_update(entries: list[dict[str, Any]], candidate_name: str, sha256: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("candidate_name") == candidate_name:
            return entry
    for entry in entries:
        if entry.get("sha256") == sha256:
            return entry
    return None


def has_submitted_duplicate(entries: list[dict[str, Any]], sha256: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("sha256") != sha256:
            continue
        if entry.get("kaggle_submission_timestamp") or entry.get("bounty_request_id"):
            return entry
    return None


def parse_bounty_request_id(summary_json: Path | None) -> str | None:
    if summary_json is None or not summary_json.exists():
        return None
    try:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    request_id = payload.get("request_id")
    return str(request_id) if request_id else None


def parse_kaggle_rows(csv_text: str) -> list[dict[str, str]]:
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return [dict(row) for row in reader]


def fetch_latest_kaggle_row(
    *,
    competition: str,
    file_name: str,
    message: str,
    attempts: int,
    interval_sec: float,
) -> dict[str, str] | None:
    cmd = ["kaggle", "competitions", "submissions", competition, "--csv", "--page-size", "200"]
    latest_row: dict[str, str] | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        result = run_checked(cmd, capture_output=True)
        rows = parse_kaggle_rows(result.stdout)
        matching = [
            row
            for row in rows
            if row.get("fileName") == file_name and row.get("description") == message
        ]
        if matching:
            matching.sort(key=lambda r: (r.get("date") or ""), reverse=True)
            latest_row = matching[0]
            status = latest_row.get("status", "")
            if status == "SubmissionStatus.COMPLETE":
                return latest_row
        if attempt < attempts:
            time.sleep(interval_sec)
    return latest_row


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    args = parse_args()
    zip_file = args.zip_file.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_summary_json.expanduser().resolve() if args.out_summary_json is not None else None
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_name = args.candidate_name or zip_file.stem
    zip_sha = compute_sha256(zip_file)

    ledger_path = args.ledger_path.expanduser().resolve() if args.ledger_path is not None else None
    ledger: dict[str, Any] | None = None
    ledger_entry: dict[str, Any] | None = None
    if ledger_path is not None:
        ledger = load_ledger(ledger_path)
        entries = ledger.setdefault("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Invalid ledger schema at {ledger_path}: 'entries' must be a list")
        duplicate_entry = has_submitted_duplicate(entries, zip_sha)
        if duplicate_entry and not args.force_duplicate:
            raise RuntimeError(
                "Duplicate SHA already submitted via ledger guard. "
                f"candidate={duplicate_entry.get('candidate_name')} sha256={zip_sha}"
            )
        ledger_entry = find_entry_for_update(entries, candidate_name, zip_sha)
        if ledger_entry is None:
            ledger_entry = {
                "candidate_name": candidate_name,
                "artifact_path": str(zip_file),
                "parent_artifacts": {"zip_file": str(zip_file)},
                "rule_expression": "manual_or_external_candidate",
                "replacement_count": None,
                "sha256": zip_sha,
                "bounty_request_id": None,
                "kaggle_submission_timestamp": None,
                "kaggle_public_score": None,
            }
            entries.append(ledger_entry)
        else:
            ledger_entry["candidate_name"] = candidate_name
            ledger_entry["artifact_path"] = str(zip_file)
            ledger_entry["sha256"] = zip_sha
        save_ledger(ledger_path, ledger)

    bounty_cmd = [
        sys.executable,
        str(BOUNTY_SUBMIT_SCRIPT),
        "--zip-file",
        str(zip_file),
        "--team-name",
        args.team_name,
        "--email",
        args.email,
        "--kaggle-username",
        args.kaggle_username,
        "--github-repo",
        args.github_repo,
        "--out-csv",
        str(out_csv),
        "--poll-interval-sec",
        str(args.poll_interval_sec),
        "--timeout-sec",
        str(args.timeout_sec),
        "--request-timeout-sec",
        str(args.request_timeout_sec),
    ]
    if summary_path is not None:
        bounty_cmd.extend(["--out-summary-json", str(summary_path)])

    run_checked(bounty_cmd)
    print(f"Bounty scoring complete: {out_csv}")

    bounty_request_id = parse_bounty_request_id(summary_path)
    if ledger_path is not None and ledger is not None and ledger_entry is not None:
        if bounty_request_id is not None:
            ledger_entry["bounty_request_id"] = bounty_request_id
        save_ledger(ledger_path, ledger)

    if args.skip_kaggle_submit:
        print("Skipping Kaggle submit as requested.")
        return

    message = args.message or build_default_message(zip_file)
    kaggle_cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        args.competition,
        "-f",
        str(out_csv),
        "-m",
        message,
    ]
    run_checked(kaggle_cmd)
    print("Kaggle submission command completed.")

    row = fetch_latest_kaggle_row(
        competition=args.competition,
        file_name=out_csv.name,
        message=message,
        attempts=args.kaggle_poll_attempts,
        interval_sec=args.kaggle_poll_interval_sec,
    )
    if row:
        print(
            "Kaggle row:"
            f" date={row.get('date')} status={row.get('status')} publicScore={row.get('publicScore')}"
        )
    else:
        print("Kaggle row not found in submissions poll window.")

    if ledger_path is not None and ledger is not None and ledger_entry is not None:
        if row is not None:
            ledger_entry["kaggle_submission_timestamp"] = row.get("date")
            public_score = row.get("publicScore")
            ledger_entry["kaggle_public_score"] = float(public_score) if public_score else None
            ledger_entry["kaggle_status"] = row.get("status")
        ledger_entry["kaggle_submission_message"] = message
        save_ledger(ledger_path, ledger)
        print(f"Ledger updated: {ledger_path}")


if __name__ == "__main__":
    main()
