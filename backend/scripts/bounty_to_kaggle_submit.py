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
import zipfile
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
        "--allow-unverified-lineage",
        action="store_true",
        help=(
            "Bypass strict lineage checks (default: block Kaggle submit unless scored CSV "
            "exactly matches a 1000-JPG source zip)."
        ),
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
        has_kaggle_metadata = bool(
            entry.get("kaggle_submission_timestamp")
            or entry.get("kaggle_status")
            or entry.get("kaggle_submission_message")
            or (
                "kaggle_public_score" in entry
                and entry.get("kaggle_public_score") is not None
            )
        )
        if has_kaggle_metadata:
            return entry
    return None


def reset_ledger_entry_run_state(entry: dict[str, Any]) -> None:
    entry["bounty_request_id"] = None
    entry["kaggle_submission_timestamp"] = None
    entry["kaggle_public_score"] = None
    entry["kaggle_status"] = None
    entry["kaggle_submission_message"] = None
    entry["kaggle_submit_invoked_at"] = None
    entry["lineage_verified"] = False
    entry["lineage_verification_skipped"] = False
    entry["lineage_zip_jpg_count"] = None
    entry["lineage_csv_row_count"] = None
    entry["lineage_summary_avg_score_normalized"] = None


def _sample(values: set[str], limit: int = 5) -> list[str]:
    return sorted(values)[:limit]


def load_zip_image_ids(zip_path: Path) -> tuple[set[str], int, int]:
    image_ids: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if not lower.endswith(".jpg"):
                continue
            file_name = Path(name).name
            if not file_name:
                continue
            image_ids.append(Path(file_name).stem)
    unique_ids = set(image_ids)
    duplicate_count = len(image_ids) - len(unique_ids)
    return unique_ids, len(image_ids), duplicate_count


def load_scored_csv_stats(csv_path: Path) -> tuple[set[str], int, int, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Scored CSV has no header row: {csv_path}")
        fieldnames = {name.strip() for name in reader.fieldnames if name}
        required = {"image_id", "score"}
        if not required.issubset(fieldnames):
            raise RuntimeError(
                "Scored CSV is missing required columns "
                f"{sorted(required)}: {csv_path} has {sorted(fieldnames)}"
            )
        ids: list[str] = []
        score_sum = 0.0
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                raise RuntimeError(f"Scored CSV contains blank image_id: {csv_path}")
            score_text = (row.get("score") or "").strip()
            if not score_text:
                raise RuntimeError(f"Scored CSV contains blank score: {csv_path}")
            try:
                score_value = float(score_text)
            except ValueError as exc:
                raise RuntimeError(
                    f"Scored CSV contains non-numeric score '{score_text}' in {csv_path}"
                ) from exc
            if score_value < 0.0 or score_value > 100.0:
                raise RuntimeError(
                    f"Scored CSV contains out-of-range score {score_value:.4f} in {csv_path} "
                    "(expected range: 0..100)"
                )
            ids.append(image_id)
            score_sum += score_value
    unique_ids = set(ids)
    duplicate_count = len(ids) - len(unique_ids)
    return unique_ids, len(ids), duplicate_count, score_sum


def verify_submission_lineage(zip_path: Path, scored_csv_path: Path) -> dict[str, int]:
    if not zip_path.exists():
        raise RuntimeError(f"Missing source zip for lineage verification: {zip_path}")
    if not scored_csv_path.exists():
        raise RuntimeError(f"Missing scored CSV for lineage verification: {scored_csv_path}")

    zip_ids, zip_row_count, zip_duplicate_count = load_zip_image_ids(zip_path)
    csv_ids, csv_row_count, csv_duplicate_count, _ = load_scored_csv_stats(scored_csv_path)

    if zip_row_count != 1000:
        raise RuntimeError(
            f"Lineage check failed: source zip must contain exactly 1000 JPG files, found {zip_row_count} in {zip_path}"
        )
    if zip_duplicate_count:
        raise RuntimeError(
            "Lineage check failed: source zip contains duplicate image IDs "
            f"(count={zip_duplicate_count}) in {zip_path}"
        )
    if csv_row_count != 1000:
        raise RuntimeError(
            f"Lineage check failed: scored CSV must contain exactly 1000 rows, found {csv_row_count} in {scored_csv_path}"
        )
    if csv_duplicate_count:
        raise RuntimeError(
            "Lineage check failed: scored CSV contains duplicate image_id rows "
            f"(count={csv_duplicate_count}) in {scored_csv_path}"
        )

    missing_ids = zip_ids - csv_ids
    extra_ids = csv_ids - zip_ids
    if missing_ids or extra_ids:
        raise RuntimeError(
            "Lineage check failed: scored CSV image set does not match ZIP image set "
            f"(missing_from_csv={len(missing_ids)} sample={_sample(missing_ids)} "
            f"extra_in_csv={len(extra_ids)} sample={_sample(extra_ids)})"
        )

    print(
        "Lineage verified:"
        f" zip_jpg_count={zip_row_count} csv_rows={csv_row_count} matched_ids={len(zip_ids)}"
    )
    return {
        "zip_jpg_count": zip_row_count,
        "csv_row_count": csv_row_count,
        "matched_ids": len(zip_ids),
    }


def verify_csv_matches_bounty_summary(scored_csv_path: Path, summary_json_path: Path) -> dict[str, float]:
    if not summary_json_path.exists():
        raise RuntimeError(
            "Lineage check failed: bounty summary JSON missing; cannot verify scored CSV provenance "
            f"({summary_json_path})"
        )
    try:
        payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Lineage check failed: invalid bounty summary JSON ({summary_json_path})"
        ) from exc

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(
            f"Lineage check failed: bounty summary JSON missing 'summary' object ({summary_json_path})"
        )

    total_images = int(summary.get("total_images", 0) or 0)
    scored_images = int(summary.get("scored", 0) or 0)
    missing_images = int(summary.get("missing", 0) or 0)
    if total_images != 1000 or scored_images != 1000 or missing_images != 0:
        raise RuntimeError(
            "Lineage check failed: bounty summary image counts are invalid "
            f"(total={total_images}, scored={scored_images}, missing={missing_images})"
        )

    csv_ids, csv_row_count, csv_duplicate_count, csv_score_sum = load_scored_csv_stats(scored_csv_path)
    if csv_duplicate_count:
        raise RuntimeError(
            "Lineage check failed: duplicate image_id rows detected during summary consistency check "
            f"(count={csv_duplicate_count})"
        )
    if csv_row_count != scored_images:
        raise RuntimeError(
            "Lineage check failed: scored CSV row count does not match bounty summary "
            f"(csv_rows={csv_row_count}, summary_scored={scored_images})"
        )

    per_image = summary.get("per_image")
    if isinstance(per_image, list) and per_image:
        summary_ids = {str(item.get("image_id", "")).strip() for item in per_image if item.get("image_id")}
        if len(summary_ids) != len(per_image):
            raise RuntimeError(
                "Lineage check failed: bounty summary contains duplicate/blank per_image IDs"
            )
        missing_ids = summary_ids - csv_ids
        extra_ids = csv_ids - summary_ids
        if missing_ids or extra_ids:
            raise RuntimeError(
                "Lineage check failed: scored CSV ID set does not match bounty summary ID set "
                f"(missing_from_csv={len(missing_ids)} sample={_sample(missing_ids)} "
                f"extra_in_csv={len(extra_ids)} sample={_sample(extra_ids)})"
            )

    summary_avg_score = float(summary.get("avg_score", 0.0) or 0.0)
    csv_avg_score_normalized = (csv_score_sum / csv_row_count) / 100.0 if csv_row_count else 0.0
    if abs(csv_avg_score_normalized - summary_avg_score) > 0.01:
        raise RuntimeError(
            "Lineage check failed: scored CSV mean does not match bounty summary mean "
            f"(csv_avg_norm={csv_avg_score_normalized:.6f}, summary_avg={summary_avg_score:.6f})"
        )

    print(
        "Bounty summary verified:"
        f" rows={csv_row_count} avg_norm={csv_avg_score_normalized:.6f}"
        f" request_id={payload.get('request_id')}"
    )
    return {
        "csv_row_count": float(csv_row_count),
        "csv_avg_score_normalized": csv_avg_score_normalized,
        "summary_avg_score_normalized": summary_avg_score,
    }


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
    summary_path = (
        args.out_summary_json.expanduser().resolve()
        if args.out_summary_json is not None
        else out_csv.with_name(f"{out_csv.stem}_summary_autogen.json")
    )
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
        reset_ledger_entry_run_state(ledger_entry)
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

    lineage_metrics: dict[str, int] | None = None
    summary_metrics: dict[str, float] | None = None
    if args.allow_unverified_lineage:
        print("WARNING: --allow-unverified-lineage set; skipping strict lineage verification.")
        if ledger_path is not None and ledger is not None and ledger_entry is not None:
            ledger_entry["lineage_verified"] = False
            ledger_entry["lineage_verification_skipped"] = True
            save_ledger(ledger_path, ledger)
    else:
        lineage_metrics = verify_submission_lineage(zip_file, out_csv)
        summary_metrics = verify_csv_matches_bounty_summary(out_csv, summary_path)
        if ledger_path is not None and ledger is not None and ledger_entry is not None:
            ledger_entry["lineage_verified"] = True
            ledger_entry["lineage_verification_skipped"] = False
            ledger_entry["lineage_zip_jpg_count"] = lineage_metrics["zip_jpg_count"]
            ledger_entry["lineage_csv_row_count"] = lineage_metrics["csv_row_count"]
            ledger_entry["lineage_summary_avg_score_normalized"] = summary_metrics[
                "summary_avg_score_normalized"
            ]
            save_ledger(ledger_path, ledger)

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
    if ledger_path is not None and ledger is not None and ledger_entry is not None:
        ledger_entry["kaggle_submission_message"] = message
        ledger_entry["kaggle_submit_invoked_at"] = datetime.utcnow().isoformat() + "Z"
        save_ledger(ledger_path, ledger)

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
        save_ledger(ledger_path, ledger)
        print(f"Ledger updated: {ledger_path}")


if __name__ == "__main__":
    main()
