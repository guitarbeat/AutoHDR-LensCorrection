#!/usr/bin/env python3
"""Submit a zipped prediction artifact to bounty.autohdr.com and download scored CSV."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://bounty.autohdr.com"
DEFAULT_POLL_INTERVAL_SEC = 3.0
DEFAULT_TIMEOUT_SEC = 660.0
DEFAULT_REQUEST_TIMEOUT_SEC = 300.0

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
load_dotenv(override=False)


class SubmissionError(RuntimeError):
    """Raised when remote submission flow fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a submission zip to bounty.autohdr.com and save scored CSV."
    )
    parser.add_argument(
        "--zip-file",
        required=True,
        type=Path,
        help="Path to submission zip (contains 1000 corrected JPGs).",
    )
    parser.add_argument(
        "--team-name",
        default=os.getenv("AUTOHDR_BOUNTY_TEAM_NAME"),
        help="Team name shown in bounty UI (or AUTOHDR_BOUNTY_TEAM_NAME).",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("AUTOHDR_BOUNTY_EMAIL", ""),
        help="Contact email (or AUTOHDR_BOUNTY_EMAIL).",
    )
    parser.add_argument(
        "--kaggle-username",
        default=os.getenv("AUTOHDR_BOUNTY_KAGGLE_USERNAME", ""),
        help="Kaggle username (or AUTOHDR_BOUNTY_KAGGLE_USERNAME).",
    )
    parser.add_argument(
        "--github-repo",
        default=os.getenv("AUTOHDR_BOUNTY_GITHUB_REPO", ""),
        help="Optional repo URL (or AUTOHDR_BOUNTY_GITHUB_REPO).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Bounty base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"Polling interval while waiting for score (default: {DEFAULT_POLL_INTERVAL_SEC}).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Max wait for scoring to complete (default: {DEFAULT_TIMEOUT_SEC}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <zip_stem>_scored_<timestamp>.csv.",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        default=None,
        help="Optional path to write scoring summary JSON payload.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
        help="Per-request timeout in seconds for API calls.",
    )
    return parser.parse_args()


def parse_response_json(resp: requests.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except ValueError as exc:
        body = resp.text.strip()
        snippet = body[:400] if body else "<empty>"
        raise SubmissionError(f"Expected JSON response but got: {snippet}") from exc


def require_ok(resp: requests.Response, context: str) -> dict[str, Any]:
    payload = parse_response_json(resp)
    if resp.ok:
        return payload
    error = payload.get("detail") or payload.get("error") or payload
    raise SubmissionError(f"{context} failed ({resp.status_code}): {error}")


def upload_zip(
    session: requests.Session,
    base_url: str,
    zip_path: Path,
    team_name: str,
    request_timeout_sec: float,
) -> str:
    print("1/4 Requesting upload URL...")
    url_resp = session.post(
        f"{base_url}/api/upload-url",
        json={"team_name": team_name},
        timeout=request_timeout_sec,
    )
    upload_payload = require_ok(url_resp, "Upload URL request")
    upload_url = upload_payload.get("upload_url")
    s3_key = upload_payload.get("s3_key")
    if not upload_url or not s3_key:
        raise SubmissionError(f"Upload URL response missing fields: {upload_payload}")

    print(f"2/4 Uploading zip: {zip_path}")
    with zip_path.open("rb") as fh:
        put_resp = session.put(
            upload_url,
            data=fh,
            headers={"Content-Type": "application/zip"},
            timeout=request_timeout_sec,
        )
    if not put_resp.ok:
        text = put_resp.text.strip()[:400]
        raise SubmissionError(
            f"Zip upload failed ({put_resp.status_code}): {text or '<empty body>'}"
        )
    return str(s3_key)


def submit_for_score(
    session: requests.Session,
    base_url: str,
    s3_key: str,
    team_name: str,
    email: str,
    kaggle_username: str,
    github_repo: str,
    request_timeout_sec: float,
) -> str:
    print("3/4 Submitting scoring request...")
    payload = {
        "s3_key": s3_key,
        "team_name": team_name,
        "kaggle_username": kaggle_username,
        "email": email,
        "github_repo": github_repo,
    }
    submit_resp = session.post(
        f"{base_url}/api/score",
        json=payload,
        timeout=request_timeout_sec,
    )
    submit_payload = require_ok(submit_resp, "Score submission")
    request_id = submit_payload.get("request_id")
    if not request_id:
        raise SubmissionError(f"Score submission missing request_id: {submit_payload}")
    return str(request_id)


def wait_for_score(
    session: requests.Session,
    base_url: str,
    request_id: str,
    poll_interval_sec: float,
    timeout_sec: float,
    request_timeout_sec: float,
) -> dict[str, Any]:
    print(f"4/4 Waiting for score (request_id={request_id})...")
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_sec:
            raise SubmissionError(
                f"Timed out after {timeout_sec:.0f}s waiting for request_id={request_id}"
            )

        resp = session.get(
            f"{base_url}/api/score",
            params={"request_id": request_id},
            timeout=request_timeout_sec,
        )
        status_payload = require_ok(resp, "Score status check")
        status = str(status_payload.get("status", "")).upper()
        detail = status_payload.get("detail")

        if status == "COMPLETED":
            if status_payload.get("success") is False:
                raise SubmissionError(
                    f"Scoring completed with failure: {detail or status_payload}"
                )
            if "csv" not in status_payload:
                raise SubmissionError(
                    f"COMPLETED response missing csv payload: {status_payload}"
                )
            print(f"Scoring completed in {elapsed:.1f}s")
            return status_payload

        if status == "FAILED" or detail:
            raise SubmissionError(f"Scoring failed: {detail or status_payload}")

        print(f"  status={status or 'PENDING'} elapsed={elapsed:.1f}s")
        time.sleep(max(0.2, poll_interval_sec))


def default_output_csv(zip_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return zip_path.with_name(f"{zip_path.stem}_scored_{ts}.csv")


def write_outputs(
    status_payload: dict[str, Any],
    request_id: str,
    out_csv: Path,
    out_summary_json: Path | None,
) -> None:
    csv_blob = status_payload.get("csv")
    if not isinstance(csv_blob, str) or not csv_blob.strip():
        raise SubmissionError("CSV payload is empty.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(csv_blob, encoding="utf-8")
    print(f"Wrote scored CSV: {out_csv}")

    if out_summary_json is not None:
        out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "request_id": request_id,
            "status": status_payload.get("status"),
            "success": status_payload.get("success"),
            "summary": status_payload.get("summary"),
        }
        out_summary_json.write_text(
            json.dumps(summary_payload, indent=2), encoding="utf-8"
        )
        print(f"Wrote summary JSON: {out_summary_json}")

    summary = status_payload.get("summary")
    if isinstance(summary, dict):
        avg_score = summary.get("avg_score")
        hard_fails = summary.get("hard_fails")
        missing = summary.get("missing")
        elapsed = summary.get("elapsed")
        print(
            "Summary:",
            f"avg_score={avg_score}",
            f"hard_fails={hard_fails}",
            f"missing={missing}",
            f"elapsed={elapsed}",
        )


def validate_args(args: argparse.Namespace) -> None:
    if not args.team_name or not str(args.team_name).strip():
        raise SubmissionError(
            "Team name is required. Pass --team-name or set AUTOHDR_BOUNTY_TEAM_NAME."
        )
    zip_path: Path = args.zip_file
    if not zip_path.exists():
        raise SubmissionError(f"Zip file not found: {zip_path}")
    if not zip_path.is_file():
        raise SubmissionError(f"Path is not a file: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise SubmissionError(f"Expected .zip file, got: {zip_path.name}")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        base_url = args.base_url.rstrip("/")
        zip_path = args.zip_file.resolve()
        out_csv = (
            args.out_csv.resolve() if args.out_csv else default_output_csv(zip_path)
        )
        out_summary_json = (
            args.out_summary_json.resolve() if args.out_summary_json else None
        )

        with requests.Session() as session:
            s3_key = upload_zip(
                session=session,
                base_url=base_url,
                zip_path=zip_path,
                team_name=args.team_name.strip(),
                request_timeout_sec=args.request_timeout_sec,
            )
            request_id = submit_for_score(
                session=session,
                base_url=base_url,
                s3_key=s3_key,
                team_name=args.team_name.strip(),
                email=args.email.strip(),
                kaggle_username=args.kaggle_username.strip(),
                github_repo=args.github_repo.strip(),
                request_timeout_sec=args.request_timeout_sec,
            )
            status_payload = wait_for_score(
                session=session,
                base_url=base_url,
                request_id=request_id,
                poll_interval_sec=args.poll_interval_sec,
                timeout_sec=args.timeout_sec,
                request_timeout_sec=args.request_timeout_sec,
            )
        write_outputs(status_payload, request_id, out_csv, out_summary_json)
    except SubmissionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
