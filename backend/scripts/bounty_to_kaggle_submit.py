#!/usr/bin/env python3
"""Upload zip to bounty, save scored CSV, then submit to Kaggle competition."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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
    return parser.parse_args()


def run_checked(cmd: list[str]) -> None:
    print(f"+ {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, check=True)


def build_default_message(zip_path: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Auto submission from {zip_path.name} at {ts}"


def main() -> None:
    args = parse_args()
    zip_file = args.zip_file.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
    if args.out_summary_json is not None:
        summary_path = args.out_summary_json.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        bounty_cmd.extend(["--out-summary-json", str(summary_path)])

    run_checked(bounty_cmd)
    print(f"Bounty scoring complete: {out_csv}")

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


if __name__ == "__main__":
    main()
