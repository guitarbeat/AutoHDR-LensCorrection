#!/usr/bin/env python3
"""Kaggle Notebook runner via MCP HTTP endpoint."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from backend.config import get_config, require_kaggle_token


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_sse_json(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if not payload:
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


class KaggleMCPClient:
    def __init__(self, url: str, token: str, timeout: int = 180):
        self.url = url
        self.token = token
        self.timeout = timeout

    def _rpc(self, payload: dict) -> dict:
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.token}",
        }
        response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        events = parse_sse_json(response.text)
        if not events:
            raise RuntimeError(f"No MCP data events returned. Raw response: {response.text[:500]}")
        event = events[-1]
        if "error" in event:
            raise RuntimeError(f"MCP error: {event['error']}")
        return event

    def initialize(self) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "autohdr", "version": "1.0"},
            },
        }
        return self._rpc(payload)

    def call_tool(self, name: str, arguments: Optional[dict] = None, request_id: int = 2) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }
        return self._rpc(payload)


def decode_tool_text(tool_response: dict) -> dict:
    result = tool_response.get("result", {})
    if result.get("isError"):
        raise RuntimeError(f"Kaggle MCP tool call failed: {result.get('content')}")
    content = result.get("content") or []
    if not content:
        return {}
    text = content[0].get("text", "")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def get_notebook_info(client: KaggleMCPClient, owner: str, slug: str) -> dict:
    resp = client.call_tool(
        "get_notebook_info",
        {"request": {"userName": owner, "kernelSlug": slug}},
        request_id=11,
    )
    return decode_tool_text(resp)


def get_notebook_status(client: KaggleMCPClient, owner: str, slug: str) -> dict:
    resp = client.call_tool(
        "get_notebook_session_status",
        {"request": {"userName": owner, "kernelSlug": slug}},
        request_id=12,
    )
    return decode_tool_text(resp)


def get_notebook_session_output(client: KaggleMCPClient, owner: str, slug: str) -> dict:
    try:
        resp = client.call_tool(
            "list_notebook_session_output",
            {"request": {"userName": owner, "kernelSlug": slug}},
            request_id=14,
        )
    except requests.HTTPError as exc:
        # Kaggle MCP can return 400 once session output is unavailable in terminal states.
        if exc.response is not None and exc.response.status_code == 400:
            return {
                "error": "session_output_unavailable",
                "detail": str(exc),
            }
        raise
    return decode_tool_text(resp)


def parse_session_log_entries(session_output: dict) -> list[dict]:
    raw_log = session_output.get("log")
    if not raw_log:
        return []
    if isinstance(raw_log, list):
        return raw_log
    if isinstance(raw_log, str):
        try:
            parsed = json.loads(raw_log)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def notebook_summary(info: dict) -> dict:
    metadata = info.get("metadata") or {}
    return {
        "ref": metadata.get("ref"),
        "url": f"https://www.kaggle.com/code/{metadata.get('ref')}" if metadata.get("ref") else None,
        "kernel_id": metadata.get("id"),
        "slug": metadata.get("slug"),
        "title": metadata.get("title"),
        "current_version_number": metadata.get("current_version_number"),
    }


def run_notebook(
    client: KaggleMCPClient,
    owner: str,
    slug: str,
    enable_gpu: bool,
    local_notebook_path: Optional[str] = None,
) -> dict:
    info = get_notebook_info(client, owner, slug)
    metadata = info.get("metadata") or {}
    if not metadata:
        raise RuntimeError("Notebook metadata not returned by get_notebook_info.")

    source = (info.get("blob") or {}).get("source")
    source_origin = "kaggle_remote"
    if local_notebook_path:
        nb_path = Path(local_notebook_path).expanduser().resolve()
        if not nb_path.exists():
            raise FileNotFoundError(f"Local notebook path does not exist: {nb_path}")
        source = nb_path.read_text()
        source_origin = str(nb_path)

    if not source:
        raise RuntimeError("Notebook metadata/source not returned by get_notebook_info.")

    request = {
        "id": metadata["id"],
        "hasId": True,
        "slug": metadata.get("slug", slug),
        "hasSlug": True,
        "text": source,
        "hasText": True,
        "language": metadata.get("language", "python"),
        "hasLanguage": True,
        "kernelType": metadata.get("kernel_type", "notebook"),
        "hasKernelType": True,
        "enableGpu": bool(enable_gpu),
        "hasEnableGpu": True,
        "enableInternet": bool(metadata.get("enable_internet", True)),
        "hasEnableInternet": True,
        "kernelExecutionType": "SaveAndRunAll",
        "hasKernelExecutionType": True,
        # Ensure data sources are attached for every run.
        "competitionDataSourcesSetter": ["automatic-lens-correction"],
        "competitionDataSources": ["automatic-lens-correction"],
        "datasetDataSourcesSetter": ["hmnshudhmn24/automatic-lens-correction"],
        "datasetDataSources": ["hmnshudhmn24/automatic-lens-correction"],
        "kernelDataSourcesSetter": ["hmnshudhmn24/automatic-lens-correction"],
        "kernelDataSources": ["hmnshudhmn24/automatic-lens-correction"],
    }

    save_resp = client.call_tool("save_notebook", {"request": request}, request_id=13)
    save_info = decode_tool_text(save_resp)

    return {
        "timestamp_utc": utc_now(),
        "owner": owner,
        "slug": slug,
        "enable_gpu": enable_gpu,
        "source_origin": source_origin,
        "notebook": notebook_summary(info),
        "kernel_id": metadata.get("id"),
        "save_result": save_info,
    }


def run_and_wait(
    client: KaggleMCPClient,
    owner: str,
    slug: str,
    enable_gpu: bool,
    wait_timeout_min: int,
    local_notebook_path: Optional[str] = None,
    poll_interval_sec: int = 15,
) -> dict:
    run_summary = run_notebook(
        client,
        owner,
        slug,
        enable_gpu,
        local_notebook_path=local_notebook_path,
    )
    deadline = time.time() + (wait_timeout_min * 60)
    status_history: list[dict] = []
    final_status = "UNKNOWN"
    terminal = {"COMPLETE", "FAILED", "ERROR", "CANCELLED", "CANCELED"}
    final_status_payload: dict[str, Any] = {}

    while True:
        status_payload = get_notebook_status(client, owner, slug)
        final_status_payload = status_payload
        status = str(status_payload.get("status", "UNKNOWN"))
        status_history.append(
            {
                "timestamp_utc": utc_now(),
                "status": status,
                "failure_message": status_payload.get("failure_message"),
            }
        )
        final_status = status
        if status in terminal:
            break
        if time.time() >= deadline:
            final_status = "TIMEOUT"
            break
        time.sleep(poll_interval_sec)

    session_output = get_notebook_session_output(client, owner, slug)
    log_entries = parse_session_log_entries(session_output)

    return {
        **run_summary,
        "wait_timeout_min": wait_timeout_min,
        "final_status": final_status,
        "final_status_payload": final_status_payload,
        "status_history": status_history,
        "session_output_summary": {
            "has_log": bool(session_output.get("log")),
            "log_entry_count": len(log_entries),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kaggle notebooks via MCP")
    parser.add_argument("--owner", default="alwoods", help="Kaggle owner username")
    parser.add_argument("--slug", default="train-unet-lens-correction", help="Notebook slug")
    parser.add_argument("--enable-gpu", action="store_true", help="Enable GPU during save-and-run")
    parser.add_argument("--wait-timeout-min", type=int, default=60, help="Max wait for run-and-wait")
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=15,
        help="Polling interval while waiting for terminal status",
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=60,
        help="Lines to include for session-output log tail",
    )
    parser.add_argument(
        "--local-notebook-path",
        default=None,
        help="Optional local .ipynb file to upload before running",
    )
    parser.add_argument(
        "command",
        choices=["status", "run", "run-and-wait", "session-output"],
        help="Action to execute",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    token = require_kaggle_token(cfg)

    client = KaggleMCPClient(url=cfg.kaggle_mcp_url, token=token)
    client.initialize()

    if args.command == "status":
        info = get_notebook_info(client, args.owner, args.slug)
        result = {
            "timestamp_utc": utc_now(),
            "owner": args.owner,
            "slug": args.slug,
            "notebook": notebook_summary(info),
            "status": get_notebook_status(client, args.owner, args.slug),
        }
    elif args.command == "run":
        result = run_notebook(
            client,
            args.owner,
            args.slug,
            args.enable_gpu,
            local_notebook_path=args.local_notebook_path,
        )
    elif args.command == "run-and-wait":
        result = run_and_wait(
            client,
            owner=args.owner,
            slug=args.slug,
            enable_gpu=args.enable_gpu,
            wait_timeout_min=args.wait_timeout_min,
            local_notebook_path=args.local_notebook_path,
            poll_interval_sec=args.poll_interval_sec,
        )
    else:
        output = get_notebook_session_output(client, args.owner, args.slug)
        entries = parse_session_log_entries(output)
        tail_entries = entries[-max(args.log_tail_lines, 0) :]
        tail_lines = [str(entry.get("data", "")).rstrip("\n") for entry in tail_entries if entry.get("data")]
        result = {
            "timestamp_utc": utc_now(),
            "owner": args.owner,
            "slug": args.slug,
            "entry_count": len(entries),
            "log_tail_lines": tail_lines,
            "output_keys": sorted(output.keys()),
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
