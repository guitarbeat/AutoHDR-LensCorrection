#!/usr/bin/env python3
"""Review Akash deployment logs via provider websocket with JWT auth."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import ssl
from dataclasses import dataclass
from typing import Any

import requests
import websockets
from websockets.exceptions import ConnectionClosed


API_BASE_URL = "https://console-api.akash.network"

INTERESTING_PATTERNS = re.compile(
    r"401|403|unauthorized|forbidden|kaggle auth preflight|kaggle competitions files|"
    r"kaggle competitions download|traceback|error|failed|exception|epoch|validation|"
    r"inference|http\.server|no running pods|out of memory|killed",
    re.IGNORECASE,
)


@dataclass
class LeaseInfo:
    dseq: str
    gseq: int
    oseq: int
    provider_address: str
    provider_host_uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review Akash deployment logs and status."
    )
    parser.add_argument(
        "--dseq",
        action="append",
        required=True,
        help="Deployment sequence. Repeat flag for multiple values.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=200,
        help="Number of historical log lines to request from provider websocket.",
    )
    parser.add_argument(
        "--service",
        default="miner",
        help="Service name for lease log stream (default: miner).",
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Also stream kubernetes events in addition to logs.",
    )
    parser.add_argument(
        "--receive-timeout-sec",
        type=float,
        default=0.8,
        help="Stop stream if no message arrives within this timeout.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=400,
        help="Max websocket messages to read per stream.",
    )
    parser.add_argument(
        "--jwt-ttl-sec",
        type=int,
        default=1800,
        help="Provider JWT time-to-live in seconds.",
    )
    return parser.parse_args()


def api_request(
    api_key: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
    }
    response = requests.request(
        method=method,
        url=f"{API_BASE_URL}{path}",
        headers=headers,
        json=payload,
        timeout=timeout_sec,
    )
    if not response.ok:
        body = response.text[:600]
        raise RuntimeError(
            f"API {method} {path} failed ({response.status_code}): {body}"
        )
    return response.json()


def mint_provider_jwt(api_key: str, ttl_sec: int) -> str:
    payload = {
        "data": {
            "ttl": ttl_sec,
            "leases": {
                "access": "scoped",
                "scope": ["status", "shell", "events", "logs"],
            },
        }
    }
    data = api_request(api_key, "POST", "/v1/create-jwt-token", payload)
    token = data.get("data", {}).get("token")
    if not token or not isinstance(token, str):
        raise RuntimeError(f"JWT token missing in response: {data}")
    return token


def fetch_lease_info(api_key: str, dseq: str) -> LeaseInfo:
    data = api_request(api_key, "GET", f"/v1/deployments/{dseq}")
    leases = data.get("data", {}).get("leases") or []
    if not leases:
        raise RuntimeError(f"No leases found for dseq={dseq}")

    lease = leases[0]
    lease_id = lease.get("id") or {}
    provider_address = lease_id.get("provider")
    gseq_raw = lease_id.get("gseq")
    oseq_raw = lease_id.get("oseq")
    if not provider_address or gseq_raw is None or oseq_raw is None:
        raise RuntimeError(
            f"Lease is missing provider/gseq/oseq for dseq={dseq}: {lease_id}"
        )

    provider_info = None
    owner = lease_id.get("owner")
    if owner:
        try:
            legacy = api_request(api_key, "GET", f"/v1/deployment/{owner}/{dseq}")
            for item in legacy.get("leases", []):
                provider = item.get("provider") or {}
                address = provider.get("address")
                if address == provider_address:
                    provider_info = provider
                    break
        except Exception:
            provider_info = None

    provider_host_uri = (provider_info or {}).get("hostUri")
    if not provider_host_uri:
        try:
            provider_obj = api_request(
                api_key, "GET", f"/v1/providers/{provider_address}"
            )
            provider_host_uri = provider_obj.get("hostUri")
        except Exception:
            provider_host_uri = None
    if not provider_host_uri:
        raise RuntimeError(
            f"Unable to resolve provider hostUri for provider={provider_address}, dseq={dseq}"
        )

    return LeaseInfo(
        dseq=str(dseq),
        gseq=int(gseq_raw),
        oseq=int(oseq_raw),
        provider_address=str(provider_address),
        provider_host_uri=str(provider_host_uri),
    )


def fetch_provider_status(lease: LeaseInfo, token: str) -> tuple[int, str]:
    url = (
        f"{lease.provider_host_uri}/lease/{lease.dseq}/{lease.gseq}/{lease.oseq}/status"
    )
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "content-type": "application/json",
        },
        timeout=20,
        verify=False,
    )
    return response.status_code, response.text


async def stream_provider_ws(
    url: str,
    token: str,
    receive_timeout_sec: float,
    max_messages: int,
) -> tuple[list[str], str | None]:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    messages: list[str] = []
    error: str | None = None

    try:
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {token}"},
            ssl=ssl_ctx,
            open_timeout=10,
            close_timeout=2,
            max_size=4_000_000,
        ) as ws:
            for _ in range(max_messages):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=receive_timeout_sec)
                except asyncio.TimeoutError:
                    break
                except ConnectionClosed as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
                    break

                try:
                    parsed = json.loads(raw)
                    line = str(parsed.get("message", "")).rstrip("\n")
                except Exception:
                    line = str(raw).rstrip("\n")

                if line:
                    messages.append(line)

    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"

    return messages, error


def ws_url(lease: LeaseInfo, stream_type: str, tail: int, service: str) -> str:
    return (
        f"{lease.provider_host_uri.replace('https://', 'wss://')}"
        f"/lease/{lease.dseq}/{lease.gseq}/{lease.oseq}/{stream_type}"
        f"?follow=true&tail={tail}&service={service}"
    )


def summarize_lines(
    lines: list[str], limit_hits: int = 30, limit_tail: int = 25
) -> None:
    hits = [line for line in lines if INTERESTING_PATTERNS.search(line)]
    print(f"line_count={len(lines)} interesting_hits={len(hits)}")
    if hits:
        print("interesting_tail:")
        for line in hits[-limit_hits:]:
            print(f"  {line[:220]}")
    if lines:
        print("log_tail:")
        for line in lines[-limit_tail:]:
            print(f"  {line[:220]}")


async def review_single_dseq(
    api_key: str,
    dseq: str,
    token: str,
    tail: int,
    service: str,
    include_events: bool,
    receive_timeout_sec: float,
    max_messages: int,
) -> None:
    lease = fetch_lease_info(api_key, dseq)
    print(f"\n===== DSEQ {lease.dseq} =====")
    print(f"provider_address={lease.provider_address}")
    print(f"provider_host_uri={lease.provider_host_uri}")
    print(f"lease={lease.gseq}/{lease.oseq}")

    status_code, status_body = fetch_provider_status(lease, token)
    print(f"provider_status_http={status_code}")
    print(f"provider_status_body={status_body[:500]}")

    log_stream_url = ws_url(lease, "logs", tail, service)
    print(f"log_stream={log_stream_url}")
    log_lines, log_error = await stream_provider_ws(
        log_stream_url, token, receive_timeout_sec, max_messages
    )
    if log_error:
        print(f"log_stream_error={log_error}")
    summarize_lines(log_lines)

    if include_events:
        event_stream_url = ws_url(lease, "kubeevents", tail, service)
        print(f"event_stream={event_stream_url}")
        event_lines, event_error = await stream_provider_ws(
            event_stream_url, token, receive_timeout_sec, max_messages
        )
        if event_error:
            print(f"event_stream_error={event_error}")
        summarize_lines(event_lines, limit_hits=20, limit_tail=20)


async def main_async(args: argparse.Namespace) -> None:
    api_key = (os.getenv("AKASH_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing AKASH_API_KEY in environment.")

    token = mint_provider_jwt(api_key, ttl_sec=args.jwt_ttl_sec)
    for dseq in args.dseq:
        await review_single_dseq(
            api_key=api_key,
            dseq=str(dseq),
            token=token,
            tail=args.tail,
            service=args.service,
            include_events=args.events,
            receive_timeout_sec=args.receive_timeout_sec,
            max_messages=args.max_messages,
        )


def main() -> None:
    args = parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    # Provider endpoints use self-signed certs; requests verifies off for those calls only.
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    main()
