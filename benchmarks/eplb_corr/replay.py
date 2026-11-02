# SPDX-License-Identifier: Apache-2.0
"""Closed-loop replay of a frozen session-trace request set."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in results if row["error"] is None]
    output_tokens = sum(row["output_tokens"] for row in complete)
    elapsed = (
        max(row["end_s"] for row in results)
        - min(row["start_s"] for row in results)
        if results
        else 0
    )
    summary: dict[str, Any] = {
        "requests": len(results),
        "completed": len(complete),
        "failed": len(results) - len(complete),
        "outputs_at_cap": sum(
            row["output_tokens"] >= row["max_tokens"] for row in complete
        ),
        "no_visible_output": sum(row["ttft_ms"] is None for row in complete),
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "output_tokens_per_s": output_tokens / elapsed if elapsed > 0 else None,
        "requests_per_s": len(complete) / elapsed if elapsed > 0 else None,
    }
    for field in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[field] for row in complete if row[field] is not None]
        summary[field] = {
            "mean": sum(values) / len(values) if values else None,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "p99": _percentile(values, 99),
        }
    return summary


async def _send(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    first_data: float | None = None
    last_data: float | None = None
    output_tokens: int | None = None
    error: str | None = None
    status: int | None = None
    delta_keys: set[str] = set()
    stream_events = 0
    body = {
        "model": model,
        "messages": row["messages"],
        "max_tokens": row["max_tokens"],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with client.stream("POST", url, json=body) as response:
            status = response.status_code
            if status != 200:
                error = (await response.aread()).decode(errors="replace")[:500]
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    stream_events += 1
                    usage = event.get("usage")
                    if usage is not None:
                        output_tokens = usage.get("completion_tokens")
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        delta_keys.update(delta)
                        if any(
                            delta.get(key)
                            for key in (
                                "content",
                                "reasoning",
                                "reasoning_content",
                                "tool_calls",
                            )
                        ):
                            now = time.perf_counter()
                            if first_data is None:
                                first_data = now
                            last_data = now
                if output_tokens is None:
                    error = "Stream did not include completion-token usage"
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        error = str(exc)[:500]
    end = time.perf_counter()
    e2e_ms = (end - start) * 1000
    ttft_ms = (first_data - start) * 1000 if first_data is not None else None
    tpot_ms = (
        (last_data - first_data) * 1000 / (output_tokens - 1)
        if (
            last_data is not None
            and first_data is not None
            and output_tokens is not None
            and output_tokens > 1
        )
        else None
    )
    return {
        "id": row["id"],
        "status": status,
        "error": error,
        "start_s": start,
        "end_s": end,
        "output_tokens": output_tokens or 0,
        "max_tokens": row["max_tokens"],
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "delta_keys": sorted(delta_keys),
        "stream_events": stream_events,
    }


async def _phase(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    rows: list[dict[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    results: list[dict[str, Any]] = []

    async def worker() -> None:
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await _send(client, url, model, row))
            queue.task_done()

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(rows)))))
    return results


async def replay(args: argparse.Namespace) -> dict[str, Any]:
    input_hash = _sha256(args.input)
    manifest = json.loads(args.input.with_suffix(".manifest.json").read_text())
    if input_hash != manifest["input_sha256"]:
        raise ValueError("Frozen request file does not match its manifest")
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    start = args.skip_requests
    stop = start + args.warmup_requests + args.measure_requests
    if stop > len(rows):
        raise ValueError(f"Need {stop} requests, but input has {len(rows)}")

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    timeout = httpx.Timeout(connect=30, read=600, write=600, pool=30)
    limits = httpx.Limits(
        max_connections=args.concurrency + 8,
        max_keepalive_connections=args.concurrency + 8,
    )
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, headers=headers
    ) as client:
        url = args.base_url.rstrip("/") + "/v1/chat/completions"
        warmup = await _phase(
            client,
            url,
            args.model,
            rows[start : start + args.warmup_requests],
            args.concurrency,
        )
        warmup_summary = _summary(warmup)
        if warmup_summary["failed"]:
            raise RuntimeError(f"Warmup had {warmup_summary['failed']} failed requests")

        if args.reset_prefix_cache:
            response = await client.post(
                args.base_url.rstrip("/") + "/reset_prefix_cache"
            )
            response.raise_for_status()
            if not response.json().get("success"):
                raise RuntimeError("The server could not reset its prefix cache")

        measured_rows = rows[start + args.warmup_requests : stop]
        if args.profile:
            response = await client.post(args.base_url.rstrip("/") + "/start_profile")
            response.raise_for_status()
            replay_task = asyncio.create_task(
                _phase(client, url, args.model, measured_rows, args.concurrency)
            )
            try:
                await asyncio.sleep(args.profile_seconds)
            finally:
                response = await client.post(
                    args.base_url.rstrip("/") + "/stop_profile",
                    timeout=1800,
                )
                response.raise_for_status()
            results = await replay_task
        else:
            results = await _phase(
                client,
                url,
                args.model,
                measured_rows,
                args.concurrency,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "requests.jsonl").open("w") as file:
        for row in results:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "input_sha256": input_hash,
        "model": args.model,
        "concurrency": args.concurrency,
        "skip_requests": args.skip_requests,
        "warmup_requests": args.warmup_requests,
        "measure_requests": args.measure_requests,
        "profile": args.profile,
        "profile_seconds": args.profile_seconds if args.profile else None,
        "reset_prefix_cache": args.reset_prefix_cache,
        "warmup": warmup_summary,
        "measured": _summary(results),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=200)
    parser.add_argument("--skip-requests", type=int, default=0)
    parser.add_argument("--measure-requests", type=int, default=1000)
    parser.add_argument("--api-key")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-seconds", type=float, default=10)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.skip_requests < 0
        or args.warmup_requests < 0
        or args.measure_requests < 1
        or args.profile_seconds <= 0
    ):
        parser.error("Concurrency and request counts must be positive")
    summary = asyncio.run(replay(args))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
