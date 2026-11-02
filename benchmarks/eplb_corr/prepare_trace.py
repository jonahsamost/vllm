# SPDX-License-Identifier: Apache-2.0
"""Freeze completed session-trace turns as identical benchmark requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def _iter_shard(path: Path) -> Iterator[dict[str, Any]]:
    process = subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            yield json.loads(line)
    finally:
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        if process.wait() != 0:
            raise RuntimeError(f"Failed to read {path}: {stderr}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_only(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            return False
        if any(
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            for part in content
        ):
            return False
    return True


def _response_text(response: dict[str, Any]) -> str:
    parts = [response.get("reasoning_content") or "", response.get("content") or ""]
    for call in response.get("tool_calls") or []:
        function = call.get("function") or {}
        parts.extend((function.get("name") or "", function.get("arguments") or ""))
    return "".join(parts)


def prepare(
    raw_dir: Path,
    output: Path,
    tokenizer_name: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_name, tokenizer_mode="deepseek_v41")
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    source_manifests: list[dict[str, str]] = []

    for manifest_path in sorted(raw_dir.glob("run=*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        source_manifests.append(
            {
                "path": str(manifest_path.relative_to(raw_dir)),
                "sha256": _sha256(manifest_path),
            }
        )
        for shard in manifest["shards"]:
            shard_path = manifest_path.parent / shard["key"]
            if _sha256(shard_path) != shard["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {shard_path}")
            count = 0
            for session in _iter_shard(shard_path):
                count += 1
                for turn in session["turns"]:
                    if (
                        turn["result"]["status"] != 200
                        or not turn["result"]["complete"]
                    ):
                        reasons["incomplete_or_failed"] += 1
                        continue
                    response = turn.get("assistant_response")
                    if response is None:
                        reasons["missing_response"] += 1
                        continue
                    messages = turn["messages"]
                    if not _text_only(messages):
                        reasons["multimodal_or_unsupported_content"] += 1
                        continue
                    response_text = _response_text(response)
                    if not response_text:
                        reasons["empty_response"] += 1
                        continue
                    output_tokens = len(
                        tokenizer.encode(response_text, add_special_tokens=False)
                    )
                    if output_tokens < 2:
                        reasons["short_response"] += 1
                        continue
                    if output_tokens > max_output_tokens:
                        reasons["output_capped"] += 1
                        output_tokens = max_output_tokens
                    rows.append(
                        {
                            "id": turn["capture_id"],
                            "received_unix_ms": turn["received_unix_ms"],
                            "session_id": session["session_id"],
                            "turn": turn["turn"],
                            "messages": messages,
                            "max_tokens": output_tokens,
                        }
                    )
            if count != shard["rows"]:
                raise ValueError(f"Shard row count mismatch: {shard_path}")

    rows.sort(
        key=lambda row: (
            row["received_unix_ms"],
            row["session_id"],
            row["turn"],
            row["id"],
        )
    )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate capture IDs in prepared requests")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "eplb-corr-benchmark-input-v1",
        "requests": len(rows),
        "excluded": dict(sorted(reasons.items())),
        "input_sha256": _sha256(output),
        "source_manifests": source_manifests,
        "tokenizer": tokenizer_name,
        "max_output_tokens": max_output_tokens,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.max_output_tokens < 2:
        parser.error("--max-output-tokens must be at least 2")
    summary = prepare(
        args.raw_dir, args.output, args.tokenizer, args.max_output_tokens
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "source_manifests"}))


if __name__ == "__main__":
    main()
