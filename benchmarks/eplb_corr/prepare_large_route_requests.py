# SPDX-License-Identifier: Apache-2.0
"""Select fixed production prompts for a large-token DEP8 route capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt_chars(row: dict) -> int:
    return sum(len(str(message.get("content") or "")) for message in row["messages"])


def prepare(source: Path, output: Path, waves_per_band: int) -> None:
    if waves_per_band < 1:
        raise ValueError("waves_per_band must be positive")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    wave_size = 8
    needed = wave_size * waves_per_band
    # Character length only selects candidate prompts. The captured metadata
    # records actual routed-token counts for every forward.
    bands = (
        ("8k", 3_000, 6_000, 4_000),
        ("16k", 6_000, 12_000, 8_000),
        ("32k", 12_000, 24_000, 16_000),
        ("64k", 24_000, 48_000, 32_000),
        ("128k", 90_000, None, 100_000),
    )
    selected = []
    manifest_bands = []
    for name, minimum, maximum, target in bands:
        candidates = [
            (_prompt_chars(row), row)
            for row in rows
            if minimum <= _prompt_chars(row)
            and (maximum is None or _prompt_chars(row) < maximum)
        ]
        if len(candidates) < needed:
            raise ValueError(
                f"{name} has only {len(candidates)} prompts; need {needed}"
            )
        candidates.sort(key=lambda item: (abs(item[0] - target), item[1]["id"]))
        chosen = candidates[:needed]
        for char_count, row in chosen:
            selected.append(
                {
                    "id": row["id"],
                    "messages": row["messages"],
                    "max_tokens": 2,
                    "capture_band": name,
                    "prompt_chars": char_count,
                }
            )
        manifest_bands.append(
            {
                "name": name,
                "requests": needed,
                "prompt_chars_min": min(chars for chars, _ in chosen),
                "prompt_chars_max": max(chars for chars, _ in chosen),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema": "eplb-corr-large-route-input-v1",
                "source": str(source),
                "source_sha256": _sha256(source),
                "input_sha256": _sha256(output),
                "requests": len(selected),
                "waves_per_band": waves_per_band,
                "wave_size": wave_size,
                "bands": manifest_bands,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waves-per-band", type=int, default=4)
    args = parser.parse_args()
    prepare(args.source, args.output, args.waves_per_band)


if __name__ == "__main__":
    main()
