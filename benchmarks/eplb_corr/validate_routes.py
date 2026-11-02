# SPDX-License-Identifier: Apache-2.0
"""Validate a worker-local routed-expert capture before MoE replay."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np


def validate(root: Path, expected_ranks: int) -> dict:
    ranks = sorted(root.glob("rank-*/config.json"))
    if len(ranks) != expected_ranks:
        raise ValueError(f"Expected {expected_ranks} ranks, found {len(ranks)}")

    summaries = []
    reference = None
    for config_path in ranks:
        config = json.loads(config_path.read_text())
        shape = (config["layer_ids"], config["top_k"])
        if reference is None:
            reference = shape
        elif shape != reference:
            raise ValueError(f"Rank {config['world_rank']} has different MoE layers")
        dtype = np.dtype(config["dtype"])
        num_layers = len(config["layer_ids"])
        n_experts = config["num_logical_experts"]
        steps = 0
        forwards = 0
        tokens = 0
        zero_steps = 0
        routes_path = config_path.parent / "routes.bin"
        with (
            (config_path.parent / "steps.jsonl").open() as metadata_file,
            routes_path.open("rb") as routes_file,
        ):
            for line in metadata_file:
                record = json.loads(line)
                if record["step"] != steps:
                    raise ValueError(
                        f"Nonsequential step on rank {config['world_rank']}"
                    )
                if record["offset"] != routes_file.tell():
                    raise ValueError(f"Bad offset on rank {config['world_rank']}")
                length_bytes = routes_file.read(4)
                if len(length_bytes) != 4:
                    raise ValueError("Truncated trace length")
                length = struct.unpack("<I", length_bytes)[0]
                if length != record["compressed_bytes"]:
                    raise ValueError("Compressed length mismatch")
                compressed = routes_file.read(length)
                if len(compressed) != length:
                    raise ValueError("Truncated trace record")
                raw = zlib.decompress(compressed)
                n_tokens = record["num_tokens"]
                expected_bytes = (
                    n_tokens * num_layers * config["top_k"] * dtype.itemsize
                )
                if len(raw) != expected_bytes or record["raw_bytes"] != expected_bytes:
                    raise ValueError("Invalid route tensor size")
                if n_tokens:
                    ids = np.frombuffer(raw, dtype=dtype).reshape(
                        n_tokens, num_layers, config["top_k"]
                    )
                    if np.any(ids >= n_experts):
                        raise ValueError("Logical expert ID out of range")
                    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
                        raise ValueError("Duplicate expert in a token's top-k")
                    if record["query_start_loc"][-1] != n_tokens:
                        raise ValueError("Request row boundaries do not match tokens")
                else:
                    zero_steps += 1
                forwards += bool(record["has_forward"])
                tokens += n_tokens
                steps += 1
            if routes_file.read(1):
                raise ValueError(f"Extra trace bytes on rank {config['world_rank']}")
        summaries.append(
            {
                "rank": config["world_rank"],
                "steps": steps,
                "forwards": forwards,
                "tokens": tokens,
                "zero_steps": zero_steps,
                "bytes": routes_path.stat().st_size,
            }
        )

    if len({row["forwards"] for row in summaries}) != 1:
        raise ValueError(f"Forward counts differ across ranks: {summaries}")
    return {"ranks": summaries, "layer_ids": reference[0], "top_k": reference[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--ranks", type=int, default=8)
    args = parser.parse_args()
    summary = validate(args.root, args.ranks)
    (args.root / "validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
