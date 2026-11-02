# SPDX-License-Identifier: Apache-2.0
"""Turn a large-forward DEP8 capture into a three-placement DeepEP replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_captured_replay import LAYERS, _load_metadata, _read_selected


def prepare(capture: Path, placements: Path, output: Path) -> None:
    config = json.loads((capture / "rank-00" / "config.json").read_text())
    if config["dtype"] != "uint16" or config["layer_ids"] != list(range(40)):
        raise ValueError("Expected a 40-layer DSV4.1 uint16 route capture")
    ranks = 8
    metadata = _load_metadata(capture, ranks)
    counts = np.array(
        [[entry["num_tokens"] for entry in rank] for rank in metadata],
        dtype=np.int32,
    )
    totals = counts.sum(axis=0)
    steps = [int(step) for step in np.flatnonzero(totals >= 8192)]
    if not steps:
        raise ValueError("No forwards with at least 8192 total source tokens")
    output.mkdir(parents=True, exist_ok=True)
    for rank in range(ranks):
        ids, offsets = _read_selected(
            capture, rank, metadata[rank], steps, LAYERS, config["top_k"]
        )
        np.savez(output / f"rank-{rank:02d}.npz", ids=ids, offsets=offsets)

    variants = ("baseline", "default", "w03")
    for name in variants:
        placement = np.load(placements / f"{name}.npy")
        if placement.shape[0] != len(LAYERS):
            raise ValueError(f"Unexpected layer count for {name}")
        np.save(output / f"{name}.npy", placement)
    strata = {
        f"{low // 1024}k_{high // 1024}k": [
            step for step in steps if low <= totals[step] < high
        ]
        for low, high in (
            (8192, 16384),
            (16384, 32768),
            (32768, 65536),
            (65536, 131073),
        )
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "capture": str(capture),
                "placement_source": str(placements),
                "evaluation_steps": steps,
                "strata": strata,
                "step_source_counts": {
                    str(step): counts[:, step].tolist() for step in steps
                },
                "layers": list(LAYERS),
                "variants": list(variants),
                "num_ranks": ranks,
                "num_logical_experts": config["num_logical_experts"],
                "num_physical_experts": 392,
                "top_k": config["top_k"],
                "hidden_size": 5120,
                "total_source_tokens_min": int(totals[steps].min()),
                "total_source_tokens_max": int(totals[steps].max()),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.capture, args.placements, args.output)


if __name__ == "__main__":
    main()
