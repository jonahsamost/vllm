# SPDX-License-Identifier: Apache-2.0
"""Prepare captured high-volume routes and weighted EPLB placements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from prepare_captured_replay import (
    LAYERS,
    TRAIN_END,
    TRAIN_START,
    _load_metadata,
    _read_selected,
)

from vllm.distributed.eplb.policy.corr_aware import CorrAwareEplbPolicy
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy


def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((args.prepared / "manifest.json").read_text())
    metadata = _load_metadata(args.capture, source_manifest["num_ranks"])
    source_counts = np.array(
        [[record["num_tokens"] for record in rank] for rank in metadata]
    )
    totals = source_counts.sum(axis=0)
    steps = [
        int(step)
        for step in range(TRAIN_END, len(totals))
        if args.min_tokens <= totals[step] <= args.max_tokens
    ]
    if not steps:
        raise ValueError("No captured forwards in the requested token range")

    for rank in range(source_manifest["num_ranks"]):
        ids, offsets = _read_selected(
            args.capture,
            rank,
            metadata[rank],
            steps,
            LAYERS,
            source_manifest["top_k"],
        )
        np.savez(args.output / f"rank-{rank:02d}.npz", ids=ids, offsets=offsets)

    stats = np.load(args.prepared / "training_stats.npz")
    loads = torch.from_numpy(stats["loads"])
    pairs = torch.from_numpy(stats["pairs"])
    np.savez(
        args.output / "training_stats.npz", loads=loads.numpy(), pairs=pairs.numpy()
    )
    experts = source_manifest["num_logical_experts"]
    physical = source_manifest["num_physical_experts"]
    ranks = source_manifest["num_ranks"]
    placements = {
        "baseline": np.tile(np.arange(experts), (len(LAYERS), 1)),
        "default": DefaultEplbPolicy.rebalance_experts(
            loads, physical, 1, 1, ranks
        ).numpy(),
        "raw_corr": CorrAwareEplbPolicy.rebalance_experts(
            loads, physical, 1, 1, ranks, expert_pair_counts=pairs
        ).numpy(),
    }
    weights = {}
    for tenth in range(11):
        name = f"w{tenth:02d}"
        weights[name] = tenth / 10
        placements[name] = CorrAwareEplbPolicy.rebalance_experts(
            loads,
            physical,
            1,
            1,
            ranks,
            expert_pair_counts=pairs,
            correlation_weight=weights[name],
        ).numpy()

    for name, placement in placements.items():
        expected = experts if name == "baseline" else physical
        if placement.shape != (len(LAYERS), expected):
            raise ValueError(f"Unexpected placement shape for {name}")
        if any(np.any(np.bincount(row, minlength=experts) == 0) for row in placement):
            raise ValueError(f"Placement omits an expert: {name}")
        np.save(args.output / f"{name}.npy", placement.astype(np.int16))

    strata = {
        "8k_16k": [step for step in steps if totals[step] < 16384],
        "16k_32k": [step for step in steps if totals[step] >= 16384],
    }
    manifest = {
        "capture": str(args.capture),
        "training_steps": [TRAIN_START, TRAIN_END],
        "evaluation_steps": steps,
        "strata": strata,
        "layers": list(LAYERS),
        "variants": list(placements),
        "correlation_weights": weights,
        "normalization": "per-placement-decision candidate min-max spans",
        "num_ranks": ranks,
        "num_logical_experts": experts,
        "num_physical_experts": physical,
        "top_k": source_manifest["top_k"],
        "hidden_size": source_manifest["hidden_size"],
        "redundant_experts": source_manifest["redundant_experts"],
        "min_source_tokens": args.min_tokens,
        "max_source_tokens": args.max_tokens,
        "total_source_tokens_min": int(totals[steps].min()),
        "total_source_tokens_max": int(totals[steps].max()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32768)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
