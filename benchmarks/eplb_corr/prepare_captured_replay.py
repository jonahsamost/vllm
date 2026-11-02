# SPDX-License-Identifier: Apache-2.0
"""Build disjoint EPLB training and replay sets from a captured DEP8 trace."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

from vllm.distributed.eplb.policy.corr_aware import CorrAwareEplbPolicy
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

LAYERS = (0, 5, 10, 15, 20, 25, 30, 35)
TRAIN_START = 2000
TRAIN_END = 3000
SAMPLES_PER_STRATUM = 32


def _load_metadata(root: Path, ranks: int) -> list[list[dict]]:
    metadata = []
    for rank in range(ranks):
        path = root / f"rank-{rank:02d}" / "steps.jsonl"
        with path.open() as file:
            metadata.append([json.loads(line) for line in file])
    if len({len(records) for records in metadata}) != 1:
        raise ValueError("Capture ranks have different forward counts")
    return metadata


def _select_steps(metadata: list[list[dict]]) -> tuple[list[int], dict, dict]:
    counts = np.array(
        [[entry["num_tokens"] for entry in records] for records in metadata],
        dtype=np.int32,
    )
    total = counts.sum(axis=0)
    zero = (counts == 0).sum(axis=0)
    eligible = np.arange(TRAIN_END, len(total))
    strata = {
        "decode": eligible[(total[eligible] <= 64) & (zero[eligible] == 0)],
        "medium": eligible[
            (total[eligible] > 64) & (total[eligible] <= 4096) & (zero[eligible] == 0)
        ],
        "high": eligible[(total[eligible] > 4096) & (zero[eligible] == 0)],
        "zero_1_to_3": eligible[
            (zero[eligible] >= 1) & (zero[eligible] <= 3) & (total[eligible] > 0)
        ],
        "zero_4_to_7": eligible[
            (zero[eligible] >= 4) & (zero[eligible] <= 7) & (total[eligible] > 0)
        ],
    }
    selected = {}
    for name, candidates in strata.items():
        if len(candidates) < SAMPLES_PER_STRATUM:
            raise ValueError(f"Insufficient {name} steps: {len(candidates)}")
        indices = np.linspace(
            0, len(candidates) - 1, SAMPLES_PER_STRATUM, dtype=np.int32
        )
        selected[name] = candidates[indices].tolist()
    steps = sorted(step for values in selected.values() for step in values)
    if len(set(steps)) != len(steps):
        raise ValueError("A replay step was selected more than once")
    return (
        steps,
        selected,
        {name: len(candidates) for name, candidates in strata.items()},
    )


def _read_selected(
    root: Path,
    rank: int,
    metadata: list[dict],
    steps: list[int],
    layers: tuple[int, ...],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    offsets = [0]
    path = root / f"rank-{rank:02d}" / "routes.bin"
    with path.open("rb") as file:
        for step in steps:
            record = metadata[step]
            file.seek(record["offset"])
            length = struct.unpack("<I", file.read(4))[0]
            if length != record["compressed_bytes"]:
                raise ValueError(f"Frame length mismatch at rank {rank} step {step}")
            raw = zlib.decompress(file.read(length))
            n = record["num_tokens"]
            ids = np.frombuffer(raw, dtype=np.uint16).reshape(n, 40, top_k)
            arrays.append(ids[:, layers, :].copy())
            offsets.append(offsets[-1] + n)
    return np.concatenate(arrays), np.array(offsets, dtype=np.int32)


def _statistics(ids: np.ndarray, experts: int) -> tuple[np.ndarray, np.ndarray]:
    num_layers = ids.shape[1]
    top_k = ids.shape[2]
    loads = np.empty((num_layers, experts), dtype=np.int64)
    pairs = np.zeros((num_layers, experts, experts), dtype=np.int64)
    for layer in range(num_layers):
        chosen = ids[:, layer, :].astype(np.int32)
        loads[layer] = np.bincount(chosen.ravel(), minlength=experts)
        for first in range(1, top_k):
            for second in range(first):
                a = chosen[:, first]
                b = chosen[:, second]
                flattened = np.maximum(a, b) * experts + np.minimum(a, b)
                pairs[layer] += np.bincount(
                    flattened, minlength=experts * experts
                ).reshape(experts, experts)
    return loads, pairs


def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.capture / "rank-00" / "config.json").read_text())
    top_k = config["top_k"]
    experts = config["num_logical_experts"]
    if config["dtype"] != "uint16" or config["layer_ids"] != list(range(40)):
        raise ValueError("Expected the 40-layer DSV4.1 uint16 route capture")
    metadata = _load_metadata(args.capture, args.ranks)
    steps, strata, strata_population = _select_steps(metadata)
    train_arrays = []
    training_steps = list(range(TRAIN_START, TRAIN_END))
    for rank in range(args.ranks):
        train, _ = _read_selected(
            args.capture, rank, metadata[rank], training_steps, LAYERS, top_k
        )
        train_arrays.append(train)
        evaluate, offsets = _read_selected(
            args.capture, rank, metadata[rank], steps, LAYERS, top_k
        )
        np.savez(args.output / f"rank-{rank:02d}.npz", ids=evaluate, offsets=offsets)
    loads, pairs = _statistics(np.concatenate(train_arrays), experts)
    del train_arrays
    physical = experts + args.redundant_experts
    if physical % args.ranks:
        raise ValueError("Physical experts must divide evenly across ranks")
    default = DefaultEplbPolicy.rebalance_experts(
        torch.from_numpy(loads), physical, 1, 1, args.ranks
    ).numpy()
    corr = CorrAwareEplbPolicy.rebalance_experts(
        torch.from_numpy(loads),
        physical,
        1,
        1,
        args.ranks,
        expert_pair_counts=torch.from_numpy(pairs),
    ).numpy()
    placements = {
        "baseline": np.tile(np.arange(experts), (len(LAYERS), 1)),
        "default": default,
        "corr_aware": corr,
    }
    for name, placement in placements.items():
        if placement.shape != (
            len(LAYERS),
            experts if name == "baseline" else physical,
        ):
            raise ValueError(f"Unexpected {name} placement shape")
        if np.any(placement < 0) or np.any(placement >= experts):
            raise ValueError(f"Invalid logical ID in {name} placement")
        if any(np.any(np.bincount(row, minlength=experts) == 0) for row in placement):
            raise ValueError(f"{name} placement omitted a logical expert")
        np.save(args.output / f"{name}.npy", placement.astype(np.int16))
    np.savez(args.output / "training_stats.npz", loads=loads, pairs=pairs)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "capture": str(args.capture),
                "training_steps": [TRAIN_START, TRAIN_END],
                "evaluation_steps": steps,
                "strata": strata,
                "strata_population": strata_population,
                "layers": LAYERS,
                "num_ranks": args.ranks,
                "num_logical_experts": experts,
                "num_physical_experts": physical,
                "top_k": top_k,
                "hidden_size": args.hidden,
                "redundant_experts": args.redundant_experts,
                "training_token_rows": int(loads.sum() // top_k // len(LAYERS)),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--redundant-experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=5120)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
