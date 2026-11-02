# SPDX-License-Identifier: Apache-2.0
"""Replay captured logical MoE routes through candidate DeepEP placements."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

VARIANTS = ("baseline", "default", "corr_aware")
KNUTH_MULTIPLIER = np.uint64(2654435769)


def physical_map(
    placement: np.ndarray, logical_experts: int
) -> tuple[np.ndarray, np.ndarray]:
    replicas = [[] for _ in range(logical_experts)]
    for physical, logical in enumerate(placement):
        replicas[int(logical)].append(physical)
    counts = np.array([len(copies) for copies in replicas], dtype=np.int32)
    if np.any(counts == 0):
        raise ValueError("Placement omitted a logical expert")
    mapping = np.full((logical_experts, max(counts)), -1, dtype=np.int32)
    for logical, copies in enumerate(replicas):
        mapping[logical, : len(copies)] = copies
    return mapping, counts


def map_ids(ids: np.ndarray, mapping: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n = len(ids)
    if n == 0:
        return np.empty(ids.shape, dtype=np.int32)
    token_hash = (np.arange(n, dtype=np.uint64) * KNUTH_MULTIPLIER) & np.uint64(
        0xFFFFFFFF
    )
    replica = (token_hash[:, None] % counts[ids]).astype(np.int32)
    return mapping[ids, replica]


def expected_fanout(physical_ids: np.ndarray, slots_per_rank: int) -> np.ndarray:
    if len(physical_ids) == 0:
        return np.empty((0,), dtype=np.int32)
    destinations = np.sort(physical_ids // slots_per_rank, axis=1)
    return (1 + np.count_nonzero(np.diff(destinations, axis=1), axis=1)).astype(
        np.int32
    )


def run_once(buffer, tokens, topk, weights, experts, dispatch_config, combine_config):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    layout = buffer.get_dispatch_layout(topk, experts)
    torch.cuda.synchronize()
    after_layout = time.perf_counter_ns()
    received, _, _, _, handle, _ = buffer.dispatch(
        x=tokens,
        num_tokens_per_rank=layout[0],
        num_tokens_per_rdma_rank=layout[1],
        num_tokens_per_expert=layout[2],
        is_token_in_rank=layout[3],
        topk_idx=topk,
        topk_weights=weights,
        config=dispatch_config,
    )
    torch.cuda.synchronize()
    after_dispatch = time.perf_counter_ns()
    combined, _, _ = buffer.combine(received, handle, config=combine_config)
    torch.cuda.synchronize()
    after_combine = time.perf_counter_ns()
    ms = 1e-6
    return (
        (after_layout - start) * ms,
        (after_dispatch - after_layout) * ms,
        (after_combine - after_dispatch) * ms,
        layout[0],
        combined,
    )


def worker(local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    import deep_ep
    from deep_ep.utils.envs import init_dist

    rank, ranks, group = init_dist(local_rank, world_size)
    manifest = json.loads((args.input / "manifest.json").read_text())
    if ranks != manifest["num_ranks"]:
        raise ValueError("Replay rank count differs from the capture")
    variants = tuple(manifest.get("variants", VARIANTS))
    layers = manifest["layers"]
    steps = manifest["evaluation_steps"]
    if args.quick:
        steps = sorted(values[0] for values in manifest["strata"].values() if values)
        layers = layers[:1]
    ids_file = np.load(args.input / f"rank-{rank:02d}.npz")
    all_ids = ids_file["ids"]
    offsets = ids_file["offsets"]
    step_indices = {
        step: index for index, step in enumerate(manifest["evaluation_steps"])
    }
    placements = {name: np.load(args.input / f"{name}.npy") for name in variants}
    maps = {
        name: [physical_map(row, manifest["num_logical_experts"]) for row in placement]
        for name, placement in placements.items()
    }
    buffer = deep_ep.Buffer(
        group,
        int(args.buffer_gib * 1024**3),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )
    dispatch_config = deep_ep.Buffer.get_dispatch_config(ranks)
    combine_config = deep_ep.Buffer.get_combine_config(ranks)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "input": str(args.input),
                    "created_utc": datetime.now(UTC).isoformat(),
                    "gpu": torch.cuda.get_device_name(local_rank),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "steps": steps,
                    "layers": layers,
                    "variants": variants,
                    "timing": "synchronized wall time per rank",
                    "buffer_gib": args.buffer_gib,
                },
                indent=2,
            )
            + "\n"
        )
    fields = (
        "step",
        "layer",
        "variant",
        "repetition",
        "source_tokens",
        "dest_counts",
        "layout_ms",
        "dispatch_ms",
        "combine_ms",
        "total_ms",
    )
    try:
        with (args.output / f"rank-{rank:02d}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            case_index = 0
            for step in steps:
                index = step_indices[step]
                start, end = int(offsets[index]), int(offsets[index + 1])
                tokens = torch.full(
                    (end - start, manifest["hidden_size"]),
                    rank + 1,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                weights = torch.full(
                    (end - start, manifest["top_k"]),
                    1 / manifest["top_k"],
                    dtype=torch.float32,
                    device="cuda",
                )
                for layer in layers:
                    layer_pos = manifest["layers"].index(layer)
                    logical = all_ids[start:end, layer_pos, :].astype(np.int32)
                    order = (
                        variants[case_index % len(variants) :]
                        + variants[: case_index % len(variants)]
                    )
                    for variant in order:
                        placement = placements[variant][layer_pos]
                        mapping, counts = maps[variant][layer_pos]
                        physical = map_ids(logical, mapping, counts)
                        topk = torch.from_numpy(physical).to(
                            device="cuda", dtype=deep_ep.topk_idx_t
                        )
                        fanout = expected_fanout(physical, len(placement) // ranks)
                        measured = []
                        for repetition in range(args.warmups + args.repeats):
                            dist.barrier(group=group)
                            result = run_once(
                                buffer,
                                tokens,
                                topk,
                                weights,
                                len(placement),
                                dispatch_config,
                                combine_config,
                            )
                            if repetition == 0:
                                expected = (
                                    tokens
                                    * torch.as_tensor(fanout, device="cuda")[:, None]
                                )
                                if result[4].shape != expected.shape or not (
                                    torch.allclose(
                                        result[4], expected, atol=0.05, rtol=0.01
                                    )
                                ):
                                    raise ValueError(
                                        f"Combine mismatch: {step=}, {layer=}, "
                                        f"{rank=}, {variant=}"
                                    )
                            if repetition >= args.warmups:
                                measured.append(result[:3])
                        dest_counts = ";".join(map(str, result[3].cpu().tolist()))
                        for repetition, times in enumerate(measured):
                            writer.writerow(
                                {
                                    "step": step,
                                    "layer": layer,
                                    "variant": variant,
                                    "repetition": repetition,
                                    "source_tokens": end - start,
                                    "dest_counts": dest_counts,
                                    "layout_ms": times[0],
                                    "dispatch_ms": times[1],
                                    "combine_ms": times[2],
                                    "total_ms": sum(times),
                                }
                            )
                    case_index += 1
                file.flush()
                if rank == 0:
                    print(f"completed captured step {step}", flush=True)
        dist.barrier(group=group)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--buffer-gib", type=float, default=0.5)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1 or args.buffer_gib <= 0:
        parser.error("Warmups, repeats, and buffer size must be positive")
    torch.multiprocessing.spawn(
        worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
