# SPDX-License-Identifier: Apache-2.0
"""Measure eight-rank DeepEP HT across controlled routing patterns."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Case:
    sweep: str
    value: int
    source_counts: tuple[int, ...]
    fanout: int = 2
    hot_weight: int = 1


def make_cases(ranks: int, base_tokens: int, top_k: int) -> list[Case]:
    volumes = (32, 64, 128, 256, 512)
    skews = (1, 2, 4, 8, 16)
    cases = [Case("source_tokens", n, (n,) * ranks) for n in volumes]
    cases += [
        Case("destination_weight", weight, (base_tokens,) * ranks, hot_weight=weight)
        for weight in skews
    ]
    cases += [
        Case("fanout", n, (base_tokens,) * ranks, fanout=n)
        for n in range(1, min(ranks, top_k) + 1)
    ]
    for zero_ranks in (0, 1, 2, 4, 6, 7):
        if zero_ranks >= ranks:
            continue
        total = ranks * base_tokens
        active = ranks - zero_ranks
        counts = tuple(
            0
            if rank < zero_ranks
            else total // active + (rank - zero_ranks < total % active)
            for rank in range(ranks)
        )
        cases.append(Case("zero_source_ranks", zero_ranks, counts))
    cases += [
        Case("volume_x_weight", weight, (n,) * ranks, hot_weight=weight)
        for n in volumes
        for weight in skews
    ]
    return cases


def make_topk(
    case: Case,
    rank: int,
    index: int,
    ranks: int,
    experts: int,
    top_k: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = random.Random(seed + index * 100_003 + rank)
    experts_per_rank = experts // ranks
    rows = []
    for _ in range(case.source_counts[rank]):
        remaining = list(range(ranks))
        destinations = []
        for _ in range(case.fanout):
            weights = [case.hot_weight if d == 0 else 1 for d in remaining]
            dest = rng.choices(remaining, weights=weights, k=1)[0]
            remaining.remove(dest)
            destinations.append(dest)
        ids = []
        for slot, dest in enumerate(destinations):
            count = top_k // case.fanout + (slot < top_k % case.fanout)
            ids.extend(
                dest * experts_per_rank + local_id
                for local_id in rng.sample(range(experts_per_rank), count)
            )
        rng.shuffle(ids)
        rows.append(ids)
    if not rows:
        return torch.empty((0, top_k), dtype=torch.int64, device=device)
    return torch.tensor(rows, dtype=torch.int64, device=device)


def run_once(
    buffer,
    tokens,
    topk,
    weights,
    experts,
    dispatch_config,
    combine_config,
    fanout: int,
    verify: bool = False,
) -> tuple[float, float, float, list[int]]:
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
    if combined.shape != tokens.shape:
        raise ValueError(f"Unexpected combined shape: {combined.shape}")
    # Without expert reduction, each destination returns one copy of its token.
    if verify and not torch.allclose(combined, tokens * fanout, atol=0.05, rtol=0.01):
        raise ValueError("Dispatch/combine returned unexpected token values")
    ms = 1e-6
    return (
        (after_layout - start) * ms,
        (after_dispatch - after_layout) * ms,
        (after_combine - after_dispatch) * ms,
        layout[0].cpu().tolist(),
    )


def worker(local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    import deep_ep
    from deep_ep.utils.envs import init_dist

    rank, num_ranks, group = init_dist(local_rank, world_size)
    device = torch.device("cuda", local_rank)
    buffer = deep_ep.Buffer(
        group,
        512 * 1024 * 1024,
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )
    try:
        cases = make_cases(num_ranks, args.base_tokens, args.top_k)
        if args.quick:
            first_cases = []
            for sweep in (
                "source_tokens",
                "destination_weight",
                "fanout",
                "zero_source_ranks",
            ):
                first_cases.extend([c for c in cases if c.sweep == sweep][:2])
            cases = first_cases
        dispatch_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
        combine_config = deep_ep.Buffer.get_combine_config(num_ranks)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        **{
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                        },
                        "created_utc": datetime.now(UTC).isoformat(),
                        "gpu": torch.cuda.get_device_name(device),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "timing": "synchronized wall time, slowest rank per repeat",
                    },
                    indent=2,
                )
                + "\n"
            )
            output_file = args.output.open("w", newline="")
            writer = None
        for index, case in enumerate(cases):
            topk = make_topk(
                case,
                rank,
                index,
                num_ranks,
                args.num_experts,
                args.top_k,
                args.seed,
                device,
            ).to(deep_ep.topk_idx_t)
            tokens = torch.full(
                (case.source_counts[rank], args.hidden),
                rank + 1,
                dtype=torch.bfloat16,
                device=device,
            )
            weights = torch.full(
                topk.shape,
                1 / args.top_k,
                dtype=torch.float32,
                device=device,
            )
            local_results = []
            for iteration in range(args.warmups + args.repeats):
                dist.barrier(group=group)
                result = run_once(
                    buffer,
                    tokens,
                    topk,
                    weights,
                    args.num_experts,
                    dispatch_config,
                    combine_config,
                    case.fanout,
                    verify=iteration == 0,
                )
                if iteration >= args.warmups:
                    local_results.append(result[:3])
            gathered = [None] * num_ranks
            dist.all_gather_object(
                gathered,
                {"times": local_results, "dest_counts": result[3]},
                group=group,
            )
            if rank == 0:
                dest_counts = [
                    sum(item["dest_counts"][dest] for item in gathered)
                    for dest in range(num_ranks)
                ]
                total_tokens = sum(case.source_counts)
                total_dispatches = sum(dest_counts)
                for repetition in range(args.repeats):
                    stage_max = [
                        max(item["times"][repetition][stage] for item in gathered)
                        for stage in range(3)
                    ]
                    total_max = max(sum(item["times"][repetition]) for item in gathered)
                    row = {
                        "sweep": case.sweep,
                        "value": case.value,
                        "repetition": repetition,
                        "source_tokens_per_rank": case.source_counts[0],
                        "total_source_tokens": total_tokens,
                        "zero_source_ranks": case.source_counts.count(0),
                        "fanout_target": case.fanout,
                        "hot_weight": case.hot_weight,
                        "actual_fanout": total_dispatches / total_tokens,
                        "dest_max_over_mean": max(dest_counts)
                        * num_ranks
                        / total_dispatches,
                        "dest_counts": ";".join(map(str, dest_counts)),
                        "layout_ms": stage_max[0],
                        "dispatch_ms": stage_max[1],
                        "combine_ms": stage_max[2],
                        "total_ms": total_max,
                    }
                    if writer is None:
                        writer = csv.DictWriter(output_file, fieldnames=row)
                        writer.writeheader()
                    writer.writerow(row)
                output_file.flush()
                print(
                    f"{case.sweep}={case.value}: "
                    f"{total_tokens} tokens, "
                    f"fanout {total_dispatches / total_tokens:.2f}, "
                    f"destination skew "
                    f"{max(dest_counts) * num_ranks / total_dispatches:.2f}",
                    flush=True,
                )
        if rank == 0:
            output_file.close()
    finally:
        buffer.destroy()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--base-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.num_experts % args.num_processes:
        parser.error("Expert count must divide rank count")
    if args.top_k < 2 or args.top_k > args.num_processes:
        parser.error("Top-k must be between 2 and rank count")
    if args.top_k > args.num_experts // args.num_processes:
        parser.error("Top-k exceeds experts per rank")
    if args.base_tokens <= 0 or args.repeats <= 0 or args.warmups < 0:
        parser.error("Base tokens and repeats must be positive; warmups nonnegative")
    torch.multiprocessing.spawn(
        worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
