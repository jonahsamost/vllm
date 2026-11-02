# SPDX-License-Identifier: Apache-2.0
"""Isolate an eight-rank DeepEP HT dispatch/combine from model execution."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist


def worker(local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    import deep_ep
    from deep_ep.utils.envs import init_dist

    rank, num_ranks, group = init_dist(local_rank, world_size)
    buffer = deep_ep.Buffer(
        group,
        512 * 1024 * 1024,
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )
    try:
        torch.manual_seed(rank)
        tokens = torch.full(
            (args.num_tokens, args.hidden),
            rank + 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk = torch.randint(
            args.num_experts,
            (args.num_tokens, args.top_k),
            device="cuda",
            dtype=torch.int32,
        ).to(deep_ep.topk_idx_t)
        weights = torch.full(
            (args.num_tokens, args.top_k),
            1 / args.top_k,
            device="cuda",
            dtype=torch.float32,
        )
        layout = buffer.get_dispatch_layout(topk, args.num_experts)
        received, _, _, _, handle, _ = buffer.dispatch(
            x=tokens,
            num_tokens_per_rank=layout[0],
            num_tokens_per_rdma_rank=layout[1],
            num_tokens_per_expert=layout[2],
            is_token_in_rank=layout[3],
            topk_idx=topk,
            topk_weights=weights,
            config=deep_ep.Buffer.get_dispatch_config(num_ranks),
        )
        combined, _, _ = buffer.combine(
            received,
            handle,
            config=deep_ep.Buffer.get_combine_config(num_ranks),
        )
        torch.cuda.synchronize()
        assert combined.shape == tokens.shape
        dist.barrier(group=group)
        if rank == 0:
            print("DeepEP HT dispatch/combine passed", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.num_experts % args.num_processes:
        parser.error("Expert count must divide the number of ranks")
    torch.multiprocessing.spawn(
        worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
