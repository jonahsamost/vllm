# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark EPLB mapping with pair recording disabled and enabled.

Run with: .venv/bin/python benchmarks/kernels/benchmark_corr_aware_eplb.py
"""

import argparse

import torch
from triton.testing import do_bench

from vllm.model_executor.layers.fused_moe.router.base_router import (
    _eplb_map_and_record_triton,
)
from vllm.model_executor.layers.fused_moe.router.corr_aware_eplb_kernel import (
    eplb_corr_map_and_record_triton,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20, help="Warmup time in ms")
    parser.add_argument("--rep", type=int, default=100, help="Timing time in ms")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.num_tokens < 1 or not 1 <= args.top_k <= args.num_experts:
        parser.error("Require num_tokens >= 1 and 1 <= top_k <= num_experts")

    device = torch.device("cuda")
    token = torch.arange(args.num_tokens, dtype=torch.int32, device=device)[:, None]
    slot = torch.arange(args.top_k, dtype=torch.int32, device=device)[None, :]
    topk_ids = (7 * token + slot) % args.num_experts
    replica_counts = torch.full(
        (args.num_experts,), 2, dtype=torch.int32, device=device
    )
    physical_map = torch.arange(
        args.num_experts * 2, dtype=torch.int32, device=device
    ).reshape(args.num_experts, 2)
    baseline_load = torch.zeros(args.num_experts * 2, dtype=torch.int32, device=device)
    corr_load = torch.zeros_like(baseline_load)
    pair_counts = torch.zeros(
        (args.num_experts, args.num_experts), dtype=torch.int32, device=device
    )
    load_flag = torch.tensor(True, dtype=torch.bool, device=device)
    pair_flag = torch.tensor(False, dtype=torch.bool, device=device)
    num_unpadded = torch.tensor(args.num_tokens, dtype=torch.int32, device=device)

    def baseline() -> None:
        _eplb_map_and_record_triton(
            topk_ids, physical_map, replica_counts, baseline_load,
            load_flag, num_unpadded,
        )

    def corr_aware() -> None:
        eplb_corr_map_and_record_triton(
            topk_ids, physical_map, replica_counts, corr_load,
            pair_counts, load_flag, pair_flag, num_unpadded,
        )

    baseline_ms = do_bench(baseline, warmup=args.warmup, rep=args.rep)
    pair_flag.fill_(False)
    disabled_ms = do_bench(corr_aware, warmup=args.warmup, rep=args.rep)
    pair_flag.fill_(True)
    enabled_ms = do_bench(corr_aware, warmup=args.warmup, rep=args.rep)

    print(
        f"tokens={args.num_tokens}, experts={args.num_experts}, top_k={args.top_k}"
    )
    print(f"Existing EPLB mapping/load: {baseline_ms * 1000:.1f} us")
    print(f"Corr-aware, pairs disabled: {disabled_ms * 1000:.1f} us")
    print(f"Corr-aware, pairs enabled:  {enabled_ms * 1000:.1f} us")
    print("Timings include the Python wrapper and output allocation.")


if __name__ == "__main__":
    main()
