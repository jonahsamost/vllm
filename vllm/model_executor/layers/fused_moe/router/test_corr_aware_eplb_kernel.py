# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import combinations

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "Correlation-aware EPLB kernel requires CUDA", allow_module_level=True
    )

from vllm.model_executor.layers.fused_moe.router.base_router import (  # noqa: E402
    _eplb_map_and_record_triton,
)
from vllm.model_executor.layers.fused_moe.router.corr_aware_eplb_kernel import (  # noqa: E402
    eplb_corr_map_and_record_triton,
)


@pytest.mark.parametrize("top_k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize(
    "record_load,record_pairs",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_mapping_and_counts_match_references(
    top_k: int, record_load: bool, record_pairs: bool
) -> None:
    """Mapping and physical load match EPLB; pairs count real tokens only."""
    num_tokens = 263  # Cross the kernel's 256-work-item program boundary.
    num_real_tokens = num_tokens - 4
    num_logical_experts = 13
    max_replicas = 3
    device = torch.device("cuda")

    token_idx = torch.arange(num_tokens, dtype=torch.int32)[:, None]
    slot_idx = torch.arange(top_k, dtype=torch.int32)[None, :]
    logical_ids = (3 * token_idx + 5 * slot_idx) % num_logical_experts
    logical_ids[0, 0] = -1
    logical_ids[1, top_k - 1] = num_logical_experts
    if top_k > 1:
        logical_ids[2, 1] = logical_ids[2, 0]
    topk_ids = logical_ids.to(device)

    replica_counts = torch.tensor(
        [1 + expert % max_replicas for expert in range(num_logical_experts)],
        dtype=torch.int32,
        device=device,
    )
    physical_map = torch.arange(
        num_logical_experts * max_replicas, dtype=torch.int32, device=device
    ).reshape(num_logical_experts, max_replicas)
    num_physical_experts = physical_map.numel()
    baseline_load = torch.zeros(num_physical_experts, dtype=torch.int32, device=device)
    actual_load = torch.zeros_like(baseline_load)
    actual_pairs = torch.zeros(
        (num_logical_experts, num_logical_experts),
        dtype=torch.int32,
        device=device,
    )
    load_flag = torch.tensor(record_load, dtype=torch.bool, device=device)
    pair_flag = torch.tensor(record_pairs, dtype=torch.bool, device=device)
    num_unpadded = torch.tensor(num_real_tokens, dtype=torch.int32, device=device)

    baseline_ids = _eplb_map_and_record_triton(
        topk_ids,
        physical_map,
        replica_counts,
        baseline_load,
        load_flag,
        num_unpadded,
    )
    actual_ids = eplb_corr_map_and_record_triton(
        topk_ids,
        physical_map,
        replica_counts,
        actual_load,
        actual_pairs,
        load_flag,
        pair_flag,
        num_unpadded,
    )

    torch.testing.assert_close(actual_ids, baseline_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_load, baseline_load, rtol=0, atol=0)

    expected_pairs = torch.zeros(
        (num_logical_experts, num_logical_experts), dtype=torch.int32
    )
    if record_pairs:
        for experts in logical_ids[:num_real_tokens].tolist():
            for first, second in combinations(experts, 2):
                if (
                    0 <= first < num_logical_experts
                    and 0 <= second < num_logical_experts
                    and first != second
                ):
                    expected_pairs[max(first, second), min(first, second)] += 1
    torch.testing.assert_close(actual_pairs.cpu(), expected_pairs, rtol=0, atol=0)


def test_without_unpadded_count_records_all_tokens() -> None:
    """An absent count includes all tokens, and repeated calls accumulate."""
    device = torch.device("cuda")
    topk_ids = torch.tensor([[0, 2], [1, 2], [0, 1]], device=device)
    replica_counts = torch.ones(3, dtype=torch.int32, device=device)
    physical_map = torch.arange(3, dtype=torch.int32, device=device)[:, None]
    expert_load = torch.zeros(3, dtype=torch.int32, device=device)
    expert_pairs = torch.zeros((3, 3), dtype=torch.int32, device=device)
    enabled = torch.tensor(True, dtype=torch.bool, device=device)

    mapped_ids = eplb_corr_map_and_record_triton(
        topk_ids,
        physical_map,
        replica_counts,
        expert_load,
        expert_pairs,
        enabled,
        enabled,
        None,
    )

    torch.testing.assert_close(mapped_ids, topk_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        expert_load.cpu(), torch.tensor([2, 2, 2], dtype=torch.int32),
        rtol=0, atol=0,
    )
    expected_pairs = torch.zeros((3, 3), dtype=torch.int32)
    expected_pairs[2, 0] = 1
    expected_pairs[2, 1] = 1
    expected_pairs[1, 0] = 1
    torch.testing.assert_close(expert_pairs.cpu(), expected_pairs, rtol=0, atol=0)

    eplb_corr_map_and_record_triton(
        topk_ids,
        physical_map,
        replica_counts,
        expert_load,
        expert_pairs,
        enabled,
        enabled,
        None,
    )
    torch.testing.assert_close(
        expert_load.cpu(), torch.tensor([4, 4, 4], dtype=torch.int32),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        expert_pairs.cpu(), expected_pairs * 2, rtol=0, atol=0
    )
