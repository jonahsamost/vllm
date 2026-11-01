import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _eplb_corr_map_and_record_i32_kernel(
    topk_ids_ptr,
    logical_replica_count_ptr,
    logical_to_physical_ptr,
    out_ids_ptr,
    expert_load_ptr,
    expert_pair_ptr,
    record_enabled_ptr,
    pair_record_enabled_ptr,
    num_unpadded_tokens_ptr,
    num_logical_experts,
    map_slots,
    num_physical_experts,
    num_tokens,
    TOP_K: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
    ITEMS_PER_TOKEN: tl.constexpr,
    HAS_NUM_UNPADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token = idx // ITEMS_PER_TOKEN
    local_idx = idx % ITEMS_PER_TOKEN
    valid_token = token < num_tokens

    real_token = valid_token
    if HAS_NUM_UNPADDED:
        real_token = real_token & (
            token < tl.load(num_unpadded_tokens_ptr)
        )

    map_mask = valid_token & (local_idx < TOP_K)
    logical_id = tl.load(
        topk_ids_ptr + token * TOP_K + local_idx,
        mask=map_mask,
        other=-1
    ).to(tl.int64)
    valid_logical = (
        (logical_id >= 0) & (logical_id < num_logical_experts)
    )
    safe_logical = tl.where(valid_logical, logical_id, 0)

    replica_count = tl.load(
        logical_replica_count_ptr + safe_logical,
        mask=map_mask & valid_logical,
        other=-1
    )
    replica_count = tl.maximum(replica_count, 1)

    hashed_token = (token.to(tl.int64) * 2654435769) & 0xFFFFFFFF
    replica_idx = hashed_token % replica_count
    physical_id = tl.load(
        logical_to_physical_ptr + safe_logical * map_slots + replica_idx,
        mask=map_mask & valid_logical,
        other=-1
    )
    tl.store(
        out_ids_ptr + token * TOP_K + local_idx,
        physical_id,
        mask=map_mask,
    )

    if tl.load(record_enabled_ptr) != 0:
        load_mask = (
            map_mask
            & real_token
            & (physical_id >= 0)
            & (physical_id < num_physical_experts)
        )
        tl.atomic_add(
            expert_load_ptr + tl.maximum(physical_id, 0),
            1,
            mask=load_mask
        )

    if NUM_PAIRS > 0:
        if tl.load(pair_record_enabled_ptr) != 0:
            pair_mask = real_token & (local_idx < NUM_PAIRS)
            pair_idx = local_idx

            first_slot = (
                (1.0 + tl.sqrt_rn((8 * pair_idx + 1).to(tl.float32)))
                * 0.5
            ).to(tl.int32)
            second_slot = pair_idx - first_slot * (first_slot - 1) // 2

            first_id = tl.load(
                topk_ids_ptr + token * TOP_K + first_slot,
                mask=pair_mask,
                other=-1,
            )
            second_id = tl.load(
                topk_ids_ptr + token * TOP_K + second_slot,
                mask=pair_mask,
                other=-1,
            )
            valid_pair = (
                pair_mask
                & (first_id >= 0)
                & (first_id < num_logical_experts)
                & (second_id >= 0)
                & (second_id < num_logical_experts)
                & (first_id != second_id)
            )
            row = tl.maximum(first_id, second_id)
            col = tl.minimum(first_id, second_id)
            pair_offset = tl.where(
                valid_pair,
                row * num_logical_experts + col,
                0,
            )
            tl.atomic_add(
                expert_pair_ptr + pair_offset,
                1,
                mask=valid_pair,
            )


def eplb_corr_map_and_record_triton(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_load_view: torch.Tensor,
    expert_token_view: torch.Tensor,
    record_enabled: torch.Tensor,
    pair_record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor | None,
) -> torch.Tensor:
    """Map experts and record physical loads and logical expert pairs."""
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [num_tokens, top_k]")

    num_tokens, top_k = topk_ids.shape
    if num_tokens == 0:
        return topk_ids

    num_logical_experts = logical_replica_count.shape[0]
    if expert_token_view.shape != (
        num_logical_experts,
        num_logical_experts,
    ):
        raise ValueError(
            "expert_token_view must have shape "
            "[num_logical_experts, num_logical_experts]"
        )
    if not expert_load_view.is_contiguous():
        raise ValueError("expert_load_view must be contiguous")
    if not expert_token_view.is_contiguous():
        raise ValueError("expert_token_view must be contiguous")

    num_pairs = top_k * (top_k - 1) // 2
    items_per_token = max(top_k, num_pairs)
    num_warps = 8
    block_size = num_warps * 32
    grid = (
        triton.cdiv(num_tokens * items_per_token, block_size),
    )

    topk_ids_in = topk_ids.contiguous().to(torch.int32)
    out_flat = torch.empty(
        (topk_ids.numel(),),
        device=topk_ids.device,
        dtype=topk_ids.dtype,
    )

    _eplb_corr_map_and_record_i32_kernel[grid](
        topk_ids_in,
        logical_replica_count.contiguous(),
        logical_to_physical_map.contiguous(),
        out_flat,
        expert_load_view,
        expert_token_view,
        record_enabled,
        pair_record_enabled,
        num_unpadded_tokens,
        num_logical_experts,
        logical_to_physical_map.shape[1],
        expert_load_view.numel(),
        num_tokens,
        TOP_K=top_k,
        NUM_PAIRS=num_pairs,
        ITEMS_PER_TOKEN=items_per_token,
        HAS_NUM_UNPADDED=num_unpadded_tokens is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_flat.reshape(topk_ids.shape)
