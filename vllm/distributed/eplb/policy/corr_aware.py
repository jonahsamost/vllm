# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from .default import DefaultEplbPolicy


def _pair_count(counts: np.ndarray, a: int, b: int) -> int:
    if a == b:
        return 0
    return int(counts[max(a, b), min(a, b)])


def _choose_candidate(
    candidates: list[int],
    affinity: dict[int, float],
    load: list[float],
    correlation_weight: float | None,
) -> int:
    if correlation_weight is None:
        return max(
            candidates,
            key=lambda rank: (affinity[rank] - load[rank], -load[rank], -rank),
        )

    affinity_min = min(affinity[rank] for rank in candidates)
    affinity_span = max(affinity[rank] for rank in candidates) - affinity_min
    load_min = min(load[rank] for rank in candidates)
    load_span = max(load[rank] for rank in candidates) - load_min

    def score(rank: int) -> tuple[float, float, int]:
        correlation = (
            (affinity[rank] - affinity_min) / affinity_span if affinity_span else 0.0
        )
        normalized_load = (load[rank] - load_min) / load_span if load_span else 0.0
        return (
            correlation_weight * correlation
            - (1 - correlation_weight) * normalized_load,
            -load[rank],
            -rank,
        )

    return max(candidates, key=score)


def _node_units(
    num_experts: int, num_groups: int, num_nodes: int
) -> list[list[int]] | None:
    if num_nodes == 1:
        return [list(range(num_experts))]
    if num_groups > 1 and num_groups % num_nodes == 0:
        group_size = num_experts // num_groups
        return [
            list(range(group * group_size, (group + 1) * group_size))
            for group in range(num_groups)
        ]
    if num_experts % num_nodes == 0:
        return [[expert] for expert in range(num_experts)]
    return None


def _place_units_on_nodes(
    units: list[list[int]],
    weight: np.ndarray,
    counts: np.ndarray,
    num_nodes: int,
    correlation_weight: float | None,
) -> list[list[int]]:
    """Seed nodes by load, then place each remaining unit by pair affinity."""
    unit_load = [sum(float(weight[e]) for e in unit) for unit in units]
    ordered = sorted(range(len(units)), key=lambda u: (-unit_load[u], u))
    units_per_node = len(units) // num_nodes
    node_units: list[list[int]] = [[] for _ in range(num_nodes)]
    node_load = [0.0] * num_nodes

    for node, unit_idx in enumerate(ordered[:num_nodes]):
        node_units[node].append(unit_idx)
        node_load[node] += unit_load[unit_idx]

    for unit_idx in ordered[num_nodes:]:
        candidates = [
            node for node in range(num_nodes) if len(node_units[node]) < units_per_node
        ]
        affinity = {
            node: float(
                sum(
                    _pair_count(counts, expert, resident)
                    for placed_idx in node_units[node]
                    for expert in units[unit_idx]
                    for resident in units[placed_idx]
                )
            )
            for node in candidates
        }
        chosen = _choose_candidate(candidates, affinity, node_load, correlation_weight)
        node_units[chosen].append(unit_idx)
        node_load[chosen] += unit_load[unit_idx]

    return [
        [expert for unit_idx in assigned for expert in units[unit_idx]]
        for assigned in node_units
    ]


class CorrAwareEplbPolicy(DefaultEplbPolicy):
    @classmethod
    def _place_copies_on_ranks(
        cls,
        experts: list[int],
        weight: np.ndarray,
        counts: np.ndarray,
        num_copies: int,
        num_ranks: int,
        correlation_weight: float | None = None,
    ) -> list[list[int]]:
        """Replicate by load and place copies using affinity and rank load."""
        local_weight = weight[experts]
        _, replica_counts = cls.replicate_experts(local_weight[None, :], num_copies)
        replica_counts = replica_counts[0]
        copy_load = {
            expert: float(weight[expert]) / int(replica_counts[idx])
            for idx, expert in enumerate(experts)
        }
        ordered = sorted(experts, key=lambda expert: (-float(weight[expert]), expert))
        placed: list[list[int]] = [[] for _ in range(num_ranks)]
        rank_load = [0.0] * num_ranks
        slots_per_rank = num_copies // num_ranks

        for rank, expert in enumerate(ordered[:num_ranks]):
            placed[rank].append(expert)
            rank_load[rank] += copy_load[expert]
            replica_counts[experts.index(expert)] -= 1

        remaining = [
            expert
            for idx, expert in enumerate(experts)
            for _ in range(int(replica_counts[idx]))
        ]
        remaining.sort(key=lambda expert: (-copy_load[expert], expert))

        for expert in remaining:
            candidates = [
                rank for rank in range(num_ranks) if len(placed[rank]) < slots_per_rank
            ]
            distinct = [rank for rank in candidates if expert not in placed[rank]]
            if distinct:
                candidates = distinct

            affinity = {
                rank: float(
                    sum(
                        _pair_count(counts, expert, resident)
                        for resident in set(placed[rank])
                    )
                )
                for rank in candidates
            }
            chosen = _choose_candidate(
                candidates, affinity, rank_load, correlation_weight
            )
            placed[chosen].append(expert)
            rank_load[chosen] += copy_load[expert]

        return placed

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
        *,
        expert_pair_counts: torch.Tensor | None = None,
        correlation_weight: float | None = None,
    ) -> torch.Tensor:
        if correlation_weight is not None and not 0 <= correlation_weight <= 1:
            raise ValueError("correlation_weight must be between 0 and 1")
        if expert_pair_counts is None:
            return super().rebalance_experts(
                weight,
                num_replicas,
                num_groups,
                num_nodes,
                num_ranks,
                old_global_expert_indices,
            )

        num_layers, num_experts = weight.shape
        if num_experts % num_groups != 0:
            raise ValueError("Logical experts must be divisible by expert groups")
        if num_ranks % num_nodes != 0 or num_replicas % num_ranks != 0:
            raise ValueError("Ranks and physical experts must divide evenly")
        if num_replicas < num_experts:
            raise ValueError("Physical experts must cover every logical expert")
        if expert_pair_counts.shape != (num_layers, num_experts, num_experts):
            raise ValueError("Pair counts must have shape [layers, experts, experts]")

        weight_np = weight.float().cpu().numpy()
        counts_np = expert_pair_counts.cpu().numpy()
        units = _node_units(num_experts, num_groups, num_nodes)
        result = np.empty((num_layers, num_replicas), dtype=np.int64)
        ranks_per_node = num_ranks // num_nodes
        copies_per_node = num_replicas // num_nodes

        for layer in range(num_layers):
            if units is None:
                node_experts = [list(range(num_experts))]
                layer_ranks = num_ranks
                layer_copies = num_replicas
            else:
                node_experts = _place_units_on_nodes(
                    units,
                    weight_np[layer],
                    counts_np[layer],
                    num_nodes,
                    correlation_weight,
                )
                layer_ranks = ranks_per_node
                layer_copies = copies_per_node

            offset = 0
            for experts in node_experts:
                placed = cls._place_copies_on_ranks(
                    experts,
                    weight_np[layer],
                    counts_np[layer],
                    layer_copies,
                    layer_ranks,
                    correlation_weight,
                )
                for rank_experts in placed:
                    result[layer, offset : offset + len(rank_experts)] = rank_experts
                    offset += len(rank_experts)

        if (
            old_global_expert_indices is not None
            and old_global_expert_indices.shape == result.shape
        ):
            result = cls.preserve_intragpu_slots(
                result, num_ranks, old_global_expert_indices.cpu().numpy()
            )

        return torch.from_numpy(result)
