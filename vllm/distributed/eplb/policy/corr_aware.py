# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from .default import DefaultEplbPolicy
from .pressure import rebalance_pressure


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
        placement_strategy: str = "greedy",
        max_swaps: int = 128,
        overlap_curvature: float = 0.5,
        mean_pressure_weight: float = 0.25,
        random_restarts: int = 0,
        perturb_swaps: int = 16,
        random_seed: int = 0,
        layer_ids: list[int] | None = None,
        active_layer_ids: list[int] | None = None,
        layer_max_swaps: dict[str, int] | None = None,
        layer_random_restarts: dict[str, int] | None = None,
        source_expert_loads: torch.Tensor | None = None,
        locality_permutation: bool = False,
        locality_permutation_layers: list[int] | None = None,
        layer_overlap_curvature: dict[str, float] | None = None,
        layer_mean_pressure_weight: dict[str, float] | None = None,
        training_route_ids: torch.Tensor | None = None,
        training_route_hashes: torch.Tensor | None = None,
        training_step_route_ids: torch.Tensor | None = None,
        training_step_route_hashes: torch.Tensor | None = None,
        training_step_route_offsets: torch.Tensor | None = None,
        training_step_route_sources: torch.Tensor | None = None,
        route_search_steps: int = 0,
        route_search_proposals: int = 256,
        route_search_mean_weight: float = 0.25,
        route_search_layers: list[int] | None = None,
        step_search_steps: int = 0,
        step_search_proposals: int = 128,
        step_search_skew_slack: float = 0.2,
        step_search_skew_penalty: float = 10.0,
        step_search_layers: list[int] | None = None,
        step_search_replica_move_fraction: float = 0.0,
        step_search_cycle_move_fraction: float = 0.0,
        step_search_block_swap_fraction: float = 0.0,
        step_search_guided_swap_fraction: float = 0.0,
        step_search_guided_replica_fraction: float = 0.0,
        step_search_exhaustive_replica_rounds: int = 0,
        step_search_restarts: int = 1,
        step_search_assignment_load_cap_ratio: float | None = None,
        step_search_assignment_load_cap_penalty: float = 1000.0,
        step_search_source_send_weight: float = 0.0,
        step_search_assignment_skew_slack: float = 0.05,
        step_search_assignment_skew_penalty: float = 0.0,
        step_search_assignment_skew_aggregation: str = "mean",
        step_search_objective: str = "fanout",
        step_search_assignment_weight: float = 0.2,
        step_search_fanout_weight: float = 0.05,
        step_search_method: str = "greedy",
        layer_step_search_method: dict[str, str] | None = None,
        step_search_anneal_start: float = 0.05,
        step_search_anneal_final: float = 0.002,
        step_search_anneal_polish_steps: int = 8,
        layer_step_search_steps: dict[str, int] | None = None,
        layer_step_search_proposals: dict[str, int] | None = None,
        layer_step_search_skew_slack: dict[str, float] | None = None,
        layer_step_search_replica_move_fraction: dict[str, float] | None = None,
        layer_step_search_cycle_move_fraction: dict[str, float] | None = None,
        layer_step_search_block_swap_fraction: dict[str, float] | None = None,
        layer_step_search_guided_swap_fraction: dict[str, float] | None = None,
        layer_step_search_guided_replica_fraction: dict[str, float] | None = None,
        layer_step_search_exhaustive_replica_rounds: dict[str, int] | None = None,
        layer_step_search_restarts: dict[str, int] | None = None,
        layer_step_search_assignment_load_cap_ratio: dict[str, float] | None = None,
        layer_step_search_source_send_weight: dict[str, float] | None = None,
        layer_step_search_assignment_skew_penalty: dict[str, float] | None = None,
        layer_step_search_assignment_skew_aggregation: dict[str, str] | None = None,
        layer_step_search_objective: dict[str, str] | None = None,
        layer_step_search_assignment_weight: dict[str, float] | None = None,
        layer_step_search_fanout_weight: dict[str, float] | None = None,
        source_aware_permutation_layers: list[int] | None = None,
        source_aware_permutation: bool = False,
        source_aware_fanout_weight: float = 0.1,
        hash_search_steps: int = 0,
        hash_search_proposals: int = 256,
        hash_search_mean_weight: float = 0.25,
        hash_search_layers: list[int] | None = None,
        training_wave_loads: torch.Tensor | None = None,
        training_wave_pairs: torch.Tensor | None = None,
        training_wave_tokens: torch.Tensor | None = None,
        wave_search_steps: int = 0,
        wave_search_proposals: int = 256,
        wave_search_mean_weight: float = 0.25,
        wave_weight_power: float = 0.0,
        wave_search_layers: list[int] | None = None,
        replica_partner_dispersion_weight: float = 0.0,
        replica_selection_layers: list[int] | None = None,
        replica_pair_selection_layers: list[int] | None = None,
        replica_pair_pool_size: int = 64,
        replica_pair_score: str = "raw",
        replica_pair_extras: int = 8,
        layer_replica_pair_extras: dict[str, int] | None = None,
        wave_replica_layers: list[int] | None = None,
        wave_replica_min_tokens: int = 30000,
        slot_order: str = "original",
        layer_slot_order: dict[str, str] | None = None,
        placement_seed: str = "default",
        layer_placement_seed: dict[str, str] | None = None,
        spectral_similarity: str = "cosine",
        layer_spectral_similarity: dict[str, str] | None = None,
        spectral_seed_layers: list[int] | None = None,
        matching_group_size: int = 2,
        matching_affinity_weight: float = 0.5,
        matching_load_penalty: float = 0.0,
        matching_load_cap_ratio: float = 1.5,
        layer_matching_load_penalty: dict[str, float] | None = None,
        spectral_extra_assignment: str = "matching",
        anneal_steps: int = 0,
        anneal_start_temperature: float = 1500.0,
        anneal_final_temperature: float = 50.0,
        anneal_replica_move_probability: float = 0.1,
        anneal_polish_swaps: int = 128,
        anneal_layers: list[int] | None = None,
        graph_seed_correlation_weight: float = 0.5,
        hypergraph_seed_layers: list[int] | None = None,
        hypergraph_correlation_weight: float = 0.7,
        hypergraph_arrival_cap_ratio: float | None = None,
        hypergraph_assignment_cap_ratio: float | None = None,
        hypergraph_wave_load_cap_ratio: float | None = None,
        hypergraph_wave_min_tokens: int = 0,
        hypergraph_seed_objective: str = "affinity",
        hypergraph_load_weight: float = 0.5,
        spectral_kmeans_restarts: int = 4,
        spectral_kmeans_dimensions: int | None = None,
        spectral_kmeans_selection: str = "pressure",
        spectral_kmeans_load_penalty: float = 0.0,
        layer_spectral_kmeans_load_penalty: dict[str, float] | None = None,
        spectral_kmeans_replica_adjustment: bool = False,
        kl_neighbors: int = 24,
        kl_restarts: int = 2,
        louvain_resolution: float = 3.0,
        louvain_neighbors: int = 24,
        louvain_affinity_weight: float = 0.5,
    ) -> torch.Tensor:
        if correlation_weight is not None and not 0 <= correlation_weight <= 1:
            raise ValueError("correlation_weight must be between 0 and 1")
        if placement_strategy not in ("greedy", "pressure_swaps"):
            raise ValueError("Unknown correlation-aware placement strategy")
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

        if placement_strategy == "pressure_swaps":
            return rebalance_pressure(
                weight,
                num_replicas,
                num_groups,
                num_nodes,
                num_ranks,
                old_global_expert_indices,
                expert_pair_counts,
                max_swaps=max_swaps,
                overlap_curvature=overlap_curvature,
                mean_pressure_weight=mean_pressure_weight,
                random_restarts=random_restarts,
                perturb_swaps=perturb_swaps,
                random_seed=random_seed,
                layer_ids=layer_ids,
                active_layer_ids=active_layer_ids,
                layer_max_swaps=layer_max_swaps,
                layer_random_restarts=layer_random_restarts,
                source_expert_loads=source_expert_loads,
                locality_permutation=locality_permutation,
                locality_permutation_layers=locality_permutation_layers,
                layer_overlap_curvature=layer_overlap_curvature,
                layer_mean_pressure_weight=layer_mean_pressure_weight,
                training_route_ids=training_route_ids,
                training_route_hashes=training_route_hashes,
                training_step_route_ids=training_step_route_ids,
                training_step_route_hashes=training_step_route_hashes,
                training_step_route_offsets=training_step_route_offsets,
                training_step_route_sources=training_step_route_sources,
                route_search_steps=route_search_steps,
                route_search_proposals=route_search_proposals,
                route_search_mean_weight=route_search_mean_weight,
                route_search_layers=route_search_layers,
                step_search_steps=step_search_steps,
                step_search_proposals=step_search_proposals,
                step_search_skew_slack=step_search_skew_slack,
                step_search_skew_penalty=step_search_skew_penalty,
                step_search_layers=step_search_layers,
                step_search_replica_move_fraction=step_search_replica_move_fraction,
                step_search_cycle_move_fraction=step_search_cycle_move_fraction,
                layer_step_search_cycle_move_fraction=layer_step_search_cycle_move_fraction,
                step_search_block_swap_fraction=step_search_block_swap_fraction,
                layer_step_search_block_swap_fraction=layer_step_search_block_swap_fraction,
                step_search_guided_swap_fraction=step_search_guided_swap_fraction,
                layer_step_search_guided_swap_fraction=layer_step_search_guided_swap_fraction,
                step_search_guided_replica_fraction=step_search_guided_replica_fraction,
                layer_step_search_guided_replica_fraction=layer_step_search_guided_replica_fraction,
                step_search_exhaustive_replica_rounds=step_search_exhaustive_replica_rounds,
                layer_step_search_exhaustive_replica_rounds=layer_step_search_exhaustive_replica_rounds,
                step_search_restarts=step_search_restarts,
                layer_step_search_restarts=layer_step_search_restarts,
                step_search_assignment_load_cap_ratio=step_search_assignment_load_cap_ratio,
                step_search_assignment_load_cap_penalty=step_search_assignment_load_cap_penalty,
                layer_step_search_assignment_load_cap_ratio=layer_step_search_assignment_load_cap_ratio,
                step_search_source_send_weight=step_search_source_send_weight,
                layer_step_search_source_send_weight=layer_step_search_source_send_weight,
                step_search_assignment_skew_slack=step_search_assignment_skew_slack,
                step_search_assignment_skew_penalty=step_search_assignment_skew_penalty,
                step_search_assignment_skew_aggregation=step_search_assignment_skew_aggregation,
                step_search_objective=step_search_objective,
                step_search_assignment_weight=step_search_assignment_weight,
                step_search_fanout_weight=step_search_fanout_weight,
                step_search_method=step_search_method,
                layer_step_search_method=layer_step_search_method,
                step_search_anneal_start=step_search_anneal_start,
                step_search_anneal_final=step_search_anneal_final,
                step_search_anneal_polish_steps=step_search_anneal_polish_steps,
                layer_step_search_steps=layer_step_search_steps,
                layer_step_search_proposals=layer_step_search_proposals,
                layer_step_search_skew_slack=layer_step_search_skew_slack,
                layer_step_search_replica_move_fraction=layer_step_search_replica_move_fraction,
                layer_step_search_assignment_skew_penalty=layer_step_search_assignment_skew_penalty,
                layer_step_search_assignment_skew_aggregation=layer_step_search_assignment_skew_aggregation,
                layer_step_search_objective=layer_step_search_objective,
                layer_step_search_assignment_weight=layer_step_search_assignment_weight,
                layer_step_search_fanout_weight=layer_step_search_fanout_weight,
                source_aware_permutation_layers=source_aware_permutation_layers,
                source_aware_permutation=source_aware_permutation,
                source_aware_fanout_weight=source_aware_fanout_weight,
                hash_search_steps=hash_search_steps,
                hash_search_proposals=hash_search_proposals,
                hash_search_mean_weight=hash_search_mean_weight,
                hash_search_layers=hash_search_layers,
                training_wave_loads=training_wave_loads,
                training_wave_pairs=training_wave_pairs,
                training_wave_tokens=training_wave_tokens,
                wave_search_steps=wave_search_steps,
                wave_search_proposals=wave_search_proposals,
                wave_search_mean_weight=wave_search_mean_weight,
                wave_weight_power=wave_weight_power,
                wave_search_layers=wave_search_layers,
                replica_partner_dispersion_weight=replica_partner_dispersion_weight,
                replica_selection_layers=replica_selection_layers,
                replica_pair_selection_layers=replica_pair_selection_layers,
                replica_pair_pool_size=replica_pair_pool_size,
                replica_pair_score=replica_pair_score,
                replica_pair_extras=replica_pair_extras,
                layer_replica_pair_extras=layer_replica_pair_extras,
                wave_replica_layers=wave_replica_layers,
                wave_replica_min_tokens=wave_replica_min_tokens,
                slot_order=slot_order,
                layer_slot_order=layer_slot_order,
                placement_seed=placement_seed,
                layer_placement_seed=layer_placement_seed,
                spectral_similarity=spectral_similarity,
                layer_spectral_similarity=layer_spectral_similarity,
                spectral_seed_layers=spectral_seed_layers,
                matching_group_size=matching_group_size,
                matching_affinity_weight=matching_affinity_weight,
                matching_load_penalty=matching_load_penalty,
                matching_load_cap_ratio=matching_load_cap_ratio,
                layer_matching_load_penalty=layer_matching_load_penalty,
                spectral_extra_assignment=spectral_extra_assignment,
                anneal_steps=anneal_steps,
                anneal_start_temperature=anneal_start_temperature,
                anneal_final_temperature=anneal_final_temperature,
                anneal_replica_move_probability=anneal_replica_move_probability,
                anneal_polish_swaps=anneal_polish_swaps,
                anneal_layers=anneal_layers,
                graph_seed_correlation_weight=graph_seed_correlation_weight,
                hypergraph_seed_layers=hypergraph_seed_layers,
                hypergraph_correlation_weight=hypergraph_correlation_weight,
                hypergraph_arrival_cap_ratio=hypergraph_arrival_cap_ratio,
                hypergraph_assignment_cap_ratio=hypergraph_assignment_cap_ratio,
                hypergraph_wave_load_cap_ratio=hypergraph_wave_load_cap_ratio,
                hypergraph_wave_min_tokens=hypergraph_wave_min_tokens,
                hypergraph_seed_objective=hypergraph_seed_objective,
                hypergraph_load_weight=hypergraph_load_weight,
                spectral_kmeans_restarts=spectral_kmeans_restarts,
                spectral_kmeans_dimensions=spectral_kmeans_dimensions,
                spectral_kmeans_selection=spectral_kmeans_selection,
                spectral_kmeans_load_penalty=spectral_kmeans_load_penalty,
                layer_spectral_kmeans_load_penalty=layer_spectral_kmeans_load_penalty,
                spectral_kmeans_replica_adjustment=spectral_kmeans_replica_adjustment,
                kl_neighbors=kl_neighbors,
                kl_restarts=kl_restarts,
                louvain_resolution=louvain_resolution,
                louvain_neighbors=louvain_neighbors,
                louvain_affinity_weight=louvain_affinity_weight,
            )

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
