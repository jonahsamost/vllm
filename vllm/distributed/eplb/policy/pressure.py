# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Destination-pressure placement for correlation-aware expert balancing."""

from itertools import permutations

import numpy as np
import torch

from .default import DefaultEplbPolicy
from .route_search import (
    refine_by_hash_aware_pairs,
    refine_by_sampled_routes,
    refine_by_wave_pressure,
)


def _minimize_destination_pressure_swaps(
    placement: np.ndarray,
    weight: np.ndarray,
    lower_pair_counts: np.ndarray,
    num_ranks: int,
    max_swaps: int,
    overlap_curvature: float,
    mean_pressure_weight: float,
) -> np.ndarray:
    """Greedily reduce the busiest estimated destination rank.

    Pair overlap approximates duplicate token destinations. The quadratic term
    discounts repeated overlaps when three or more experts share one rank.
    """
    slots_per_rank = len(placement) // num_ranks
    rows = placement.reshape(num_ranks, slots_per_rank).copy()
    replica_counts = np.bincount(placement, minlength=len(weight))
    copy_load = weight / replica_counts
    lower = np.tril(lower_pair_counts, k=-1)
    pair_overlap = (lower + lower.T) / np.outer(replica_counts, replica_counts)
    assignment_load = copy_load[rows].sum(axis=1)
    overlap = np.array([pair_overlap[np.ix_(row, row)].sum() / 2 for row in rows])

    def pressure(assignments: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        return assignments - pairs + overlap_curvature * pairs**2 / assignments

    current = pressure(assignment_load, overlap)
    current_score = current.max() + mean_pressure_weight * current.mean()
    for _ in range(max_swaps):
        affinity = [pair_overlap[:, row].sum(axis=1) for row in rows]
        best_score = current_score - 1e-7
        best_swap = None
        for first_rank in range(num_ranks):
            first = rows[first_rank]
            for second_rank in range(first_rank + 1, num_ranks):
                second = rows[second_rank]
                cross_pair = pair_overlap[np.ix_(first, second)]
                first_overlap_delta = (
                    affinity[first_rank][second][None, :]
                    - affinity[first_rank][first][:, None]
                    - cross_pair
                )
                second_overlap_delta = (
                    affinity[second_rank][first][:, None]
                    - affinity[second_rank][second][None, :]
                    - cross_pair
                )
                first_load = (
                    assignment_load[first_rank]
                    - copy_load[first][:, None]
                    + copy_load[second][None, :]
                )
                second_load = (
                    assignment_load[second_rank]
                    - copy_load[second][None, :]
                    + copy_load[first][:, None]
                )
                first_pressure = pressure(
                    first_load, overlap[first_rank] + first_overlap_delta
                )
                second_pressure = pressure(
                    second_load, overlap[second_rank] + second_overlap_delta
                )
                other_max = np.max(
                    np.delete(current, [first_rank, second_rank]), initial=0
                )
                new_max = np.maximum(
                    np.maximum(first_pressure, second_pressure), other_max
                )
                new_mean = (
                    current.sum()
                    - current[first_rank]
                    - current[second_rank]
                    + first_pressure
                    + second_pressure
                ) / num_ranks
                score = new_max + mean_pressure_weight * new_mean
                valid = (
                    ~np.isin(first, second)[:, None] & ~np.isin(second, first)[None, :]
                )
                score = np.where(valid, score, np.inf)
                index = np.unravel_index(int(np.argmin(score)), score.shape)
                if score[index] < best_score:
                    best_score = float(score[index])
                    best_swap = (first_rank, index[0], second_rank, index[1])
        if best_swap is None:
            break
        first_rank, first_slot, second_rank, second_slot = best_swap
        first_expert = int(rows[first_rank, first_slot])
        second_expert = int(rows[second_rank, second_slot])
        first_overlap_delta = (
            affinity[first_rank][second_expert]
            - affinity[first_rank][first_expert]
            - pair_overlap[first_expert, second_expert]
        )
        second_overlap_delta = (
            affinity[second_rank][first_expert]
            - affinity[second_rank][second_expert]
            - pair_overlap[first_expert, second_expert]
        )
        rows[first_rank, first_slot] = second_expert
        rows[second_rank, second_slot] = first_expert
        load_delta = copy_load[second_expert] - copy_load[first_expert]
        assignment_load[first_rank] += load_delta
        assignment_load[second_rank] -= load_delta
        overlap[first_rank] += first_overlap_delta
        overlap[second_rank] += second_overlap_delta
        current = pressure(assignment_load, overlap)
        current_score = current.max() + mean_pressure_weight * current.mean()

    return rows.reshape(-1)


def _rank_destination_pressure(
    row: np.ndarray,
    replica_counts: np.ndarray,
    weight: np.ndarray,
    pair_counts: np.ndarray,
    overlap_curvature: float,
) -> float:
    copies = replica_counts[row]
    assignments = (weight[row] / copies).sum()
    pair_matrix = pair_counts[np.ix_(row, row)]
    overlap = (pair_matrix / np.outer(copies, copies)).sum() / 2
    return float(assignments - overlap + overlap_curvature * overlap**2 / assignments)


def _destination_pressure_score(
    placement: np.ndarray,
    weight: np.ndarray,
    lower_pair_counts: np.ndarray,
    num_ranks: int,
    overlap_curvature: float,
    mean_pressure_weight: float,
) -> float:
    rows = placement.reshape(num_ranks, -1)
    counts = np.bincount(placement, minlength=len(weight))
    lower = np.tril(lower_pair_counts, k=-1)
    pairs = lower + lower.T
    pressures = np.array(
        [
            _rank_destination_pressure(row, counts, weight, pairs, overlap_curvature)
            for row in rows
        ]
    )
    return float(pressures.max() + mean_pressure_weight * pressures.mean())


def _anneal_placement(
    placement: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    *,
    steps: int,
    start_temperature: float,
    final_temperature: float,
    replica_move_probability: float,
    polish_swaps: int,
    overlap_curvature: float,
    mean_pressure_weight: float,
    seed: int,
) -> np.ndarray:
    """Explore placements and replica choices across pressure barriers."""
    rng = np.random.default_rng(seed)
    slots_per_rank = len(placement) // num_ranks
    current = placement.copy()
    current_score = _destination_pressure_score(
        current,
        weight,
        lower_pairs,
        num_ranks,
        overlap_curvature,
        mean_pressure_weight,
    )
    best = current.copy()
    best_score = current_score
    candidate_experts = np.argsort(-weight)[:64]
    for step in range(steps):
        proposal = current.copy()
        if rng.random() < replica_move_probability:
            copies = np.bincount(current, minlength=len(weight))
            redundant_positions = np.flatnonzero(copies[current] > 1)
            source = int(rng.choice(redundant_positions))
            target_expert = int(rng.choice(candidate_experts))
            rank = source // slots_per_rank
            row = current[rank * slots_per_rank : (rank + 1) * slots_per_rank]
            if target_expert in row:
                continue
            proposal[source] = target_expert
        else:
            first, second = rng.choice(len(current), size=2, replace=False)
            first_rank = first // slots_per_rank
            second_rank = second // slots_per_rank
            if first_rank == second_rank:
                continue
            first_row = current[
                first_rank * slots_per_rank : (first_rank + 1) * slots_per_rank
            ]
            second_row = current[
                second_rank * slots_per_rank : (second_rank + 1) * slots_per_rank
            ]
            if current[first] in second_row or current[second] in first_row:
                continue
            proposal[first], proposal[second] = proposal[second], proposal[first]
        score = _destination_pressure_score(
            proposal,
            weight,
            lower_pairs,
            num_ranks,
            overlap_curvature,
            mean_pressure_weight,
        )
        temperature = start_temperature * (final_temperature / start_temperature) ** (
            step / max(steps - 1, 1)
        )
        if score < current_score or rng.random() < np.exp(
            (current_score - score) / temperature
        ):
            current = proposal
            current_score = score
            if score < best_score:
                best = proposal.copy()
                best_score = score
    return _minimize_destination_pressure_swaps(
        best,
        weight,
        lower_pairs,
        num_ranks,
        polish_swaps,
        overlap_curvature,
        mean_pressure_weight,
    )


def _perturb_placement(
    placement: np.ndarray,
    num_ranks: int,
    num_swaps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows = placement.reshape(num_ranks, -1).copy()
    completed = 0
    attempts = 0
    while completed < num_swaps and attempts < num_swaps * 64:
        attempts += 1
        first_rank, second_rank = rng.choice(num_ranks, size=2, replace=False)
        first_slot = int(rng.integers(rows.shape[1]))
        second_slot = int(rng.integers(rows.shape[1]))
        first_expert = rows[first_rank, first_slot]
        second_expert = rows[second_rank, second_slot]
        if first_expert in rows[second_rank] or second_expert in rows[first_rank]:
            continue
        rows[first_rank, first_slot] = second_expert
        rows[second_rank, second_slot] = first_expert
        completed += 1
    return rows.reshape(-1)


def _replicate_by_partner_dispersion(
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_replicas: int,
    num_ranks: int,
    dispersion_weight: float,
) -> np.ndarray:
    """Favor replicas of busy experts with diffuse token partners."""
    num_experts = len(weight)
    pairs = np.tril(lower_pairs, k=-1)
    pairs = pairs + pairs.T
    strongest_partner_fraction = np.divide(
        pairs.max(axis=1),
        weight,
        out=np.zeros(num_experts, dtype=np.float64),
        where=weight > 0,
    )
    effective = weight * (1 + dispersion_weight * (1 - strongest_partner_fraction))
    copies = np.ones(num_experts, dtype=np.int64)
    extras = []
    for _ in range(num_replicas - num_experts):
        expert = int(np.argmax(effective / copies))
        extras.append(expert)
        copies[expert] += 1
    physical = np.concatenate((np.arange(num_experts), extras))
    copy_weight = weight[physical] / copies[physical]
    packs, positions = DefaultEplbPolicy.balanced_packing(
        copy_weight[None, :], num_ranks
    )
    slots_per_rank = num_replicas // num_ranks
    result = np.empty(num_replicas, dtype=np.int64)
    result[packs[0] * slots_per_rank + positions[0]] = physical
    return result


def _replicate_correlated_pairs(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_replicas: int,
    num_ranks: int,
    pool_size: int,
    score_mode: str,
    paired_extras: int,
) -> np.ndarray:
    """Use redundant slots on disjoint hot expert pairs with shared tokens."""
    num_experts = len(weight)
    extras_needed = num_replicas - num_experts
    if (
        extras_needed < 0
        or extras_needed % 2
        or paired_extras < 0
        or paired_extras > extras_needed
        or paired_extras % 2
        or pool_size < paired_extras
    ):
        raise ValueError("Paired replica selection needs an even redundant budget")
    if paired_extras == 0:
        return initial.copy()
    pool = np.argsort(-weight, kind="stable")[:pool_size]
    pairs = np.tril(lower_pairs, k=-1)
    pairs = pairs + pairs.T
    first, second = np.triu_indices(len(pool), k=1)
    scores = pairs[pool[first], pool[second]]
    if score_mode == "cosine":
        scores = scores / np.sqrt(
            np.maximum(weight[pool[first]] * weight[pool[second]], 1)
        )
    elif score_mode == "excess":
        total_assignments = float(weight.sum())
        total_pairs = float(np.tril(lower_pairs, k=-1).sum())
        effective_topk = 1 + 2 * total_pairs / max(total_assignments, 1)
        token_count = total_assignments / effective_topk
        scores = scores - weight[pool[first]] * weight[pool[second]] / max(
            token_count, 1
        )
    elif score_mode != "raw":
        raise ValueError("Unknown paired-replica score")
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    used = np.zeros(num_experts, dtype=bool)
    candidate_order = order if paired_extras else order[:0]
    for index in candidate_order:
        a, b = int(pool[first[index]]), int(pool[second[index]])
        if used[a] or used[b]:
            continue
        selected.extend((a, b))
        used[a] = used[b] = True
        if len(selected) == paired_extras:
            break
    if len(selected) != paired_extras:
        raise ValueError("Not enough distinct experts for paired replicas")
    base_copies = np.bincount(initial, minlength=num_experts)
    base_extras = np.flatnonzero(base_copies > 1)
    base_extras = base_extras[np.argsort(-weight[base_extras], kind="stable")]
    for expert in base_extras:
        if len(selected) == extras_needed:
            break
        if not used[expert]:
            selected.append(int(expert))
            used[expert] = True
    for expert in np.argsort(-weight, kind="stable"):
        if len(selected) == extras_needed:
            break
        if not used[expert]:
            selected.append(int(expert))
            used[expert] = True
    if len(selected) != extras_needed:
        raise ValueError("Paired selection could not fill the replica budget")
    physical = np.concatenate((np.arange(num_experts), selected))
    copies = np.bincount(physical, minlength=num_experts)
    copy_weight = weight[physical] / copies[physical]
    packs, positions = DefaultEplbPolicy.balanced_packing(
        copy_weight[None, :], num_ranks
    )
    slots_per_rank = num_replicas // num_ranks
    result = np.empty(num_replicas, dtype=np.int64)
    result[packs[0] * slots_per_rank + positions[0]] = physical
    return result


def _spectral_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    similarity: str,
    extra_assignment: str,
) -> np.ndarray:
    """Recursively split the expert co-routing graph into equal rank groups."""
    num_experts = len(weight)
    slots_per_rank = len(initial) // num_ranks
    if num_experts % num_ranks or slots_per_rank != num_experts // num_ranks + 1:
        raise ValueError("Spectral seed requires one extra slot per rank")
    copies = np.bincount(initial, minlength=num_experts)
    graph = np.tril(lower_pairs, k=-1).astype(np.float64)
    graph += graph.T
    if similarity == "cosine":
        graph /= np.sqrt(np.maximum(weight[:, None] * weight[None, :], 1))
    elif similarity == "excess":
        assignments = float(weight.sum())
        pair_count = float(np.tril(lower_pairs, k=-1).sum())
        effective_topk = 1 + 2 * pair_count / max(assignments, 1)
        token_count = assignments / effective_topk
        expected = np.outer(weight, weight) / max(token_count, 1)
        graph = np.maximum(graph - expected, 0)

    def split(experts: np.ndarray, target_groups: int) -> list[np.ndarray]:
        if target_groups == 1:
            return [experts]
        subgraph = graph[np.ix_(experts, experts)]
        degree = subgraph.sum(axis=1)
        scale = np.sqrt(np.maximum(degree, 1e-12))
        laplacian = np.eye(len(experts)) - subgraph / (scale[:, None] * scale[None, :])
        _, eigenvectors = np.linalg.eigh(laplacian)
        fiedler = eigenvectors[:, 1]
        pivot = int(np.argmax(np.abs(fiedler)))
        if fiedler[pivot] < 0:
            fiedler = -fiedler
        order = np.lexsort((experts, fiedler))
        midpoint = len(experts) // 2
        return split(experts[order[:midpoint]], target_groups // 2) + split(
            experts[order[midpoint:]], target_groups // 2
        )

    groups = split(np.arange(num_experts), num_ranks)
    rank_rows = [group.tolist() for group in groups]
    if extra_assignment == "greedy":
        return _assign_redundant_copies_greedy(rank_rows, copies, weight, lower_pairs)
    return _assign_redundant_copies(rank_rows, copies, weight, lower_pairs)


def _replicate_by_wave_peak(
    weight: np.ndarray,
    wave_loads: np.ndarray,
    wave_tokens: np.ndarray,
    num_replicas: int,
    num_ranks: int,
    min_tokens: int,
) -> np.ndarray:
    """Spend redundant slots on peak per-forward expert loads."""
    eligible = wave_tokens >= min_tokens
    if not np.any(eligible):
        raise ValueError("No training wave meets the replica threshold")
    peaks = (wave_loads[eligible] / wave_tokens[eligible, None]).max(axis=0)
    num_experts = len(weight)
    copies = np.ones(num_experts, dtype=np.int32)
    extras = []
    for _ in range(num_replicas - num_experts):
        expert = int(np.argmax(peaks / copies))
        extras.append(expert)
        copies[expert] += 1
    physical = np.concatenate((np.arange(num_experts), extras))
    copy_weight = weight[physical] / copies[physical]
    packs, positions = DefaultEplbPolicy.balanced_packing(
        copy_weight[None, :], num_ranks
    )
    slots_per_rank = num_replicas // num_ranks
    result = np.empty(num_replicas, dtype=np.int64)
    result[packs[0] * slots_per_rank + positions[0]] = physical
    return result


def _assign_redundant_copies_greedy(
    rank_rows: list[list[int]],
    copies: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
) -> np.ndarray:
    rank_loads = [float((weight[row] / copies[row]).sum()) for row in rank_rows]
    pair_counts = np.tril(lower_pairs, k=-1)
    pair_counts = pair_counts + pair_counts.T
    extras = [
        expert for expert in range(len(weight)) for _ in range(copies[expert] - 1)
    ]
    extras.sort(key=lambda expert: (-float(weight[expert] / copies[expert]), expert))

    def pressure(rank: int, expert: int) -> tuple[float, int]:
        row = rank_rows[rank]
        overlap = sum(
            pair_counts[expert, resident] / (copies[expert] * copies[resident])
            for resident in row
        )
        return rank_loads[rank] + weight[expert] / copies[expert] - overlap, rank

    for expert in extras:
        base_size = len(weight) // len(rank_rows)
        available = [
            rank
            for rank, row in enumerate(rank_rows)
            if len(row) == base_size and expert not in row
        ]
        if not available:
            raise ValueError("Greedy redundant-copy assignment has no legal rank")
        selected = min(available, key=lambda rank: pressure(rank, expert))
        rank_rows[selected].append(expert)
        rank_loads[selected] += weight[expert] / copies[expert]
    return np.array([expert for row in rank_rows for expert in row], dtype=np.int64)


def _assign_redundant_copies(
    rank_rows: list[list[int]],
    copies: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
) -> np.ndarray:
    num_ranks = len(rank_rows)
    num_experts = len(weight)
    rank_loads = [float((weight[row] / copies[row]).sum()) for row in rank_rows]
    pair_counts = np.tril(lower_pairs, k=-1)
    pair_counts = pair_counts + pair_counts.T
    extras = [
        expert for expert in range(num_experts) for _ in range(copies[expert] - 1)
    ]
    extras.sort(key=lambda expert: (-float(weight[expert] / copies[expert]), expert))

    if len(extras) != num_ranks or num_ranks > 8:
        raise ValueError("Seed supports up to eight one-extra-slot ranks")
    costs = np.full((num_ranks, num_ranks), np.inf)
    for extra, expert in enumerate(extras):
        for rank, row in enumerate(rank_rows):
            if expert in row:
                continue
            overlap = sum(
                pair_counts[expert, resident] / (copies[expert] * copies[resident])
                for resident in row
            )
            costs[extra, rank] = (
                rank_loads[rank] + weight[expert] / copies[expert] - overlap
            )
    best_score = np.inf
    best_assignment = None
    for assignment in permutations(range(num_ranks)):
        final_costs = costs[np.arange(num_ranks), assignment]
        if not np.isfinite(final_costs).all():
            continue
        score = final_costs.max() + 0.25 * final_costs.mean()
        if score < best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None:
        raise ValueError("No legal assignment for redundant experts")
    for expert, rank in zip(extras, best_assignment, strict=True):
        rank_rows[rank].append(expert)
    return np.array([expert for row in rank_rows for expert in row], dtype=np.int64)


def _matching_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    group_size: int,
    affinity_weight: float,
    similarity: str,
    load_penalty: float,
    load_cap_ratio: float,
) -> np.ndarray:
    """Build correlated expert blocks, then pack them with load awareness."""
    num_experts = len(weight)
    slots_per_rank = len(initial) // num_ranks
    if num_experts % (num_ranks * group_size):
        raise ValueError("Matching groups must divide experts per rank")
    if slots_per_rank != num_experts // num_ranks + 1:
        raise ValueError("Matching seed needs one extra slot per rank")
    copies = np.bincount(initial, minlength=num_experts)
    raw_pairs = np.tril(lower_pairs, k=-1).astype(np.float64)
    raw_pairs += raw_pairs.T
    graph = raw_pairs.copy()
    if similarity == "cosine":
        graph /= np.sqrt(np.maximum(weight[:, None] * weight[None, :], 1))
    elif similarity == "excess":
        total_assignments = float(weight.sum())
        total_pairs = float(np.tril(lower_pairs, k=-1).sum())
        effective_topk = 1 + 2 * total_pairs / max(total_assignments, 1)
        token_count = total_assignments / effective_topk
        expected = np.outer(weight, weight) / max(token_count, 1)
        graph = np.maximum(graph - expected, 0)
    copy_load = weight / copies
    blocks: list[list[int]] = [[expert] for expert in range(num_experts)]
    while len(blocks[0]) < group_size:
        membership = np.zeros((len(blocks), num_experts), dtype=np.float64)
        for index, block in enumerate(blocks):
            membership[index, block] = 1
        block_affinity = membership @ graph @ membership.T
        first, second = np.triu_indices(len(blocks), k=1)
        score = block_affinity[first, second]
        if load_penalty:
            block_load = np.array(
                [copy_load[block].sum() for block in blocks], dtype=np.float64
            )
            target = 2 * len(blocks[0]) * float(copy_load.mean())
            excess = np.maximum(
                (block_load[first] + block_load[second]) / target
                - load_cap_ratio,
                0,
            )
            score = score / max(float(score.max()), 1) - load_penalty * excess
        order = np.argsort(-score, kind="stable")
        used = np.zeros(len(blocks), dtype=bool)
        merged = []
        for edge in order:
            left, right = int(first[edge]), int(second[edge])
            if used[left] or used[right]:
                continue
            merged.append(blocks[left] + blocks[right])
            used[left] = used[right] = True
        if len(merged) * 2 != len(blocks):
            raise ValueError("Matching did not cover all experts")
        blocks = merged
    blocks.sort(key=lambda block: (-float(copy_load[block].sum()), min(block)))
    groups_per_rank = num_experts // (num_ranks * group_size)
    rank_rows: list[list[int]] = [[] for _ in range(num_ranks)]
    rank_load = np.zeros(num_ranks, dtype=np.float64)

    def cost(rank: int, block: list[int], block_load: float) -> tuple[float, int]:
        row = rank_rows[rank]
        overlap = (
            raw_pairs[np.ix_(block, row)] / (copies[block, None] * copies[None, row])
        ).sum()
        return rank_load[rank] + block_load - affinity_weight * overlap, rank

    for rank, block in enumerate(blocks[:num_ranks]):
        rank_rows[rank].extend(block)
        rank_load[rank] += copy_load[block].sum()
    for block in blocks[num_ranks:]:
        candidates = [
            rank
            for rank, row in enumerate(rank_rows)
            if len(row) < groups_per_rank * group_size
        ]
        block_load = copy_load[block].sum()

        selected = min(candidates, key=lambda rank: cost(rank, block, block_load))
        rank_rows[selected].extend(block)
        rank_load[selected] += block_load
    return _assign_redundant_copies(rank_rows, copies, weight, lower_pairs)


def _permute_ranks_for_locality(
    placement: np.ndarray,
    source_loads: np.ndarray,
    num_ranks: int,
    num_experts: int,
) -> np.ndarray:
    """Assign placement rows to ranks to maximize training local assignments."""
    rows = placement.reshape(num_ranks, -1)
    copies = np.bincount(placement, minlength=num_experts)
    local_score = np.stack(
        [(source_loads[:, row] / copies[row]).sum(axis=1) for row in rows], axis=1
    )
    # dp[mask] holds the best assignment of the first popcount(mask) ranks.
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for mask in range(1 << num_ranks):
        if mask not in dp:
            continue
        score, assignment = dp[mask]
        rank = len(assignment)
        if rank == num_ranks:
            continue
        for row in range(num_ranks):
            if mask & (1 << row):
                continue
            next_mask = mask | (1 << row)
            candidate = (score + local_score[rank, row], assignment + (row,))
            current = dp.get(next_mask)
            if current is None or candidate[0] > current[0] + 1e-9:
                dp[next_mask] = candidate
    order = dp[(1 << num_ranks) - 1][1]
    return rows[list(order)].reshape(-1)


def rebalance_pressure(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_ranks: int,
    old_global_expert_indices: torch.Tensor | None,
    expert_pair_counts: torch.Tensor,
    *,
    max_swaps: int,
    overlap_curvature: float,
    mean_pressure_weight: float,
    random_restarts: int,
    perturb_swaps: int,
    random_seed: int,
    layer_ids: list[int] | None,
    active_layer_ids: list[int] | None,
    layer_max_swaps: dict[str, int] | None,
    layer_random_restarts: dict[str, int] | None,
    source_expert_loads: torch.Tensor | None,
    locality_permutation: bool,
    locality_permutation_layers: list[int] | None,
    layer_overlap_curvature: dict[str, float] | None,
    layer_mean_pressure_weight: dict[str, float] | None,
    training_route_ids: torch.Tensor | None,
    training_route_hashes: torch.Tensor | None,
    training_step_route_ids: torch.Tensor | None,
    training_step_route_hashes: torch.Tensor | None,
    training_step_route_offsets: torch.Tensor | None,
    training_step_route_sources: torch.Tensor | None,
    route_search_steps: int,
    route_search_proposals: int,
    route_search_mean_weight: float,
    route_search_layers: list[int] | None,
    step_search_steps: int,
    step_search_proposals: int,
    step_search_skew_slack: float,
    step_search_skew_penalty: float,
    step_search_layers: list[int] | None,
    step_search_replica_move_fraction: float,
    step_search_cycle_move_fraction: float,
    layer_step_search_cycle_move_fraction: dict[str, float] | None,
    step_search_block_swap_fraction: float,
    layer_step_search_block_swap_fraction: dict[str, float] | None,
    step_search_guided_swap_fraction: float,
    layer_step_search_guided_swap_fraction: dict[str, float] | None,
    step_search_guided_replica_fraction: float,
    layer_step_search_guided_replica_fraction: dict[str, float] | None,
    step_search_exhaustive_replica_rounds: int,
    layer_step_search_exhaustive_replica_rounds: dict[str, int] | None,
    step_search_restarts: int,
    layer_step_search_restarts: dict[str, int] | None,
    step_search_assignment_load_cap_ratio: float | None,
    step_search_assignment_load_cap_penalty: float,
    layer_step_search_assignment_load_cap_ratio: dict[str, float] | None,
    step_search_source_send_weight: float,
    layer_step_search_source_send_weight: dict[str, float] | None,
    step_search_assignment_skew_slack: float,
    step_search_assignment_skew_penalty: float,
    step_search_assignment_skew_aggregation: str,
    step_search_objective: str,
    step_search_assignment_weight: float,
    step_search_fanout_weight: float,
    step_search_method: str,
    layer_step_search_method: dict[str, str] | None,
    step_search_anneal_start: float,
    step_search_anneal_final: float,
    step_search_anneal_polish_steps: int,
    layer_step_search_steps: dict[str, int] | None,
    layer_step_search_proposals: dict[str, int] | None,
    layer_step_search_skew_slack: dict[str, float] | None,
    layer_step_search_replica_move_fraction: dict[str, float] | None,
    layer_step_search_assignment_skew_penalty: dict[str, float] | None,
    layer_step_search_assignment_skew_aggregation: dict[str, str] | None,
    layer_step_search_objective: dict[str, str] | None,
    layer_step_search_assignment_weight: dict[str, float] | None,
    layer_step_search_fanout_weight: dict[str, float] | None,
    source_aware_permutation_layers: list[int] | None,
    source_aware_permutation: bool,
    source_aware_fanout_weight: float,
    hash_search_steps: int,
    hash_search_proposals: int,
    hash_search_mean_weight: float,
    hash_search_layers: list[int] | None,
    training_wave_loads: torch.Tensor | None,
    training_wave_pairs: torch.Tensor | None,
    training_wave_tokens: torch.Tensor | None,
    wave_search_steps: int,
    wave_search_proposals: int,
    wave_search_mean_weight: float,
    wave_weight_power: float,
    wave_search_layers: list[int] | None,
    replica_partner_dispersion_weight: float,
    replica_selection_layers: list[int] | None,
    replica_pair_selection_layers: list[int] | None,
    replica_pair_pool_size: int,
    replica_pair_score: str,
    replica_pair_extras: int,
    layer_replica_pair_extras: dict[str, int] | None,
    wave_replica_layers: list[int] | None,
    wave_replica_min_tokens: int,
    slot_order: str,
    layer_slot_order: dict[str, str] | None,
    placement_seed: str,
    layer_placement_seed: dict[str, str] | None,
    spectral_similarity: str,
    layer_spectral_similarity: dict[str, str] | None,
    spectral_seed_layers: list[int] | None,
    spectral_extra_assignment: str,
    matching_group_size: int,
    matching_affinity_weight: float,
    matching_load_penalty: float,
    matching_load_cap_ratio: float,
    layer_matching_load_penalty: dict[str, float] | None,
    anneal_steps: int,
    anneal_start_temperature: float,
    anneal_final_temperature: float,
    anneal_replica_move_probability: float,
    anneal_polish_swaps: int,
    anneal_layers: list[int] | None,
    graph_seed_correlation_weight: float,
    hypergraph_seed_layers: list[int] | None,
    hypergraph_correlation_weight: float,
    hypergraph_arrival_cap_ratio: float | None,
    hypergraph_assignment_cap_ratio: float | None,
    hypergraph_wave_load_cap_ratio: float | None,
    hypergraph_wave_min_tokens: int,
    hypergraph_seed_objective: str,
    hypergraph_load_weight: float,
    spectral_kmeans_restarts: int,
    spectral_kmeans_dimensions: int | None,
    spectral_kmeans_selection: str,
    spectral_kmeans_load_penalty: float,
    layer_spectral_kmeans_load_penalty: dict[str, float] | None,
    spectral_kmeans_replica_adjustment: bool,
    kl_neighbors: int,
    kl_restarts: int,
    louvain_resolution: float,
    louvain_neighbors: int,
    louvain_affinity_weight: float,
) -> torch.Tensor:
    """Place copies using pair-derived destination pressure, layer by layer."""
    num_layers, num_experts = weight.shape
    if num_groups != 1 or num_nodes != 1:
        raise ValueError("Destination-pressure swaps support one group and node")
    if num_replicas < num_experts or num_replicas % num_ranks != 0:
        raise ValueError("Physical slots must cover experts and divide by ranks")
    if expert_pair_counts.shape != (num_layers, num_experts, num_experts):
        raise ValueError("Pair counts must have shape [layers, experts, experts]")
    if (locality_permutation or locality_permutation_layers) and (
        source_expert_loads is None
        or source_expert_loads.shape != (num_layers, num_ranks, num_experts)
    ):
        raise ValueError("Locality permutation requires per-source-rank loads")
    if route_search_steps < 0 or route_search_proposals <= 0:
        raise ValueError("Route search steps and proposals must be nonnegative")
    if step_search_steps < 0 or step_search_proposals <= 0:
        raise ValueError("Step search steps and proposals must be nonnegative")
    per_layer_step_steps = layer_step_search_steps or {}
    per_layer_step_proposals = layer_step_search_proposals or {}
    per_layer_step_slack = layer_step_search_skew_slack or {}
    per_layer_step_replica = layer_step_search_replica_move_fraction or {}
    per_layer_step_cycle = layer_step_search_cycle_move_fraction or {}
    per_layer_step_block_swap = layer_step_search_block_swap_fraction or {}
    per_layer_step_guided_swap = layer_step_search_guided_swap_fraction or {}
    per_layer_step_guided = layer_step_search_guided_replica_fraction or {}
    per_layer_step_exhaustive = layer_step_search_exhaustive_replica_rounds or {}
    per_layer_step_restarts = layer_step_search_restarts or {}
    per_layer_step_load_cap = layer_step_search_assignment_load_cap_ratio or {}
    per_layer_step_source_weight = layer_step_search_source_send_weight or {}
    per_layer_step_objective = layer_step_search_objective or {}
    per_layer_step_method = layer_step_search_method or {}
    per_layer_step_assignment_weight = layer_step_search_assignment_weight or {}
    per_layer_step_fanout_weight = layer_step_search_fanout_weight or {}
    per_layer_step_assignment_penalty = layer_step_search_assignment_skew_penalty or {}
    per_layer_step_assignment_aggregation = (
        layer_step_search_assignment_skew_aggregation or {}
    )
    if step_search_steps and (
        training_step_route_ids is None
        or training_step_route_hashes is None
        or training_step_route_offsets is None
    ):
        raise ValueError("Step search requires sampled training forwards")
    if (source_aware_permutation or source_aware_permutation_layers) and (
        training_step_route_ids is None
        or training_step_route_hashes is None
        or training_step_route_offsets is None
        or training_step_route_sources is None
    ):
        raise ValueError("Source-aware permutation requires source-labeled routes")
    if route_search_mean_weight < 0:
        raise ValueError("Route search mean weight must be nonnegative")
    if hash_search_steps < 0 or hash_search_proposals <= 0:
        raise ValueError("Hash search steps and proposals must be nonnegative")
    if hash_search_mean_weight < 0:
        raise ValueError("Hash search mean weight must be nonnegative")
    if wave_search_steps < 0 or wave_search_proposals <= 0:
        raise ValueError("Wave search steps and proposals must be nonnegative")
    if wave_search_mean_weight < 0 or wave_weight_power < 0:
        raise ValueError("Wave search weights must be nonnegative")
    if replica_partner_dispersion_weight < -1:
        raise ValueError("Replica partner preference must be at least -1")
    if wave_replica_min_tokens < 0:
        raise ValueError("Wave replica minimum tokens must be nonnegative")
    if slot_order not in ("original", "logical", "load_desc"):
        raise ValueError("Unknown physical slot order")
    if layer_slot_order and any(
        order not in ("original", "logical", "load_desc")
        for order in layer_slot_order.values()
    ):
        raise ValueError("Unknown per-layer physical slot order")
    allowed_seeds = (
        "default",
        "spectral",
        "matching",
        "greedy_graph",
        "spectral_kmeans",
        "replica_graph",
        "kl_graph",
        "louvain",
    )
    if placement_seed not in allowed_seeds or any(
        seed not in allowed_seeds for seed in (layer_placement_seed or {}).values()
    ):
        raise ValueError("Unknown placement seed")
    if not 0 <= graph_seed_correlation_weight <= 1:
        raise ValueError("Graph seed correlation weight must be in [0, 1]")
    if spectral_similarity not in ("cosine", "raw", "excess"):
        raise ValueError("Unknown spectral similarity")
    if any(
        similarity not in ("cosine", "raw", "excess")
        for similarity in (layer_spectral_similarity or {}).values()
    ):
        raise ValueError("Unknown layer spectral similarity")
    if spectral_extra_assignment not in ("greedy", "matching"):
        raise ValueError("Unknown spectral extra-copy assignment")
    if (
        matching_group_size not in (2, 4, 8)
        or matching_affinity_weight < 0
        or matching_load_penalty < 0
        or matching_load_cap_ratio <= 0
        or any(value < 0 for value in (layer_matching_load_penalty or {}).values())
    ):
        raise ValueError("Invalid matching group size or affinity weight")
    if (
        anneal_steps < 0
        or anneal_polish_swaps < 0
        or anneal_start_temperature <= 0
        or anneal_final_temperature <= 0
        or not 0 <= anneal_replica_move_probability <= 1
    ):
        raise ValueError("Invalid annealing parameters")
    if wave_search_steps and (
        training_wave_loads is None
        or training_wave_pairs is None
        or training_wave_tokens is None
        or training_wave_loads.ndim != 3
        or training_wave_loads.shape[1:] != (num_layers, num_experts)
        or training_wave_pairs.shape
        != (training_wave_loads.shape[0], num_layers, num_experts, num_experts)
        or training_wave_tokens.shape != (training_wave_loads.shape[0],)
    ):
        raise ValueError("Wave search requires per-wave training statistics")
    if route_search_steps and (
        training_route_ids is None
        or training_route_hashes is None
        or training_route_ids.ndim != 3
        or training_route_ids.shape[1:] != (num_layers, 6)
        or training_route_hashes.shape != (training_route_ids.shape[0],)
    ):
        raise ValueError("Route search requires sampled training token routes")
    if any(
        value < 0
        for value in (
            max_swaps,
            overlap_curvature,
            mean_pressure_weight,
            random_restarts,
            perturb_swaps,
        )
    ):
        raise ValueError("Pressure-search parameters must be nonnegative")

    resolved_layer_ids = list(range(num_layers)) if layer_ids is None else layer_ids
    if (
        len(resolved_layer_ids) != num_layers
        or len(set(resolved_layer_ids)) != num_layers
    ):
        raise ValueError("Layer IDs must uniquely identify each statistics row")
    active_layers = set(active_layer_ids) if active_layer_ids is not None else None
    locality_layers = set(locality_permutation_layers or ())
    if not locality_layers <= set(resolved_layer_ids):
        raise ValueError("Locality permutation layers must be present")
    if active_layers is not None and not active_layers <= set(resolved_layer_ids):
        raise ValueError("Active layer IDs must be present in the statistics")
    per_layer_swaps = {
        int(layer): count for layer, count in (layer_max_swaps or {}).items()
    }
    per_layer_restarts = {
        int(layer): count for layer, count in (layer_random_restarts or {}).items()
    }
    per_layer_curvature = {
        int(layer): value for layer, value in (layer_overlap_curvature or {}).items()
    }
    per_layer_mean_weight = {
        int(layer): value for layer, value in (layer_mean_pressure_weight or {}).items()
    }
    if (
        set(per_layer_swaps)
        | set(per_layer_restarts)
        | set(per_layer_curvature)
        | set(per_layer_mean_weight)
    ) - set(resolved_layer_ids):
        raise ValueError("Layer settings must refer to a statistics row")
    if any(
        value < 0
        for value in (
            *per_layer_swaps.values(),
            *per_layer_restarts.values(),
            *per_layer_curvature.values(),
            *per_layer_mean_weight.values(),
        )
    ):
        raise ValueError("Layer swap, restart, and curvature must be nonnegative")
    replica_layers = (
        set(replica_selection_layers)
        if replica_selection_layers is not None
        else set(resolved_layer_ids)
    )
    if not replica_layers <= set(resolved_layer_ids):
        raise ValueError("Replica selection layers must exist in statistics")
    paired_replica_layers = set(replica_pair_selection_layers or ())
    if not paired_replica_layers <= set(resolved_layer_ids):
        raise ValueError("Paired replica selection layers must exist in statistics")
    if replica_pair_pool_size < num_replicas - num_experts:
        raise ValueError("Replica pair pool is smaller than the redundant budget")
    seed_layers = (
        set(spectral_seed_layers)
        if spectral_seed_layers is not None
        else set(resolved_layer_ids)
    )
    if not seed_layers <= set(resolved_layer_ids):
        raise ValueError("Spectral seed layers must exist in statistics")
    if not {int(layer) for layer in (layer_placement_seed or {})} <= set(
        resolved_layer_ids
    ):
        raise ValueError("Layer placement seeds must exist in statistics")
    if not {int(layer) for layer in (layer_spectral_similarity or {})} <= set(
        resolved_layer_ids
    ):
        raise ValueError("Layer similarities must exist in statistics")
    annealed_layers = (
        set(anneal_layers) if anneal_layers is not None else set(resolved_layer_ids)
    )
    if not annealed_layers <= set(resolved_layer_ids):
        raise ValueError("Annealing layers must exist in statistics")

    initial = (
        DefaultEplbPolicy.rebalance_experts(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_ranks,
            old_global_expert_indices,
        )
        .cpu()
        .numpy()
    )
    default_rows = initial.copy()
    weights = weight.float().cpu().numpy()
    pairs = expert_pair_counts.cpu().numpy()
    wave_loads = (
        training_wave_loads.cpu().numpy() if training_wave_loads is not None else None
    )
    wave_tokens = (
        training_wave_tokens.cpu().numpy() if training_wave_tokens is not None else None
    )
    wave_replica_set = set(wave_replica_layers or ())
    if wave_replica_set and (wave_loads is None or wave_tokens is None):
        raise ValueError("Wave replica selection requires training waves")
    if not wave_replica_set <= set(resolved_layer_ids):
        raise ValueError("Wave replica layers must exist in statistics")
    for layer, layer_id in enumerate(resolved_layer_ids):
        if layer_id in wave_replica_set:
            assert wave_loads is not None and wave_tokens is not None
            initial[layer] = _replicate_by_wave_peak(
                weights[layer],
                wave_loads[:, layer],
                wave_tokens,
                num_replicas,
                num_ranks,
                wave_replica_min_tokens,
            )
    if replica_partner_dispersion_weight:
        for layer, layer_id in enumerate(resolved_layer_ids):
            if layer_id in replica_layers and layer_id not in wave_replica_set:
                initial[layer] = _replicate_by_partner_dispersion(
                    weights[layer],
                    pairs[layer],
                    num_replicas,
                    num_ranks,
                    replica_partner_dispersion_weight,
                )
    for layer, layer_id in enumerate(resolved_layer_ids):
        if layer_id in paired_replica_layers:
            initial[layer] = _replicate_correlated_pairs(
                initial[layer],
                weights[layer],
                pairs[layer],
                num_replicas,
                num_ranks,
                replica_pair_pool_size,
                replica_pair_score,
                (layer_replica_pair_extras or {}).get(
                    str(layer_id), replica_pair_extras
                ),
            )
    if placement_seed != "default" or layer_placement_seed:
        for layer, layer_id in enumerate(resolved_layer_ids):
            if layer_id in seed_layers:
                layer_seed = (layer_placement_seed or {}).get(
                    str(layer_id), placement_seed
                )
                layer_similarity = (layer_spectral_similarity or {}).get(
                    str(layer_id), spectral_similarity
                )
                if layer_seed == "spectral":
                    initial[layer] = _spectral_seed(
                        initial[layer],
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        layer_similarity,
                        spectral_extra_assignment,
                    )
                elif layer_seed == "matching":
                    initial[layer] = _matching_seed(
                        initial[layer],
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        matching_group_size,
                        matching_affinity_weight,
                        layer_similarity,
                        (layer_matching_load_penalty or {}).get(
                            str(layer_id), matching_load_penalty
                        ),
                        matching_load_cap_ratio,
                    )
                elif layer_seed == "greedy_graph":
                    from vllm.distributed.eplb.policy.graph_seed import (
                        greedy_graph_seed,
                    )

                    initial[layer] = greedy_graph_seed(
                        initial[layer],
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        graph_seed_correlation_weight,
                    )
                elif layer_seed == "spectral_kmeans":
                    from vllm.distributed.eplb.policy.graph_seed import (
                        spectral_kmeans_seed,
                    )

                    initial[layer] = spectral_kmeans_seed(
                        initial[layer],
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        similarity=layer_similarity,
                        restarts=spectral_kmeans_restarts,
                        seed=random_seed + layer * 1009,
                        dimensions=spectral_kmeans_dimensions,
                        selection=spectral_kmeans_selection,
                        training_routes=(
                            training_step_route_ids[:, layer, :].cpu().numpy()
                            if spectral_kmeans_selection == "exact"
                            and training_step_route_ids is not None
                            else None
                        ),
                        training_hashes=(
                            training_step_route_hashes.cpu().numpy()
                            if spectral_kmeans_selection == "exact"
                            and training_step_route_hashes is not None
                            else None
                        ),
                        training_offsets=(
                            training_step_route_offsets.cpu().numpy()
                            if spectral_kmeans_selection == "exact"
                            and training_step_route_offsets is not None
                            else None
                        ),
                        load_penalty=(layer_spectral_kmeans_load_penalty or {}).get(
                            str(layer_id), spectral_kmeans_load_penalty
                        ),
                        replica_adjustment=spectral_kmeans_replica_adjustment,
                    )
                elif layer_seed == "replica_graph":
                    from vllm.distributed.eplb.policy.graph_seed import (
                        replica_graph_seed,
                    )

                    initial[layer] = replica_graph_seed(
                        initial[layer],
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        restarts=spectral_kmeans_restarts,
                        seed=random_seed + layer * 1009,
                    )
                elif layer_seed == "kl_graph":
                    from vllm.distributed.eplb.policy.graph_seed import kl_graph_seed

                    initial[layer] = kl_graph_seed(
                        initial[layer],
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        similarity=layer_similarity,
                        neighbors=kl_neighbors,
                        restarts=kl_restarts,
                        seed=random_seed + layer * 1009,
                    )
                elif layer_seed == "louvain":
                    from vllm.distributed.eplb.policy.graph_seed import louvain_seed

                    initial[layer] = louvain_seed(
                        initial[layer],
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        resolution=louvain_resolution,
                        neighbors=louvain_neighbors,
                        affinity_weight=louvain_affinity_weight,
                        seed=random_seed + layer * 1009,
                    )
    if hypergraph_seed_layers:
        from vllm.distributed.eplb.policy.graph_seed import hypergraph_seed

        if training_step_route_ids is None:
            raise ValueError("Hypergraph seed requires sampled training routes")
        if hypergraph_wave_load_cap_ratio is not None and wave_loads is None:
            raise ValueError("Hypergraph wave cap requires training wave loads")
        if hypergraph_wave_min_tokens < 0:
            raise ValueError("Hypergraph wave minimum tokens must be nonnegative")
        if hypergraph_wave_load_cap_ratio is not None:
            assert wave_tokens is not None
            selected_waves = wave_tokens >= hypergraph_wave_min_tokens
            if not np.any(selected_waves):
                raise ValueError("No training waves satisfy the minimum token count")
        if not set(hypergraph_seed_layers) <= set(resolved_layer_ids):
            raise ValueError("Hypergraph seed layers must exist in statistics")
        sampled = training_step_route_ids.cpu().numpy()
        for layer, layer_id in enumerate(resolved_layer_ids):
            if layer_id in hypergraph_seed_layers:
                initial[layer] = hypergraph_seed(
                    initial[layer],
                    weights[layer],
                    pairs[layer],
                    sampled[:, layer, :],
                    num_ranks,
                    hypergraph_correlation_weight,
                    hypergraph_arrival_cap_ratio,
                    hypergraph_assignment_cap_ratio,
                    hypergraph_seed_objective,
                    hypergraph_load_weight,
                    wave_loads[selected_waves, layer]
                    if hypergraph_wave_load_cap_ratio is not None
                    else None,
                    hypergraph_wave_load_cap_ratio,
                )
    source_loads = (
        source_expert_loads.cpu().numpy() if source_expert_loads is not None else None
    )
    route_ids = (
        training_route_ids.cpu().numpy() if training_route_ids is not None else None
    )
    step_route_ids = (
        training_step_route_ids.cpu().numpy()
        if training_step_route_ids is not None
        else None
    )
    step_route_hashes = (
        training_step_route_hashes.cpu().numpy()
        if training_step_route_hashes is not None
        else None
    )
    step_route_offsets = (
        training_step_route_offsets.cpu().numpy()
        if training_step_route_offsets is not None
        else None
    )
    step_route_sources = (
        training_step_route_sources.cpu().numpy()
        if training_step_route_sources is not None
        else None
    )
    route_hashes = (
        training_route_hashes.cpu().numpy()
        if training_route_hashes is not None
        else None
    )
    wave_pairs = (
        training_wave_pairs.cpu().numpy() if training_wave_pairs is not None else None
    )
    route_layers = (
        set(route_search_layers) if route_search_layers else set(resolved_layer_ids)
    )
    step_layers = (
        set(step_search_layers) if step_search_layers else set(resolved_layer_ids)
    )
    source_permutation_layers = (
        set(resolved_layer_ids)
        if source_aware_permutation
        else set(source_aware_permutation_layers or ())
    )
    hash_layers = (
        set(hash_search_layers) if hash_search_layers else set(resolved_layer_ids)
    )
    wave_layers = (
        set(wave_search_layers) if wave_search_layers else set(resolved_layer_ids)
    )
    if not (
        route_layers
        | step_layers
        | hash_layers
        | wave_layers
        | source_permutation_layers
    ) <= set(resolved_layer_ids):
        raise ValueError("Search layers must be present in the statistics")
    result_rows = []
    for layer in range(num_layers):
        layer_id = resolved_layer_ids[layer]
        layer_curvature = per_layer_curvature.get(layer_id, overlap_curvature)
        layer_mean_weight = per_layer_mean_weight.get(layer_id, mean_pressure_weight)
        if active_layers is not None and layer_id not in active_layers:
            row = initial[layer]
            if locality_permutation or layer_id in locality_layers:
                assert source_loads is not None
                row = _permute_ranks_for_locality(
                    row, source_loads[layer], num_ranks, num_experts
                )
            if route_search_steps and layer_id in route_layers:
                assert route_ids is not None and route_hashes is not None
                row = refine_by_sampled_routes(
                    row,
                    route_ids[:, layer, :],
                    route_hashes,
                    num_ranks,
                    steps=route_search_steps,
                    proposals=route_search_proposals,
                    mean_weight=route_search_mean_weight,
                    seed=random_seed + layer * 1009,
                )
            if step_search_steps and layer_id in step_layers:
                from vllm.distributed.eplb.policy.route_search import (
                    refine_by_training_steps,
                )

                assert step_route_ids is not None
                assert step_route_hashes is not None
                assert step_route_offsets is not None
                row = refine_by_training_steps(
                    row,
                    step_route_ids[:, layer, :],
                    step_route_hashes,
                    step_route_offsets,
                    step_route_sources,
                    default_rows[layer],
                    num_ranks,
                    steps=per_layer_step_steps.get(str(layer_id), step_search_steps),
                    proposals=per_layer_step_proposals.get(
                        str(layer_id), step_search_proposals
                    ),
                    skew_slack=per_layer_step_slack.get(
                        str(layer_id), step_search_skew_slack
                    ),
                    skew_penalty=step_search_skew_penalty,
                    assignment_skew_slack=step_search_assignment_skew_slack,
                    assignment_skew_penalty=per_layer_step_assignment_penalty.get(
                        str(layer_id), step_search_assignment_skew_penalty
                    ),
                    assignment_skew_aggregation=per_layer_step_assignment_aggregation.get(
                        str(layer_id), step_search_assignment_skew_aggregation
                    ),
                    objective=per_layer_step_objective.get(
                        str(layer_id), step_search_objective
                    ),
                    assignment_weight=per_layer_step_assignment_weight.get(
                        str(layer_id), step_search_assignment_weight
                    ),
                    fanout_weight=per_layer_step_fanout_weight.get(
                        str(layer_id), step_search_fanout_weight
                    ),
                    source_send_weight=per_layer_step_source_weight.get(
                        str(layer_id), step_search_source_send_weight
                    ),
                    search_method=per_layer_step_method.get(
                        str(layer_id), step_search_method
                    ),
                    anneal_start_temperature=step_search_anneal_start,
                    anneal_final_temperature=step_search_anneal_final,
                    anneal_polish_steps=step_search_anneal_polish_steps,
                    replica_move_fraction=per_layer_step_replica.get(
                        str(layer_id), step_search_replica_move_fraction
                    ),
                    cycle_move_fraction=per_layer_step_cycle.get(
                        str(layer_id), step_search_cycle_move_fraction
                    ),
                    block_swap_fraction=per_layer_step_block_swap.get(
                        str(layer_id), step_search_block_swap_fraction
                    ),
                    guided_swap_fraction=per_layer_step_guided_swap.get(
                        str(layer_id), step_search_guided_swap_fraction
                    ),
                    guided_replica_fraction=per_layer_step_guided.get(
                        str(layer_id), step_search_guided_replica_fraction
                    ),
                    exhaustive_replica_rounds=per_layer_step_exhaustive.get(
                        str(layer_id), step_search_exhaustive_replica_rounds
                    ),
                    search_restarts=per_layer_step_restarts.get(
                        str(layer_id), step_search_restarts
                    ),
                    assignment_load_cap_ratio=per_layer_step_load_cap.get(
                        str(layer_id), step_search_assignment_load_cap_ratio
                    ),
                    assignment_load_cap_penalty=step_search_assignment_load_cap_penalty,
                    replica_pair_counts=pairs[layer],
                    replica_candidate_loads=weights[layer],
                    seed=random_seed + layer * 1009 + 61,
                )
            if hash_search_steps and layer_id in hash_layers:
                row = refine_by_hash_aware_pairs(
                    row,
                    weights[layer],
                    pairs[layer],
                    num_ranks,
                    steps=hash_search_steps,
                    proposals=hash_search_proposals,
                    overlap_curvature=layer_curvature,
                    mean_weight=hash_search_mean_weight,
                    seed=random_seed + layer * 1009 + 17,
                )
            if wave_search_steps and layer_id in wave_layers:
                assert wave_loads is not None and wave_pairs is not None
                assert wave_tokens is not None
                row = refine_by_wave_pressure(
                    row,
                    wave_loads[:, layer],
                    wave_pairs[:, layer],
                    wave_tokens,
                    num_ranks,
                    steps=wave_search_steps,
                    proposals=wave_search_proposals,
                    overlap_curvature=layer_curvature,
                    mean_weight=wave_search_mean_weight,
                    wave_weight_power=wave_weight_power,
                    seed=random_seed + layer * 1009 + 31,
                )
            if anneal_steps and layer_id in annealed_layers:
                row = _anneal_placement(
                    row,
                    weights[layer],
                    pairs[layer],
                    num_ranks,
                    steps=anneal_steps,
                    start_temperature=anneal_start_temperature,
                    final_temperature=anneal_final_temperature,
                    replica_move_probability=anneal_replica_move_probability,
                    polish_swaps=anneal_polish_swaps,
                    overlap_curvature=layer_curvature,
                    mean_pressure_weight=layer_mean_weight,
                    seed=random_seed + layer * 1009 + 43,
                )
            if layer_id in source_permutation_layers:
                from vllm.distributed.eplb.policy.route_search import (
                    source_aware_rank_permutation,
                )

                assert step_route_ids is not None
                assert step_route_hashes is not None
                assert step_route_offsets is not None
                assert step_route_sources is not None
                row = source_aware_rank_permutation(
                    row,
                    step_route_ids[:, layer, :],
                    step_route_hashes,
                    step_route_sources,
                    step_route_offsets,
                    num_ranks,
                    fanout_weight=source_aware_fanout_weight,
                )
            result_rows.append(row)
            continue
        layer_swaps = per_layer_swaps.get(layer_id, max_swaps)
        layer_restarts = per_layer_restarts.get(layer_id, random_restarts)
        row = _minimize_destination_pressure_swaps(
            initial[layer],
            weights[layer],
            pairs[layer],
            num_ranks,
            layer_swaps,
            layer_curvature,
            layer_mean_weight,
        )
        best_score = _destination_pressure_score(
            row,
            weights[layer],
            pairs[layer],
            num_ranks,
            layer_curvature,
            layer_mean_weight,
        )
        rng = np.random.default_rng(random_seed + layer * 1009)
        for _ in range(layer_restarts):
            perturbed = _perturb_placement(row, num_ranks, perturb_swaps, rng)
            candidate = _minimize_destination_pressure_swaps(
                perturbed,
                weights[layer],
                pairs[layer],
                num_ranks,
                layer_swaps,
                layer_curvature,
                layer_mean_weight,
            )
            score = _destination_pressure_score(
                candidate,
                weights[layer],
                pairs[layer],
                num_ranks,
                layer_curvature,
                layer_mean_weight,
            )
            if score < best_score:
                row = candidate
                best_score = score
        if locality_permutation or layer_id in locality_layers:
            assert source_loads is not None
            row = _permute_ranks_for_locality(
                row, source_loads[layer], num_ranks, num_experts
            )
        if route_search_steps and layer_id in route_layers:
            assert route_ids is not None and route_hashes is not None
            row = refine_by_sampled_routes(
                row,
                route_ids[:, layer, :],
                route_hashes,
                num_ranks,
                steps=route_search_steps,
                proposals=route_search_proposals,
                mean_weight=route_search_mean_weight,
                seed=random_seed + layer * 1009,
            )
        if step_search_steps and layer_id in step_layers:
            from vllm.distributed.eplb.policy.route_search import (
                refine_by_training_steps,
            )

            assert step_route_ids is not None
            assert step_route_hashes is not None
            assert step_route_offsets is not None
            row = refine_by_training_steps(
                row,
                step_route_ids[:, layer, :],
                step_route_hashes,
                step_route_offsets,
                step_route_sources,
                default_rows[layer],
                num_ranks,
                steps=per_layer_step_steps.get(str(layer_id), step_search_steps),
                proposals=per_layer_step_proposals.get(
                    str(layer_id), step_search_proposals
                ),
                skew_slack=per_layer_step_slack.get(
                    str(layer_id), step_search_skew_slack
                ),
                skew_penalty=step_search_skew_penalty,
                assignment_skew_slack=step_search_assignment_skew_slack,
                assignment_skew_penalty=per_layer_step_assignment_penalty.get(
                    str(layer_id), step_search_assignment_skew_penalty
                ),
                assignment_skew_aggregation=per_layer_step_assignment_aggregation.get(
                    str(layer_id), step_search_assignment_skew_aggregation
                ),
                objective=per_layer_step_objective.get(
                    str(layer_id), step_search_objective
                ),
                assignment_weight=per_layer_step_assignment_weight.get(
                    str(layer_id), step_search_assignment_weight
                ),
                fanout_weight=per_layer_step_fanout_weight.get(
                    str(layer_id), step_search_fanout_weight
                ),
                source_send_weight=per_layer_step_source_weight.get(
                    str(layer_id), step_search_source_send_weight
                ),
                search_method=per_layer_step_method.get(
                    str(layer_id), step_search_method
                ),
                anneal_start_temperature=step_search_anneal_start,
                anneal_final_temperature=step_search_anneal_final,
                anneal_polish_steps=step_search_anneal_polish_steps,
                replica_move_fraction=per_layer_step_replica.get(
                    str(layer_id), step_search_replica_move_fraction
                ),
                cycle_move_fraction=per_layer_step_cycle.get(
                    str(layer_id), step_search_cycle_move_fraction
                ),
                block_swap_fraction=per_layer_step_block_swap.get(
                    str(layer_id), step_search_block_swap_fraction
                ),
                guided_swap_fraction=per_layer_step_guided_swap.get(
                    str(layer_id), step_search_guided_swap_fraction
                ),
                guided_replica_fraction=per_layer_step_guided.get(
                    str(layer_id), step_search_guided_replica_fraction
                ),
                exhaustive_replica_rounds=per_layer_step_exhaustive.get(
                    str(layer_id), step_search_exhaustive_replica_rounds
                ),
                search_restarts=per_layer_step_restarts.get(
                    str(layer_id), step_search_restarts
                ),
                assignment_load_cap_ratio=per_layer_step_load_cap.get(
                    str(layer_id), step_search_assignment_load_cap_ratio
                ),
                assignment_load_cap_penalty=step_search_assignment_load_cap_penalty,
                replica_pair_counts=pairs[layer],
                replica_candidate_loads=weights[layer],
                seed=random_seed + layer * 1009 + 61,
            )
        if hash_search_steps and layer_id in hash_layers:
            row = refine_by_hash_aware_pairs(
                row,
                weights[layer],
                pairs[layer],
                num_ranks,
                steps=hash_search_steps,
                proposals=hash_search_proposals,
                overlap_curvature=layer_curvature,
                mean_weight=hash_search_mean_weight,
                seed=random_seed + layer * 1009 + 17,
            )
        if wave_search_steps and layer_id in wave_layers:
            assert wave_loads is not None and wave_pairs is not None
            assert wave_tokens is not None
            row = refine_by_wave_pressure(
                row,
                wave_loads[:, layer],
                wave_pairs[:, layer],
                wave_tokens,
                num_ranks,
                steps=wave_search_steps,
                proposals=wave_search_proposals,
                overlap_curvature=layer_curvature,
                mean_weight=wave_search_mean_weight,
                wave_weight_power=wave_weight_power,
                seed=random_seed + layer * 1009 + 31,
            )
        if anneal_steps and layer_id in annealed_layers:
            row = _anneal_placement(
                row,
                weights[layer],
                pairs[layer],
                num_ranks,
                steps=anneal_steps,
                start_temperature=anneal_start_temperature,
                final_temperature=anneal_final_temperature,
                replica_move_probability=anneal_replica_move_probability,
                polish_swaps=anneal_polish_swaps,
                overlap_curvature=layer_curvature,
                mean_pressure_weight=layer_mean_weight,
                seed=random_seed + layer * 1009 + 43,
            )
        if layer_id in source_permutation_layers:
            from vllm.distributed.eplb.policy.route_search import (
                source_aware_rank_permutation,
            )

            assert step_route_ids is not None
            assert step_route_hashes is not None
            assert step_route_offsets is not None
            assert step_route_sources is not None
            row = source_aware_rank_permutation(
                row,
                step_route_ids[:, layer, :],
                step_route_hashes,
                step_route_sources,
                step_route_offsets,
                num_ranks,
                fanout_weight=source_aware_fanout_weight,
            )
        result_rows.append(row)

    result = np.stack(result_rows).reshape(num_layers, num_ranks, -1)
    per_layer_slot_order = layer_slot_order or {}
    for layer in range(num_layers):
        selected_order = per_layer_slot_order.get(
            str(resolved_layer_ids[layer]), slot_order
        )
        if selected_order == "logical":
            result[layer] = np.sort(result[layer], axis=1)
        elif selected_order == "load_desc":
            for rank in range(num_ranks):
                row = result[layer, rank]
                result[layer, rank] = row[
                    np.argsort(-weights[layer, row], kind="stable")
                ]
    return torch.from_numpy(result.reshape(num_layers, num_replicas))
