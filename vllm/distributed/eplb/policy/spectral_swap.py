# SPDX-License-Identifier: Apache-2.0
"""Spectral expert placement with load-aware correlation swaps."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
from numba import njit

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from .default import DefaultEplbPolicy
from .graph_seed import spectral_kmeans_seed


@njit(cache=True, nogil=True)
def _build_copy_correlation(
    placement: np.ndarray, weight: np.ndarray, lower_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    num_experts = len(weight)
    num_copies = len(placement)
    copies = np.zeros(num_experts, dtype=np.int32)
    ordinals = np.empty(num_copies, dtype=np.int32)
    for copy in range(num_copies):
        expert = placement[copy]
        ordinals[copy] = copies[expert]
        copies[expert] += 1
    for expert in range(num_experts):
        if copies[expert] == 0:
            raise ValueError("Every logical expert needs a physical copy")

    correlation = np.zeros((num_copies, num_copies), dtype=np.float64)
    copy_load = np.empty(num_copies, dtype=np.float64)
    for first in range(num_copies):
        first_expert = placement[first]
        first_count = copies[first_expert]
        copy_load[first] = float(weight[first_expert]) / first_count
        for second in range(first + 1, num_copies):
            second_expert = placement[second]
            if first_expert == second_expert:
                continue
            second_count = copies[second_expert]
            left = first_count
            right = second_count
            while right:
                left, right = right, left % right
            gcd = left
            if (ordinals[first] - ordinals[second]) % gcd:
                continue
            larger = max(first_expert, second_expert)
            smaller = min(first_expert, second_expert)
            pair_count = lower_pairs[larger, smaller]
            value = float(pair_count) / (first_count * second_count // gcd)
            correlation[first, second] = value
            correlation[second, first] = value
    return correlation, copy_load


def _copy_correlation(
    placement: np.ndarray, weight: np.ndarray, lower_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate hash-aligned co-routing and load for each physical copy."""
    return _build_copy_correlation(placement, weight, lower_pairs)


def _colocated_sum(matrix: np.ndarray, slots_per_rank: int) -> float:
    return float(
        sum(
            np.triu(
                matrix[start : start + slots_per_rank, start : start + slots_per_rank],
                k=1,
            ).sum()
            for start in range(0, len(matrix), slots_per_rank)
        )
    )


def _check_distinct_rank_copies(placement: np.ndarray, num_ranks: int) -> None:
    if len(placement) % num_ranks:
        raise ValueError("Physical copies must divide evenly among ranks")
    for row in placement.reshape(num_ranks, -1):
        if len(np.unique(row)) != len(row):
            raise ValueError("A rank contains two copies of the same expert")


def _normalization_scales(
    placement: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
) -> tuple[float, float]:
    slots = len(placement) // num_ranks
    copies = np.bincount(placement, minlength=len(weight))
    if np.any(copies == 0):
        raise ValueError("Every logical expert needs a physical copy")
    order = np.argsort(placement, kind="stable")
    ordinals = np.empty(len(placement), dtype=np.int32)
    ordinals[order] = np.arange(len(placement)) - np.repeat(
        np.cumsum(copies) - copies, copies
    )

    rows = placement.reshape(num_ranks, slots)
    ordinal_rows = ordinals.reshape(num_ranks, slots)
    first_slot, second_slot = np.triu_indices(slots, k=1)
    first = rows[:, first_slot]
    second = rows[:, second_slot]
    first_count = copies[first]
    second_count = copies[second]
    gcd = np.gcd(first_count, second_count)
    lcm = first_count * second_count // gcd
    aligned = (ordinal_rows[:, first_slot] - ordinal_rows[:, second_slot]) % gcd == 0
    pair_counts = lower_pairs[np.maximum(first, second), np.minimum(first, second)]
    correlation = np.where(first != second, pair_counts * aligned / lcm, 0.0)

    copy_load = (weight[placement] / copies[placement]).reshape(num_ranks, slots)
    rank_load = copy_load.sum(axis=1)
    load_pairs = (np.square(rank_load).sum() - np.square(copy_load).sum()) / 2
    return (
        max(float(correlation.sum()), 1.0),
        max(float(load_pairs), 1.0),
    )


def _placement_score(
    placement: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    alpha: float,
    scales: tuple[float, float],
) -> float:
    correlation, copy_load = _copy_correlation(placement, weight, lower_pairs)
    slots = len(placement) // num_ranks
    return (
        alpha * _colocated_sum(correlation, slots) / scales[0]
        - (1 - alpha)
        * _colocated_sum(np.outer(copy_load, copy_load), slots)
        / scales[1]
    )


def _has_replica_matching(
    extras: list[int], base_rank: np.ndarray, num_ranks: int
) -> bool:
    assigned = [-1] * num_ranks

    def augment(index: int, seen: list[bool]) -> bool:
        for rank in range(num_ranks):
            if rank == base_rank[extras[index]] or seen[rank]:
                continue
            seen[rank] = True
            if assigned[rank] < 0 or augment(assigned[rank], seen):
                assigned[rank] = index
                return True
        return False

    return all(augment(index, [False] * num_ranks) for index in range(len(extras)))


def _choose_replicas(base: np.ndarray, weight: np.ndarray, num_ranks: int) -> list[int]:
    """Split the busiest copy while preserving a distinct-rank assignment."""
    base_rank = np.empty(len(weight), dtype=np.int32)
    for rank, row in enumerate(base.reshape(num_ranks, -1)):
        base_rank[row] = rank
    copies = np.ones(len(weight), dtype=np.int32)
    extras: list[int] = []
    for _ in range(num_ranks):
        for expert in np.argsort(-weight / copies, kind="stable"):
            expert = int(expert)
            if copies[expert] >= num_ranks:
                continue
            if _has_replica_matching(extras + [expert], base_rank, num_ranks):
                extras.append(expert)
                copies[expert] += 1
                break
        else:
            raise ValueError("No distinct-rank assignment for redundant experts")
    return extras


def _plan_replicas(weight: np.ndarray, num_ranks: int) -> tuple[list[int], np.ndarray]:
    """Choose extra copies from load before assigning their ranks."""
    copies = np.ones(len(weight), dtype=np.int32)
    extras = []
    for _ in range(num_ranks):
        scores = np.where(copies < num_ranks, weight / copies, -np.inf)
        expert = int(np.argmax(scores))
        extras.append(expert)
        copies[expert] += 1
    return extras, copies


def _greedy_legal_assignment(
    valid: np.ndarray, scores: np.ndarray | None = None
) -> tuple[int, ...]:
    """Choose ranks in score order, backtracking only to keep the map legal."""
    num_ranks = len(valid)
    assignment = [-1] * num_ranks

    def assign(index: int, used: int) -> bool:
        if index == num_ranks:
            return True
        candidates = (
            range(num_ranks)
            if scores is None
            else np.argsort(-scores[index], kind="stable")
        )
        for rank in candidates:
            rank = int(rank)
            if not valid[index, rank] or used & (1 << rank):
                continue
            assignment[index] = rank
            if assign(index + 1, used | (1 << rank)):
                return True
        assignment[index] = -1
        return False

    if not assign(0, 0):
        raise ValueError("No distinct-rank assignment for redundant experts")
    return tuple(assignment)


def place_spectral_replicas(
    base: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    alpha: float,
    planned_extras: list[int] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Place one extra copy per rank using the swap objective as a proxy."""
    if len(base) != len(weight) or len(base) % num_ranks:
        raise ValueError("Expected one base copy of each expert on equal-sized ranks")
    if not np.array_equal(np.sort(base), np.arange(len(weight))):
        raise ValueError("Spectral base must contain every logical expert once")
    _check_distinct_rank_copies(base, num_ranks)
    rows = base.reshape(num_ranks, -1)
    extras = (
        _choose_replicas(base, weight, num_ranks)
        if planned_extras is None
        else list(planned_extras)
    )
    if len(extras) != num_ranks or any(
        expert < 0 or expert >= len(weight) for expert in extras
    ):
        raise ValueError("Expected one valid redundant expert per rank")
    copies = np.ones(len(weight), dtype=np.int32)
    np.add.at(copies, extras, 1)
    extras.sort(key=lambda expert: (-float(weight[expert] / copies[expert]), expert))
    extra_ids = np.asarray(extras, dtype=np.int64)
    valid = ~(extra_ids[:, None, None] == rows[None, :, :]).any(axis=2)

    def build(assignment: tuple[int, ...]) -> np.ndarray:
        by_rank = np.empty(num_ranks, dtype=np.int64)
        by_rank[list(assignment)] = extras
        return np.column_stack((rows, by_rank)).reshape(-1)

    reference = _greedy_legal_assignment(valid)
    scales = _normalization_scales(build(reference), weight, lower_pairs, num_ranks)

    copy_load = weight / copies
    rank_load = copy_load[rows].sum(axis=1)
    first = extra_ids[:, None, None]
    second = rows[None, :, :]
    pair_counts = lower_pairs[np.maximum(first, second), np.minimum(first, second)]
    affinity = (pair_counts / (copies[first] * copies[second])).sum(axis=2)
    load_cost = copy_load[extra_ids, None] * rank_load[None, :]
    scores = alpha * affinity / scales[0] - (1 - alpha) * load_cost / scales[1]
    scores[~valid] = -np.inf
    placement = build(_greedy_legal_assignment(valid, scores))
    _check_distinct_rank_copies(placement, num_ranks)
    return placement, scales


def _ordinal_bounds(
    positions: np.ndarray, replicas: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(positions), -1, dtype=np.int32)
    upper = np.full(len(positions), len(positions), dtype=np.int32)
    for group in replicas:
        for ordinal, copy in enumerate(group):
            if ordinal:
                lower[copy] = positions[group[ordinal - 1]]
            if ordinal + 1 < len(group):
                upper[copy] = positions[group[ordinal + 1]]
    return lower, upper


@njit(cache=True, nogil=True)
def _swap_copy_ids(
    score: np.ndarray,
    logical: np.ndarray,
    previous_copy: np.ndarray,
    next_copy: np.ndarray,
    rank_sums: np.ndarray,
    num_ranks: int,
    slots: int,
    max_swaps: int,
) -> np.ndarray:
    """Search legal swaps in rank, slot order to retain deterministic ties."""
    num_copies = len(logical)
    ids = np.arange(num_copies, dtype=np.int32)
    positions = ids.copy()
    membership = np.zeros((num_ranks, len(score)), dtype=np.bool_)
    for rank in range(num_ranks):
        for slot in range(slots):
            membership[rank, logical[rank * slots + slot]] = True

    for _ in range(max_swaps):
        logical_at_position = np.empty(num_copies, dtype=np.int32)
        previous_position = np.empty(num_copies, dtype=np.int32)
        next_position = np.empty(num_copies, dtype=np.int32)
        for position in range(num_copies):
            copy = ids[position]
            logical_at_position[position] = logical[copy]
            previous = previous_copy[copy]
            previous_position[position] = (
                positions[previous] if previous >= 0 else -1
            )
            following = next_copy[copy]
            next_position[position] = (
                positions[following] if following >= 0 else num_copies
            )
        best_gain = 1e-12
        best_first = -1
        best_second = -1
        best_first_rank = -1
        best_second_rank = -1
        for first_rank in range(num_ranks):
            for second_rank in range(first_rank + 1, num_ranks):
                for first_slot in range(slots):
                    first_pos = first_rank * slots + first_slot
                    first_copy = ids[first_pos]
                    first_logical = logical_at_position[first_pos]
                    if membership[second_rank, first_logical]:
                        continue
                    previous_first_pos = previous_position[first_pos]
                    next_first_pos = next_position[first_pos]
                    first_gain = (
                        rank_sums[first_copy, second_rank]
                        - rank_sums[first_copy, first_rank]
                    )
                    for second_slot in range(slots):
                        second_pos = second_rank * slots + second_slot
                        second_copy = ids[second_pos]
                        second_logical = logical_at_position[second_pos]
                        if (
                            first_logical == second_logical
                            or membership[first_rank, second_logical]
                        ):
                            continue
                        if previous_first_pos >= second_pos:
                            continue
                        if second_pos >= next_first_pos:
                            continue
                        if previous_position[second_pos] >= first_pos:
                            continue
                        if first_pos >= next_position[second_pos]:
                            continue
                        gain = (
                            first_gain
                            + rank_sums[second_copy, first_rank]
                            - rank_sums[second_copy, second_rank]
                            - 2 * score[first_copy, second_copy]
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_first = first_pos
                            best_second = second_pos
                            best_first_rank = first_rank
                            best_second_rank = second_rank
        if best_first < 0:
            break

        first_copy = ids[best_first]
        second_copy = ids[best_second]
        first_logical = logical[first_copy]
        second_logical = logical[second_copy]
        ids[best_first], ids[best_second] = second_copy, first_copy
        positions[first_copy], positions[second_copy] = best_second, best_first
        membership[best_first_rank, first_logical] = False
        membership[best_first_rank, second_logical] = True
        membership[best_second_rank, second_logical] = False
        membership[best_second_rank, first_logical] = True
        for copy in range(num_copies):
            rank_sums[copy, best_first_rank] += (
                score[copy, second_copy] - score[copy, first_copy]
            )
            rank_sums[copy, best_second_rank] += (
                score[copy, first_copy] - score[copy, second_copy]
            )
    return ids


def improve_spectral_by_swapping(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    alpha: float,
    max_swaps: int,
    scales: tuple[float, float] | None = None,
) -> np.ndarray:
    """Maximize normalized co-location minus load concentration.

    Swaps preserve rank capacity, replica counts, and each copy's hash ordinal.
    The two terms are scaled by their values at the spectral seed, so alpha
    trades fractional affinity improvement against fractional load reduction.
    """
    if not 0 <= alpha <= 1 or max_swaps < 0:
        raise ValueError("alpha must be in [0, 1] and max_swaps nonnegative")
    _check_distinct_rank_copies(initial, num_ranks)
    if max_swaps == 0 or num_ranks < 2:
        return initial.copy()

    slots = len(initial) // num_ranks
    correlation, copy_load = _copy_correlation(initial, weight, lower_pairs)
    load_pairs = np.outer(copy_load, copy_load)
    np.fill_diagonal(load_pairs, 0.0)
    if scales is None:
        scales = _normalization_scales(initial, weight, lower_pairs, num_ranks)
    correlation_scale, load_scale = scales
    score = alpha * correlation / correlation_scale
    score -= (1 - alpha) * load_pairs / load_scale

    logical = initial.copy()
    previous_copy = np.full(len(initial), -1, dtype=np.int32)
    next_copy = np.full(len(initial), -1, dtype=np.int32)
    last_copy = np.full(len(weight), -1, dtype=np.int32)
    for copy, expert in enumerate(logical):
        prior = last_copy[expert]
        if prior >= 0:
            previous_copy[copy] = prior
            next_copy[prior] = copy
        last_copy[expert] = copy
    ids = np.arange(len(initial), dtype=np.int32)
    rank_sums = score[:, ids.reshape(num_ranks, slots)].sum(axis=2)
    ids = _swap_copy_ids(
        score,
        logical,
        previous_copy,
        next_copy,
        rank_sums,
        num_ranks,
        slots,
        max_swaps,
    )

    result = logical[ids]
    _check_distinct_rank_copies(result, num_ranks)
    return result


class SpectralSwapEplbPolicy(DefaultEplbPolicy):
    """Use windowed pair counts to refine a spectral placement."""

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
        alpha: float = 0.5,
        max_swaps: int = 64,
        spectral_restarts: int = 8,
        seed: int = 0,
        assignment_load_weight: float | None = None,
    ) -> torch.Tensor:
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
        if num_nodes != 1:
            raise ValueError("Spectral swaps currently support one node")
        if num_experts % num_ranks or num_replicas not in (
            num_experts,
            num_experts + num_ranks,
        ):
            raise ValueError("Spectral swaps need zero or one redundant slot per rank")
        if expert_pair_counts.shape != (num_layers, num_experts, num_experts):
            raise ValueError("Invalid expert-pair count shape")
        if not 0 <= alpha <= 1 or spectral_restarts < 1 or max_swaps < 0:
            raise ValueError("Invalid spectral swap parameters")
        if assignment_load_weight is not None and (
            assignment_load_weight < 0 or max_swaps != 0
        ):
            raise ValueError("Load-aware seed requires no swaps and nonnegative weight")

        weights = weight.float().cpu().numpy()
        pairs = expert_pair_counts.cpu().numpy()
        has_pairs = np.any(pairs, axis=(1, 2))
        default = (
            DefaultEplbPolicy.rebalance_experts(
                weight, num_replicas, num_groups, num_nodes, num_ranks
            ).numpy()
            if not np.all(has_pairs)
            else None
        )

        def place_layer(layer: int) -> np.ndarray:
            if not has_pairs[layer]:
                assert default is not None
                return default[layer].copy()
            planned_extras = None
            planned_copies = None
            if assignment_load_weight is not None and num_replicas > num_experts:
                planned_extras, planned_copies = _plan_replicas(
                    weights[layer], num_ranks
                )
            try:
                spectral = spectral_kmeans_seed(
                    np.arange(num_experts, dtype=np.int64),
                    weights[layer],
                    pairs[layer],
                    num_ranks,
                    similarity="raw",
                    restarts=spectral_restarts,
                    seed=seed + layer * 1009,
                    load_penalty=0.0,
                    assignment_load_weight=assignment_load_weight,
                    planned_copies=planned_copies,
                )
            except ValueError as exc:
                if str(exc) != "No legal spectral seed for planned replicas":
                    raise
                spectral = spectral_kmeans_seed(
                    np.arange(num_experts, dtype=np.int64),
                    weights[layer],
                    pairs[layer],
                    num_ranks,
                    similarity="raw",
                    restarts=spectral_restarts,
                    seed=seed + layer * 1009,
                    load_penalty=0.0,
                )
                planned_extras = None
            if num_replicas > num_experts:
                try:
                    spectral, scales = place_spectral_replicas(
                        spectral,
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        alpha,
                        planned_extras=planned_extras,
                    )
                except ValueError as exc:
                    if (
                        planned_extras is None
                        or str(exc)
                        != "No distinct-rank assignment for redundant experts"
                    ):
                        raise
                    spectral = spectral_kmeans_seed(
                        np.arange(num_experts, dtype=np.int64),
                        weights[layer],
                        pairs[layer],
                        num_ranks,
                        similarity="raw",
                        restarts=spectral_restarts,
                        seed=seed + layer * 1009,
                        load_penalty=0.0,
                    )
                    spectral, scales = place_spectral_replicas(
                        spectral, weights[layer], pairs[layer], num_ranks, alpha
                    )
            else:
                scales = _normalization_scales(
                    spectral, weights[layer], pairs[layer], num_ranks
                )
            return (
                spectral
                if assignment_load_weight is not None
                else improve_spectral_by_swapping(
                    spectral,
                    weights[layer],
                    pairs[layer],
                    num_ranks,
                    alpha,
                    max_swaps,
                    scales,
                )
            )

        # Each rank places layers in parallel. Letting every layer's LAPACK
        # call spawn a BLAS team oversubscribes the host across EP ranks.
        blas_limit = (
            threadpool_limits(limits=1, user_api="blas")
            if threadpool_limits is not None
            else nullcontext()
        )
        with blas_limit:
            if num_layers == 0:
                result = np.empty((0, num_replicas), dtype=np.int64)
            elif num_layers == 1:
                result = place_layer(0)[None, :]
            else:
                # Layers are independent. Numba releases the GIL in the hot loops.
                with ThreadPoolExecutor(max_workers=min(8, num_layers)) as pool:
                    result = np.stack(list(pool.map(place_layer, range(num_layers))))
        if old_global_expert_indices is not None:
            result = cls.preserve_intragpu_slots(
                result, num_ranks, old_global_expert_indices.cpu().numpy()
            )
        for layer in range(num_layers):
            if np.any(pairs[layer]):
                _check_distinct_rank_copies(result[layer], num_ranks)
        return torch.from_numpy(result)
