# SPDX-License-Identifier: Apache-2.0
"""Balanced greedy graph partition seed for correlation-aware placement."""

import logging

import networkx as nx
import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


@njit(cache=True, nogil=True)
def _spectral_distances(embedded: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    num_experts, dimensions = embedded.shape
    num_ranks = len(centroids)
    costs = np.empty((num_experts, num_ranks), dtype=np.float64)
    for expert in range(num_experts):
        for rank in range(num_ranks):
            distance = 0.0
            for dimension in range(dimensions):
                difference = embedded[expert, dimension] - centroids[rank, dimension]
                distance += difference * difference
            costs[expert, rank] = distance
    return costs


@njit(cache=True, nogil=True)
def _spectral_centroids(
    embedded: np.ndarray, assignment: np.ndarray, num_ranks: int
) -> np.ndarray:
    _, dimensions = embedded.shape
    centers = np.zeros((num_ranks, dimensions), dtype=np.float64)
    counts = np.zeros(num_ranks, dtype=np.int32)
    for expert in range(len(embedded)):
        rank = assignment[expert]
        counts[rank] += 1
        for dimension in range(dimensions):
            centers[rank, dimension] += embedded[expert, dimension]
    for rank in range(num_ranks):
        for dimension in range(dimensions):
            centers[rank, dimension] /= counts[rank]
    return centers


@njit(cache=True, nogil=True)
def _spectral_unique_pressure_score(
    candidate: np.ndarray,
    weight: np.ndarray,
    graph: np.ndarray,
    num_ranks: int,
) -> float:
    """Score a unique-copy seed using its already-built pair graph."""
    slots = len(candidate) // num_ranks
    total_pressure = 0.0
    max_pressure = -np.inf
    for rank in range(num_ranks):
        offset = rank * slots
        assignments = 0.0
        overlap = 0.0
        for first in range(slots):
            expert = candidate[offset + first]
            assignments += float(weight[expert])
            for second in range(slots):
                overlap += graph[expert, candidate[offset + second]]
        if assignments == 0:
            return np.nan
        overlap *= 0.5
        pressure = assignments - overlap + 0.5 * overlap * overlap / assignments
        total_pressure += pressure
        max_pressure = max(max_pressure, pressure)
    return max_pressure + 0.25 * total_pressure / num_ranks


@njit(cache=True, nogil=True)
def _assign_spectral_zero_penalty(costs: np.ndarray, slots: int) -> np.ndarray:
    """Assign regret-ordered batches until a rank fills."""
    num_experts, num_ranks = costs.shape
    assignment = np.empty(num_experts, dtype=np.int32)
    remaining = np.ones(num_experts, dtype=np.bool_)
    capacity = np.full(num_ranks, slots, dtype=np.int32)
    unassigned = num_experts
    while unassigned:
        candidates = np.empty(unassigned, dtype=np.int32)
        ranks = np.empty(unassigned, dtype=np.int32)
        regrets = np.empty(unassigned, dtype=np.float64)
        count = 0
        for expert in range(num_experts):
            if not remaining[expert]:
                continue
            best_cost = np.inf
            second_cost = np.inf
            best_rank = -1
            for rank in range(num_ranks):
                if capacity[rank] == 0:
                    continue
                cost = costs[expert, rank]
                if cost < best_cost:
                    second_cost = best_cost
                    best_cost = cost
                    best_rank = rank
                elif cost < second_cost:
                    second_cost = cost
            candidates[count] = expert
            ranks[count] = best_rank
            regrets[count] = second_cost - best_cost
            count += 1
        order = np.argsort(-regrets, kind="mergesort")
        for index in order:
            expert = candidates[index]
            rank = ranks[index]
            assignment[expert] = rank
            remaining[expert] = False
            capacity[rank] -= 1
            unassigned -= 1
            if capacity[rank] == 0:
                break
    return assignment


@njit(cache=True, nogil=True)
def _spectral_unique_restarts(
    initial: np.ndarray,
    embedded: np.ndarray,
    weight: np.ndarray,
    graph: np.ndarray,
    first_centers: np.ndarray,
    num_ranks: int,
) -> np.ndarray:
    """Run unique-copy spectral restarts without Python calls per iteration."""
    num_experts, dimensions = embedded.shape
    slots = num_experts // num_ranks
    best = initial.copy()
    best_score = np.inf
    for restart in range(len(first_centers)):
        centers = np.empty(num_ranks, dtype=np.int32)
        centers[0] = first_centers[restart]
        closest = np.full(num_experts, np.inf)
        for center_index in range(1, num_ranks):
            latest = centers[center_index - 1]
            for expert in range(num_experts):
                distance = 0.0
                for dimension in range(dimensions):
                    difference = (
                        embedded[expert, dimension] - embedded[latest, dimension]
                    )
                    distance += difference * difference
                closest[expert] = min(closest[expert], distance)
            centers[center_index] = np.argmax(closest)
        centroids = embedded[centers].copy()
        previous = np.empty(num_experts, dtype=np.int32)
        for iteration in range(20):
            costs = _spectral_distances(embedded, centroids)
            assignment = _assign_spectral_zero_penalty(costs, slots)
            same = iteration > 0
            if same:
                for expert in range(num_experts):
                    if assignment[expert] != previous[expert]:
                        same = False
                        break
            if same:
                break
            previous[:] = assignment
            centroids = _spectral_centroids(embedded, assignment, num_ranks)
        candidate = np.empty(num_experts, dtype=np.int64)
        positions = np.zeros(num_ranks, dtype=np.int32)
        for expert in range(num_experts):
            rank = assignment[expert]
            candidate[rank * slots + positions[rank]] = expert
            positions[rank] += 1
        score = _spectral_unique_pressure_score(candidate, weight, graph, num_ranks)
        if score < best_score:
            best[:] = candidate
            best_score = score
    return best


def hypergraph_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    sampled_routes: np.ndarray,
    num_ranks: int,
    correlation_weight: float,
    arrival_cap_ratio: float | None = None,
    assignment_cap_ratio: float | None = None,
    objective: str = "affinity",
    load_weight: float = 0.5,
    wave_loads: np.ndarray | None = None,
    wave_load_cap_ratio: float | None = None,
) -> np.ndarray:
    """Greedily co-locate experts that cover the same sampled tokens.

    Each rank tracks the sampled tokens already routed to it, so overlap with
    a new expert measures exact hyperedge coverage rather than pair counts.
    """
    from vllm.distributed.eplb.policy.pressure import _assign_redundant_copies

    num_experts = len(weight)
    slots_per_rank = len(initial) // num_ranks
    if len(initial) != num_experts + num_ranks or num_experts % num_ranks:
        raise ValueError("Hypergraph seed needs one redundant slot per rank")
    if not 0 <= correlation_weight <= 1:
        raise ValueError("Correlation weight must be in [0, 1]")
    if arrival_cap_ratio is not None and arrival_cap_ratio <= 0:
        raise ValueError("Arrival cap ratio must be positive")
    if assignment_cap_ratio is not None and assignment_cap_ratio <= 0:
        raise ValueError("Assignment cap ratio must be positive")
    if wave_load_cap_ratio is not None and wave_load_cap_ratio <= 0:
        raise ValueError("Wave load cap ratio must be positive")
    if wave_load_cap_ratio is not None and (
        wave_loads is None or wave_loads.ndim != 2 or wave_loads.shape[1] != num_experts
    ):
        raise ValueError("Wave load cap requires per-wave expert loads")
    if objective not in ("affinity", "minimax") or load_weight < 0:
        raise ValueError("Unknown hypergraph objective or invalid load weight")
    if sampled_routes.ndim != 2 or np.any(
        (sampled_routes < 0) | (sampled_routes >= num_experts)
    ):
        raise ValueError("Invalid sampled training routes")

    copies = np.bincount(initial, minlength=num_experts)
    copy_load = weight / copies
    wave_copy_loads = wave_loads / copies[None, :] if wave_loads is not None else None
    wave_target = (
        np.maximum(wave_loads.sum(axis=1) / num_ranks, 1)
        if wave_loads is not None
        else None
    )
    incidence = np.zeros((len(sampled_routes), num_experts), dtype=bool)
    incidence[
        np.repeat(np.arange(len(sampled_routes)), sampled_routes.shape[1]),
        sampled_routes.reshape(-1),
    ] = True
    frequency = incidence.sum(axis=0)
    if arrival_cap_ratio is None:
        arrival_cap = None
    else:
        original_rows = initial.reshape(num_ranks, slots_per_rank)
        reference = np.mean(
            [incidence[:, row].any(axis=1).sum() for row in original_rows]
        )
        arrival_cap = arrival_cap_ratio * reference
    target_load = float(copy_load.sum() / num_ranks)
    order = np.argsort(-copy_load, kind="stable")
    rows: list[list[int]] = [[int(expert)] for expert in order[:num_ranks]]
    covered = np.stack([incidence[:, row[0]].copy() for row in rows])
    covered_counts = covered.sum(axis=1)
    rank_loads = np.array([copy_load[row[0]] for row in rows], dtype=np.float64)
    rank_wave_loads = (
        np.stack([wave_copy_loads[:, row[0]] for row in rows], axis=1)
        if wave_copy_loads is not None
        else None
    )
    for expert in order[num_ranks:]:
        available = [
            rank for rank, row in enumerate(rows) if len(row) < slots_per_rank - 1
        ]
        if assignment_cap_ratio is not None:
            allowed = [
                rank
                for rank in available
                if rank_loads[rank] + copy_load[expert]
                <= assignment_cap_ratio * target_load
            ]
            if allowed:
                available = allowed
        if wave_load_cap_ratio is not None:
            assert rank_wave_loads is not None
            assert wave_copy_loads is not None
            assert wave_target is not None
            allowed = [
                rank
                for rank in available
                if np.all(
                    rank_wave_loads[:, rank] + wave_copy_loads[:, expert]
                    <= wave_load_cap_ratio * wave_target
                )
            ]
            if allowed:
                available = allowed
        overlap = np.array(
            [
                np.count_nonzero(covered[rank] & incidence[:, expert])
                for rank in available
            ]
        )
        if arrival_cap is not None:
            projected = covered_counts[available] + frequency[expert] - overlap
            eligible = projected <= arrival_cap
            if np.any(eligible):
                available = [
                    rank
                    for rank, allowed in zip(available, eligible, strict=True)
                    if allowed
                ]
                overlap = overlap[eligible]
        if objective == "minimax":
            projected = covered_counts[available] + frequency[expert] - overlap
            scores = -projected / len(sampled_routes)
            scores -= (
                load_weight * (rank_loads[available] + copy_load[expert]) / target_load
            )
        else:
            scores = correlation_weight * overlap / max(int(frequency[expert]), 1)
            scores -= (
                (1 - correlation_weight)
                * (rank_loads[available] + copy_load[expert])
                / target_load
            )
        selected_index = int(np.argmax(scores))
        selected = available[selected_index]
        rows[selected].append(int(expert))
        covered[selected] |= incidence[:, expert]
        covered_counts[selected] += frequency[expert] - overlap[selected_index]
        rank_loads[selected] += copy_load[expert]
        if rank_wave_loads is not None and wave_copy_loads is not None:
            rank_wave_loads[:, selected] += wave_copy_loads[:, expert]
    return _assign_redundant_copies(rows, copies, weight, lower_pairs)


def greedy_graph_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    correlation_weight: float,
) -> np.ndarray:
    """Place hot experts first, balancing capacity, load, and pair affinity."""
    num_experts = len(weight)
    slots_per_rank = len(initial) // num_ranks
    if len(initial) != num_experts + num_ranks:
        raise ValueError("Graph seed requires one redundant slot per rank")
    if not 0 <= correlation_weight <= 1:
        raise ValueError("correlation_weight must be in [0, 1]")
    copies = np.bincount(initial, minlength=num_experts)
    copy_load = weight / copies
    pair_counts = np.tril(lower_pairs, k=-1).astype(np.float64)
    pair_counts += pair_counts.T
    pair_affinity = pair_counts / (copies[:, None] * copies[None, :])
    target_load = float(copy_load.sum() / num_ranks)
    order = np.argsort(-copy_load, kind="stable")
    rows: list[list[int]] = [[int(expert)] for expert in order[:num_ranks]]
    rank_loads = np.array([copy_load[row[0]] for row in rows], dtype=np.float64)
    for expert in order[num_ranks:]:
        available = [
            rank for rank, row in enumerate(rows) if len(row) < slots_per_rank - 1
        ]
        affinities = np.array(
            [pair_affinity[expert, rows[rank]].sum() for rank in available]
        )
        scale = max(float(affinities.max()), 1.0)
        scores = [
            correlation_weight * affinity / scale
            - (1 - correlation_weight)
            * (rank_loads[rank] + copy_load[expert])
            / target_load
            for rank, affinity in zip(available, affinities, strict=True)
        ]
        selected = available[int(np.argmax(scores))]
        rows[selected].append(int(expert))
        rank_loads[selected] += copy_load[expert]
    # Every rank has exactly one free slot. Choose a legal assignment of extras.
    from vllm.distributed.eplb.policy.pressure import _assign_redundant_copies

    return _assign_redundant_copies(rows, copies, weight, lower_pairs)


def spectral_kmeans_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    *,
    similarity: str,
    restarts: int,
    seed: int,
    dimensions: int | None = None,
    selection: str = "pressure",
    training_routes: np.ndarray | None = None,
    training_hashes: np.ndarray | None = None,
    training_offsets: np.ndarray | None = None,
    load_penalty: float = 0.0,
    replica_adjustment: bool = False,
    assignment_load_weight: float | None = None,
    planned_copies: np.ndarray | None = None,
) -> np.ndarray:
    """Balanced k-means in a multi-axis spectral co-routing embedding."""
    from vllm.distributed.eplb.policy.pressure import (
        _assign_redundant_copies,
        _destination_pressure_score,
    )

    num_experts = len(weight)
    slots_per_rank = len(initial) // num_ranks
    redundant = len(initial) - num_experts
    if num_experts % num_ranks or redundant not in (0, num_ranks):
        raise ValueError("Spectral k-means needs zero or one extra slot per rank")
    base_slots_per_rank = num_experts // num_ranks
    if similarity not in ("raw", "cosine") or restarts < 1 or load_penalty < 0:
        raise ValueError("Invalid spectral k-means parameters")
    if dimensions is None:
        dimensions = num_ranks
    if dimensions < 1 or dimensions > num_experts:
        raise ValueError("Invalid spectral embedding dimensions")
    if selection not in ("pressure", "exact"):
        raise ValueError("Unknown spectral k-means seed selection")
    if assignment_load_weight is not None and assignment_load_weight < 0:
        raise ValueError("Assignment load weight must be nonnegative")
    if planned_copies is not None and (
        planned_copies.shape != (num_experts,) or np.any(planned_copies < 1)
    ):
        raise ValueError("Invalid planned replica counts")
    if selection == "exact" and (
        training_routes is None
        or training_hashes is None
        or training_offsets is None
        or training_routes.ndim != 2
        or len(training_hashes) != len(training_routes)
        or training_offsets[0] != 0
        or training_offsets[-1] != len(training_routes)
        or np.any(np.diff(training_offsets) <= 0)
    ):
        raise ValueError("Exact spectral selection needs nonempty training steps")
    copies = np.bincount(initial, minlength=num_experts)
    copy_load = weight / copies
    assignment_load = (
        weight / planned_copies if planned_copies is not None else copy_load
    )
    target_load = max(float(copy_load.sum() / num_ranks), 1.0)
    if assignment_load_weight is not None:
        load_order = np.argsort(-assignment_load, kind="stable")
        load_coefficients = (
            2 * assignment_load_weight * assignment_load / target_load**2
        )
    graph = np.tril(lower_pairs, k=-1).astype(np.float64)
    graph += graph.T
    if replica_adjustment:
        graph /= copies[:, None] * copies[None, :]
    if similarity == "cosine":
        graph /= np.sqrt(np.maximum(weight[:, None] * weight[None, :], 1))
    degree = np.maximum(graph.sum(axis=1), 1e-12)
    normalized = graph / np.sqrt(degree[:, None] * degree[None, :])
    _, eigenvectors = np.linalg.eigh(normalized)
    embedded = eigenvectors[:, -dimensions:]
    embedded /= np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    if (
        redundant == 0
        and similarity == "raw"
        and not replica_adjustment
        and selection == "pressure"
        and assignment_load_weight is None
        and load_penalty == 0
    ):
        first_centers = rng.integers(num_experts, size=restarts)
        return _spectral_unique_restarts(
            initial, embedded, weight, graph, first_centers, num_ranks
        )
    exact_offsets = training_offsets[:-1] if training_offsets is not None else None
    exact_lengths = np.diff(training_offsets) if training_offsets is not None else None

    def exact_metrics(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert training_routes is not None
        assert training_hashes is not None
        assert exact_offsets is not None
        assert exact_lengths is not None
        row_copies = np.bincount(row, minlength=num_experts)
        mapping = np.full((num_experts, int(row_copies.max())), -1, dtype=np.int16)
        for expert in range(num_experts):
            mapping[expert, : row_copies[expert]] = (
                np.flatnonzero(row == expert) // slots_per_rank
            )
        ordinals = (
            training_hashes[:, None].astype(np.uint64)
            % row_copies[training_routes].astype(np.uint64)
        ).astype(np.intp)
        destinations = mapping[training_routes, ordinals]
        masks = np.bitwise_or.reduce(1 << destinations, axis=1)
        seen = (masks[:, None] & (1 << np.arange(num_ranks))) != 0
        arrivals = np.add.reduceat(seen.astype(np.int32), exact_offsets, axis=0)
        fanout = arrivals.sum(axis=1) / exact_lengths
        skew = arrivals.max(axis=1) * num_ranks / arrivals.sum(axis=1)
        return fanout, skew

    reference_skew = exact_metrics(initial)[1] if selection == "exact" else None
    best = None
    best_score = float("inf")
    for _ in range(restarts):
        # Spread initial centers through the embedding rather than choosing
        # nearly identical experts from the same dense community.
        centers = [int(rng.integers(num_experts))]
        closest = np.full(num_experts, np.inf)
        while len(centers) < num_ranks:
            closest = np.minimum(
                closest,
                ((embedded - embedded[centers[-1]]) ** 2).sum(axis=1),
            )
            centers.append(int(np.argmax(closest)))
        centroids = embedded[centers].copy()
        previous = None
        for _ in range(20):
            costs = _spectral_distances(embedded, centroids)
            assignment = np.empty(num_experts, dtype=np.int32)
            capacity = np.full(num_ranks, base_slots_per_rank)
            rank_loads = np.zeros(num_ranks, dtype=np.float64)
            if assignment_load_weight is not None:
                # Score only the next expert, so updating rank load does not
                # require rescoring every unplaced expert.
                for expert in load_order:
                    offered = costs[expert] + load_coefficients[expert] * rank_loads
                    offered[capacity == 0] = np.inf
                    rank = int(offered.argmin())
                    assignment[expert] = rank
                    capacity[rank] -= 1
                    rank_loads[rank] += assignment_load[expert]
            elif load_penalty == 0:
                assignment = _assign_spectral_zero_penalty(costs, base_slots_per_rank)
            else:
                remaining = set(range(num_experts))
                while remaining:
                    available = np.flatnonzero(capacity)
                    candidates = np.array(sorted(remaining))
                    offered = costs[candidates][:, available] + (
                        load_penalty * rank_loads[available] / target_load
                    )
                    ranked = np.argsort(offered, axis=1)
                    best_cost = offered[np.arange(len(candidates)), ranked[:, 0]]
                    second_cost = (
                        offered[np.arange(len(candidates)), ranked[:, 1]]
                        if len(available) > 1
                        else np.full(len(candidates), np.inf)
                    )
                    chosen_index = int(np.argmax(second_cost - best_cost))
                    chosen = int(candidates[chosen_index])
                    rank = int(available[np.argmin(offered[chosen_index])])
                    assignment[chosen] = rank
                    capacity[rank] -= 1
                    rank_loads[rank] += copy_load[chosen]
                    remaining.remove(chosen)
            if previous is not None and np.array_equal(assignment, previous):
                break
            previous = assignment
            centroids = _spectral_centroids(embedded, assignment, num_ranks)
        rows = [
            np.flatnonzero(assignment == rank).tolist() for rank in range(num_ranks)
        ]
        if any(len(row) != base_slots_per_rank for row in rows):
            raise ValueError("Balanced assignment produced the wrong rank size")
        if planned_copies is not None:
            planned_extras = np.repeat(np.arange(num_experts), planned_copies - 1)
            if len(planned_extras) == num_ranks and np.all(
                assignment[planned_extras] == assignment[planned_extras[0]]
            ):
                continue
        if redundant:
            try:
                candidate = _assign_redundant_copies(rows, copies, weight, lower_pairs)
            except ValueError as exc:
                if str(exc) != "No legal assignment for redundant experts":
                    raise
                # A cluster may leave no legal rank for one of its replicas.
                continue
        else:
            candidate = np.array(
                [expert for row in rows for expert in row], dtype=np.int64
            )
        if selection == "exact":
            assert reference_skew is not None
            fanout, skew = exact_metrics(candidate)
            excess = np.maximum(skew - reference_skew - 0.05, 0)
            score = float(np.mean(fanout + 100 * excess**2))
        elif redundant == 0 and similarity == "raw" and not replica_adjustment:
            score = _spectral_unique_pressure_score(candidate, weight, graph, num_ranks)
        else:
            score = _destination_pressure_score(
                candidate, weight, lower_pairs, num_ranks, 0.5, 0.25
            )
        if score < best_score:
            best, best_score = candidate, score
    if best is None:
        if planned_copies is not None:
            raise ValueError("No legal spectral seed for planned replicas")
        logger.warning(
            "All spectral seeds violated redundant-copy placement; "
            "retaining the valid load-balanced placement"
        )
        return initial.copy()
    return best


def replica_graph_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    *,
    restarts: int,
    seed: int,
) -> np.ndarray:
    """Cluster physical copies using their hash-coupled co-routing graph."""
    from vllm.distributed.eplb.policy.pressure import _destination_pressure_score

    num_slots = len(initial)
    num_experts = len(weight)
    if num_slots % num_ranks or restarts < 1:
        raise ValueError("Invalid replica graph dimensions or restarts")
    copies = np.bincount(initial, minlength=num_experts)
    if np.any(copies == 0):
        raise ValueError("Replica graph requires every logical expert")
    ordinals = np.empty(num_slots, dtype=np.int32)
    for expert in range(num_experts):
        slots = np.flatnonzero(initial == expert)
        ordinals[slots] = np.arange(len(slots))
    copy_counts = copies[initial]
    gcd = np.gcd(copy_counts[:, None], copy_counts[None, :])
    lcm = copy_counts[:, None] * copy_counts[None, :] // gcd
    aligned = (ordinals[:, None] - ordinals[None, :]) % gcd == 0
    pair_graph = np.tril(lower_pairs, k=-1).astype(np.float64)
    pair_graph += pair_graph.T
    graph = pair_graph[np.ix_(initial, initial)] * aligned / lcm
    degree = np.maximum(graph.sum(axis=1), 1e-12)
    normalized = graph / np.sqrt(degree[:, None] * degree[None, :])
    _, eigenvectors = np.linalg.eigh(normalized)
    embedded = eigenvectors[:, -num_ranks:]
    embedded /= np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-12)

    slots_per_rank = num_slots // num_ranks
    rng = np.random.default_rng(seed)
    best = None
    best_score = float("inf")
    for _ in range(restarts):
        centers = [int(rng.integers(num_slots))]
        closest = np.full(num_slots, np.inf)
        while len(centers) < num_ranks:
            closest = np.minimum(
                closest,
                ((embedded - embedded[centers[-1]]) ** 2).sum(axis=1),
            )
            centers.append(int(np.argmax(closest)))
        centroids = embedded[centers].copy()
        previous = None
        for _ in range(20):
            costs = ((embedded[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            assignment = np.empty(num_slots, dtype=np.int32)
            capacity = np.full(num_ranks, slots_per_rank)
            remaining = set(range(num_slots))
            while remaining:
                available = np.flatnonzero(capacity)
                candidates = np.array(sorted(remaining))
                offered = costs[candidates][:, available]
                ranked = np.argsort(offered, axis=1)
                best_cost = offered[np.arange(len(candidates)), ranked[:, 0]]
                second_cost = (
                    offered[np.arange(len(candidates)), ranked[:, 1]]
                    if len(available) > 1
                    else np.full(len(candidates), np.inf)
                )
                chosen = int(candidates[np.argmax(second_cost - best_cost)])
                rank = int(available[np.argmin(costs[chosen, available])])
                assignment[chosen] = rank
                capacity[rank] -= 1
                remaining.remove(chosen)
            if previous is not None and np.array_equal(assignment, previous):
                break
            previous = assignment
            centroids = np.stack(
                [embedded[assignment == rank].mean(axis=0) for rank in range(num_ranks)]
            )

        rows = [
            np.flatnonzero(assignment == rank).tolist() for rank in range(num_ranks)
        ]
        for rank in range(num_ranks):
            seen = set()
            for node in list(rows[rank]):
                expert = int(initial[node])
                if expert not in seen:
                    seen.add(expert)
                    continue
                occupied = set(initial[rows[rank]])
                best_exchange = None
                best_cost = float("inf")
                for other_rank in range(num_ranks):
                    if other_rank == rank or expert in set(initial[rows[other_rank]]):
                        continue
                    for other_node in rows[other_rank]:
                        other_expert = int(initial[other_node])
                        if other_expert in occupied:
                            continue
                        change = (
                            costs[node, other_rank]
                            + costs[other_node, rank]
                            - costs[node, rank]
                            - costs[other_node, other_rank]
                        )
                        if change < best_cost:
                            best_cost = float(change)
                            best_exchange = (other_rank, other_node)
                if best_exchange is None:
                    raise ValueError("Could not separate copies on distinct ranks")
                other_rank, other_node = best_exchange
                rows[rank].remove(node)
                rows[rank].append(other_node)
                rows[other_rank].remove(other_node)
                rows[other_rank].append(node)
                seen.add(int(initial[other_node]))
        candidate = np.concatenate(
            [initial[np.array(sorted(row), dtype=np.intp)] for row in rows]
        )
        if any(
            len(set(candidate[r * slots_per_rank : (r + 1) * slots_per_rank]))
            != slots_per_rank
            for r in range(num_ranks)
        ):
            raise ValueError("Replica graph left duplicate experts on one rank")
        score = _destination_pressure_score(
            candidate, weight, lower_pairs, num_ranks, 0.5, 0.25
        )
        if score < best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def kl_graph_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    *,
    similarity: str,
    neighbors: int,
    restarts: int,
    seed: int,
) -> np.ndarray:
    """Recursively cut the strongest co-routing edges with KL bisection."""
    from vllm.distributed.eplb.policy.pressure import (
        _assign_redundant_copies,
        _destination_pressure_score,
    )

    num_experts = len(weight)
    if len(initial) != num_experts + num_ranks or num_experts % num_ranks:
        raise ValueError("KL graph seed needs one extra slot per rank")
    if similarity not in ("raw", "cosine") or not 1 <= neighbors < num_experts:
        raise ValueError("Invalid KL graph parameters")
    if restarts < 1:
        raise ValueError("KL graph restarts must be positive")
    copies = np.bincount(initial, minlength=num_experts)
    graph = np.tril(lower_pairs, k=-1).astype(np.float64)
    graph += graph.T
    if similarity == "cosine":
        graph /= np.sqrt(np.maximum(weight[:, None] * weight[None, :], 1))
    edges = nx.Graph()
    edges.add_nodes_from(range(num_experts))
    for expert in range(num_experts):
        strong = np.argsort(-graph[expert], kind="stable")[: neighbors + 1]
        for partner in strong:
            if expert != partner and graph[expert, partner] > 0:
                edges.add_edge(
                    expert, int(partner), weight=float(graph[expert, partner])
                )

    def split(nodes: list[int], parts: int, random_seed: int) -> list[list[int]]:
        if parts == 1:
            return [sorted(nodes)]
        left, right = nx.community.kernighan_lin_bisection(
            edges.subgraph(nodes), weight="weight", seed=random_seed, max_iter=20
        )
        return split(list(left), parts // 2, random_seed + 1) + split(
            list(right), parts // 2, random_seed + 1 + parts
        )

    best = None
    best_score = float("inf")
    for restart in range(restarts):
        rows = split(list(range(num_experts)), num_ranks, seed + restart * 101)
        candidate = _assign_redundant_copies(rows, copies, weight, lower_pairs)
        score = _destination_pressure_score(
            candidate, weight, lower_pairs, num_ranks, 0.5, 0.25
        )
        if score < best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def louvain_seed(
    initial: np.ndarray,
    weight: np.ndarray,
    lower_pairs: np.ndarray,
    num_ranks: int,
    *,
    resolution: float,
    neighbors: int,
    affinity_weight: float,
    seed: int,
) -> np.ndarray:
    """Pack co-routing communities into equal-capacity expert ranks."""
    from vllm.distributed.eplb.policy.pressure import _assign_redundant_copies

    num_experts = len(weight)
    if len(initial) != num_experts + num_ranks or num_experts % num_ranks:
        raise ValueError("Community seed needs one redundant slot per rank")
    if resolution <= 0 or affinity_weight < 0 or not 1 <= neighbors < num_experts:
        raise ValueError("Invalid community seed parameters")
    copies = np.bincount(initial, minlength=num_experts)
    pair_counts = np.tril(lower_pairs, k=-1).astype(np.float64)
    pair_counts += pair_counts.T
    graph = nx.Graph()
    graph.add_nodes_from(range(num_experts))
    for expert in range(num_experts):
        for partner in np.argsort(-pair_counts[expert], kind="stable")[: neighbors + 1]:
            if expert != partner and pair_counts[expert, partner] > 0:
                graph.add_edge(
                    expert, int(partner), weight=float(pair_counts[expert, partner])
                )
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed
    )
    blocks = [sorted(group) for group in communities]
    blocks.sort(key=lambda group: (-len(group), -float(weight[group].sum())))
    target_size = num_experts // num_ranks
    rank_rows: list[list[int]] = [[] for _ in range(num_ranks)]
    rank_loads = np.zeros(num_ranks, dtype=np.float64)
    while blocks:
        block = blocks.pop(0)
        eligible = [
            rank
            for rank, row in enumerate(rank_rows)
            if target_size - len(row) >= len(block)
        ]
        if not eligible:
            # Fill one rank with the most connected portion, then pack the
            # remainder as another block.
            spaces = [target_size - len(row) for row in rank_rows]
            chosen = int(np.argmax(spaces))
            degree = pair_counts[np.ix_(block, block)].sum(axis=1)
            order = np.argsort(-degree, kind="stable")
            take = int(spaces[chosen])
            selected = [block[index] for index in order[:take]]
            remaining = [block[index] for index in order[take:]]
            rank_rows[chosen].extend(selected)
            rank_loads[chosen] += (weight[selected] / copies[selected]).sum()
            if remaining:
                blocks.insert(0, remaining)
            continue

        scores = []
        for rank in eligible:
            resident = rank_rows[rank]
            block_load = (weight[block] / copies[block]).sum()
            overlap = (
                pair_counts[np.ix_(block, resident)]
                / (copies[block, None] * copies[None, resident])
            ).sum()
            scores.append(
                (rank_loads[rank] + block_load - affinity_weight * overlap, rank)
            )

        chosen = min(scores)[1]
        rank_rows[chosen].extend(block)
        rank_loads[chosen] += (weight[block] / copies[block]).sum()
    if any(len(row) != target_size for row in rank_rows):
        raise ValueError("Community packing failed to fill rank capacities")
    return _assign_redundant_copies(rank_rows, copies, weight, lower_pairs)
