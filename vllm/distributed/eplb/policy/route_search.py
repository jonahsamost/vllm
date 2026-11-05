# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training-statistics swap searches for a fixed replica count and hash."""

from collections.abc import Callable

import numpy as np


def sampled_destination_score(
    placement: np.ndarray,
    logical_ids: np.ndarray,
    token_hashes: np.ndarray,
    num_ranks: int,
    mean_weight: float,
) -> tuple[float, np.ndarray]:
    """Score actual unique destination sets on sampled training tokens."""
    num_experts = int(logical_ids.max()) + 1
    slots_per_rank = len(placement) // num_ranks
    counts = np.bincount(placement, minlength=num_experts)
    mapping = np.full((len(counts), int(counts.max())), -1, dtype=np.int32)
    for expert in range(len(counts)):
        if counts[expert]:
            mapping[expert, : counts[expert]] = np.flatnonzero(placement == expert)
    chosen_copy = (
        token_hashes[:, None].astype(np.uint64) % counts[logical_ids].astype(np.uint64)
    ).astype(np.intp)
    physical = mapping[logical_ids, chosen_copy]
    destinations = physical // slots_per_rank
    destination_mask = np.bitwise_or.reduce(1 << destinations, axis=1)
    arrivals = np.array(
        [np.count_nonzero(destination_mask & (1 << rank)) for rank in range(num_ranks)]
    )
    score = float(arrivals.max() + mean_weight * arrivals.mean())
    return score, arrivals


def refine_by_sampled_routes(
    placement: np.ndarray,
    logical_ids: np.ndarray,
    token_hashes: np.ndarray,
    num_ranks: int,
    *,
    steps: int,
    proposals: int,
    mean_weight: float,
    seed: int,
) -> np.ndarray:
    """Greedily swap copies to lower sampled training destination pressure."""
    if len(logical_ids) != len(token_hashes) or not len(logical_ids):
        raise ValueError("Sampled routes and hashes must align and be nonempty")
    return _refine_by_swaps(
        placement,
        num_ranks,
        steps=steps,
        proposals=proposals,
        seed=seed,
        scorer=lambda row: sampled_destination_score(
            row, logical_ids, token_hashes, num_ranks, mean_weight
        ),
    )


def refine_by_training_steps(
    placement: np.ndarray,
    logical_ids: np.ndarray,
    token_hashes: np.ndarray,
    step_offsets: np.ndarray,
    source_ranks: np.ndarray | None,
    default_placement: np.ndarray,
    num_ranks: int,
    *,
    steps: int,
    proposals: int,
    skew_slack: float,
    skew_penalty: float,
    assignment_skew_slack: float,
    assignment_skew_penalty: float,
    assignment_skew_aggregation: str,
    objective: str,
    assignment_weight: float,
    fanout_weight: float,
    source_send_weight: float,
    search_method: str,
    anneal_start_temperature: float,
    anneal_final_temperature: float,
    anneal_polish_steps: int,
    replica_move_fraction: float,
    cycle_move_fraction: float,
    block_swap_fraction: float,
    guided_swap_fraction: float,
    guided_replica_fraction: float,
    exhaustive_replica_rounds: int,
    search_restarts: int,
    assignment_load_cap_ratio: float | None,
    assignment_load_cap_penalty: float,
    replica_pair_counts: np.ndarray,
    replica_candidate_loads: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Lower fan-out subject to per-forward destination-skew limits."""
    if step_offsets[0] != 0 or step_offsets[-1] != len(logical_ids):
        raise ValueError("Training step offsets do not cover all routes")
    if len(token_hashes) != len(logical_ids):
        raise ValueError("Training step routes and hashes must align")
    if (
        skew_slack < 0
        or skew_penalty < 0
        or assignment_skew_slack < 0
        or assignment_skew_penalty < 0
    ):
        raise ValueError("Skew search parameters must be nonnegative")
    if not 0 <= replica_move_fraction <= 1:
        raise ValueError("Replica move fraction must be in [0, 1]")
    if not 0 <= cycle_move_fraction <= 1:
        raise ValueError("Cycle move fraction must be in [0, 1]")
    if not 0 <= block_swap_fraction <= 1:
        raise ValueError("Block swap fraction must be in [0, 1]")
    if not 0 <= guided_replica_fraction <= 1:
        raise ValueError("Guided replica fraction must be in [0, 1]")
    if not 0 <= guided_swap_fraction <= 1:
        raise ValueError("Guided swap fraction must be in [0, 1]")
    if exhaustive_replica_rounds < 0:
        raise ValueError("Exhaustive replica rounds must be nonnegative")
    if search_restarts < 1:
        raise ValueError("Search restarts must be positive")
    if assignment_load_cap_ratio is not None and assignment_load_cap_ratio < 1:
        raise ValueError("Assignment load cap ratio must be at least one")
    if assignment_load_cap_penalty < 0:
        raise ValueError("Assignment load cap penalty must be nonnegative")
    if objective not in (
        "fanout",
        "fanout_assignment",
        "combined_pressure",
        "peer_link_pressure",
    ):
        raise ValueError("Unknown exact-route search objective")
    if assignment_skew_aggregation not in ("mean", "max"):
        raise ValueError("Unknown assignment-skew aggregation")
    if search_method not in ("greedy", "anneal"):
        raise ValueError("Unknown exact-route search method")
    if anneal_start_temperature <= 0 or anneal_final_temperature <= 0:
        raise ValueError("Annealing temperatures must be positive")
    if assignment_weight < 0 or fanout_weight < 0 or source_send_weight < 0:
        raise ValueError("Combined pressure weights must be nonnegative")
    if (source_send_weight or objective == "peer_link_pressure") and (
        source_ranks is None or len(source_ranks) != len(logical_ids)
    ):
        raise ValueError("Source pressure requires aligned training sources")
    num_experts = len(np.bincount(placement))
    offsets = step_offsets.astype(np.intp)
    lengths = np.diff(offsets)
    rank_bits = 1 << np.arange(num_ranks)
    slots_per_rank = len(placement) // num_ranks
    source_masks = (
        source_ranks[:, None] == np.arange(num_ranks)
        if source_send_weight or objective == "peer_link_pressure"
        else None
    )

    def metrics(
        row: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        copies = np.bincount(row, minlength=num_experts)
        mapping = np.full((num_experts, int(copies.max())), -1, dtype=np.int32)
        for expert in range(num_experts):
            mapping[expert, : copies[expert]] = np.flatnonzero(row == expert)
        ordinals = (
            token_hashes[:, None].astype(np.uint64)
            % copies[logical_ids].astype(np.uint64)
        ).astype(np.intp)
        destinations = mapping[logical_ids, ordinals] // slots_per_rank
        masks = np.bitwise_or.reduce(1 << destinations, axis=1)
        seen = (masks[:, None] & rank_bits[None, :]) != 0
        arrivals = np.add.reduceat(seen.astype(np.int32), offsets[:-1], axis=0)
        fanout = arrivals.sum(axis=1) / lengths
        skew = arrivals.max(axis=1) * num_ranks / arrivals.sum(axis=1)
        if assignment_skew_penalty or objective in (
            "fanout_assignment",
            "combined_pressure",
        ):
            assignments = np.add.reduceat(
                (destinations[:, :, None] == np.arange(num_ranks)).sum(axis=1),
                offsets[:-1],
                axis=0,
            )
            assignment_skew = (
                assignments.max(axis=1) * num_ranks / assignments.sum(axis=1)
            )
        else:
            assignment_skew = np.zeros_like(skew)
            assignments = np.zeros_like(arrivals)
        if source_masks is not None:
            source_sends = np.add.reduceat(
                source_masks * seen.sum(axis=1, dtype=np.int32)[:, None],
                offsets[:-1],
                axis=0,
            )
            source_pressure = source_sends.max(axis=1) / lengths
        else:
            source_pressure = np.zeros_like(fanout)
        if objective == "peer_link_pressure":
            assert source_masks is not None
            peer_counts = np.add.reduceat(
                (source_masks[:, :, None] & seen[:, None, :]).astype(np.int32),
                offsets[:-1],
                axis=0,
            )
            peer_peak = peer_counts.max(axis=(1, 2)) / lengths
        else:
            peer_peak = np.zeros_like(fanout)
        return (
            fanout,
            skew,
            assignment_skew,
            arrivals,
            assignments,
            source_pressure,
            peer_peak,
        )

    _, default_skew, default_assignment_skew, _, _, _, _ = metrics(
        default_placement
    )

    def scorer(row: np.ndarray) -> tuple[float, np.ndarray]:
        (
            fanout,
            skew,
            assignment_skew,
            arrivals,
            assignments,
            source_pressure,
            peer_peak,
        ) = metrics(row)
        excess = np.maximum(skew - default_skew - skew_slack, 0)
        assignment_excess = np.maximum(
            assignment_skew - default_assignment_skew - assignment_skew_slack, 0
        )
        if objective == "combined_pressure":
            base = (
                arrivals.max(axis=1) / lengths
                + assignment_weight * assignments.max(axis=1) / lengths
                + fanout_weight * fanout
            )
        elif objective == "fanout_assignment":
            base = fanout + assignment_weight * assignments.max(axis=1) / lengths
        elif objective == "peer_link_pressure":
            base = peer_peak + fanout_weight * fanout
        else:
            base = fanout
        base = base + source_send_weight * source_pressure
        score = float(np.mean(base + skew_penalty * excess**2))
        if assignment_skew_aggregation == "max":
            score += assignment_skew_penalty * float(np.max(assignment_excess**2))
        else:
            score += assignment_skew_penalty * float(np.mean(assignment_excess**2))
        if assignment_load_cap_ratio is not None:
            copies = np.bincount(row, minlength=num_experts)
            rank_loads = (
                (replica_candidate_loads[row] / copies[row])
                .reshape(num_ranks, slots_per_rank)
                .sum(axis=1)
            )
            load_skew = rank_loads.max() / rank_loads.mean()
            score += (
                assignment_load_cap_penalty
                * max(load_skew - assignment_load_cap_ratio, 0) ** 2
            )
        return score, arrivals.mean(axis=0)

    if search_method == "greedy":
        best_score = float("inf")
        refined = placement.copy()
        for restart in range(search_restarts):
            trial = _refine_by_swaps(
                placement,
                num_ranks,
                steps=steps,
                proposals=proposals,
                seed=seed + restart * 10007,
                scorer=scorer,
                replica_move_fraction=replica_move_fraction,
                cycle_move_fraction=cycle_move_fraction,
                block_swap_fraction=block_swap_fraction,
                guided_swap_fraction=guided_swap_fraction,
                guided_replica_fraction=guided_replica_fraction,
                replica_pair_counts=replica_pair_counts,
                replica_candidate_loads=replica_candidate_loads,
            )
            trial_score, _ = scorer(trial)
            if trial_score < best_score:
                best_score, refined = trial_score, trial
    else:
        annealed = _anneal_exact_routes(
            placement,
            num_ranks,
            evaluations=steps * proposals,
            start_temperature=anneal_start_temperature,
            final_temperature=anneal_final_temperature,
            replica_move_fraction=replica_move_fraction,
            replica_candidate_loads=replica_candidate_loads,
            seed=seed,
            scorer=scorer,
        )
        refined = _refine_by_swaps(
            annealed,
            num_ranks,
            steps=anneal_polish_steps,
            proposals=proposals,
            seed=seed + 1,
            scorer=scorer,
            replica_move_fraction=replica_move_fraction,
            cycle_move_fraction=cycle_move_fraction,
            block_swap_fraction=block_swap_fraction,
            guided_swap_fraction=guided_swap_fraction,
            guided_replica_fraction=guided_replica_fraction,
            replica_pair_counts=replica_pair_counts,
            replica_candidate_loads=replica_candidate_loads,
        )
    if exhaustive_replica_rounds:
        refined = _exhaustive_replica_replacements(
            refined,
            num_ranks,
            scorer,
            len(replica_candidate_loads),
            exhaustive_replica_rounds,
        )
    return refined


def _exhaustive_replica_replacements(
    placement: np.ndarray,
    num_ranks: int,
    scorer: Callable[[np.ndarray], tuple[float, np.ndarray]],
    num_experts: int,
    rounds: int,
) -> np.ndarray:
    """Try every legal expert at each redundant slot, then take the best move."""
    slots = len(placement) // num_ranks
    current = placement.copy()
    current_score, _ = scorer(current)
    for _ in range(rounds):
        copies = np.bincount(current, minlength=num_experts)
        redundant = np.flatnonzero(copies[current] > 1)
        best_score = current_score - 1e-9
        best_move = None
        for slot in redundant:
            first = int(slot)
            row = current[first // slots * slots : (first // slots + 1) * slots]
            present = np.zeros(num_experts, dtype=bool)
            present[row] = True
            original = int(current[first])
            for target in np.flatnonzero(~present):
                current[first] = target
                score, _ = scorer(current)
                if score < best_score:
                    best_score = score
                    best_move = (first, int(target))
            current[first] = original
        if best_move is None:
            break
        current[best_move[0]] = best_move[1]
        current_score = best_score
    return current


def _anneal_exact_routes(
    placement: np.ndarray,
    num_ranks: int,
    *,
    evaluations: int,
    start_temperature: float,
    final_temperature: float,
    replica_move_fraction: float,
    replica_candidate_loads: np.ndarray,
    seed: int,
    scorer: Callable[[np.ndarray], tuple[float, np.ndarray]],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    slots_per_rank = len(placement) // num_ranks
    current = placement.copy()
    current_score, _ = scorer(current)
    best = current.copy()
    best_score = current_score
    replacement_pool = np.argsort(-replica_candidate_loads)[:128]
    for step in range(evaluations):
        candidate = current.copy()
        if rng.random() < replica_move_fraction:
            copies = np.bincount(candidate, minlength=len(replica_candidate_loads))
            positions = np.flatnonzero(copies[candidate] > 1)
            first = int(rng.choice(positions))
            target = int(rng.choice(replacement_pool))
            row = candidate[
                first // slots_per_rank * slots_per_rank : (first // slots_per_rank + 1)
                * slots_per_rank
            ]
            if target in row:
                continue
            candidate[first] = target
        else:
            ranks = rng.choice(num_ranks, size=2, replace=False)
            first = int(ranks[0] * slots_per_rank + rng.integers(slots_per_rank))
            second = int(ranks[1] * slots_per_rank + rng.integers(slots_per_rank))
            first_row = candidate[
                ranks[0] * slots_per_rank : (ranks[0] + 1) * slots_per_rank
            ]
            second_row = candidate[
                ranks[1] * slots_per_rank : (ranks[1] + 1) * slots_per_rank
            ]
            if candidate[first] in second_row or candidate[second] in first_row:
                continue
            candidate[first], candidate[second] = candidate[second], candidate[first]
        score, _ = scorer(candidate)
        temperature = start_temperature * (final_temperature / start_temperature) ** (
            step / max(evaluations - 1, 1)
        )
        if score < current_score or rng.random() < np.exp(
            (current_score - score) / temperature
        ):
            current = candidate
            current_score = score
            if score < best_score:
                best = candidate.copy()
                best_score = score
    return best


def source_aware_rank_permutation(
    placement: np.ndarray,
    logical_ids: np.ndarray,
    token_hashes: np.ndarray,
    source_ranks: np.ndarray,
    step_offsets: np.ndarray,
    num_ranks: int,
    *,
    fanout_weight: float,
    max_rounds: int = 8,
) -> np.ndarray:
    """Assign intact rank groups to source ranks using measured training routes."""
    if fanout_weight < 0 or max_rounds < 0:
        raise ValueError("Invalid source-aware rank permutation parameters")
    if not len(source_ranks) == len(token_hashes) == len(logical_ids):
        raise ValueError("Training route sources, hashes, and IDs must align")
    slots_per_rank = len(placement) // num_ranks
    offsets = step_offsets.astype(np.intp)
    lengths = np.diff(offsets)
    rank_bits = 1 << np.arange(num_ranks)
    source_masks = [source_ranks == rank for rank in range(num_ranks)]

    def score(row: np.ndarray) -> float:
        copies = np.bincount(row)
        mapping = np.full((len(copies), int(copies.max())), -1, dtype=np.int32)
        for expert in range(len(copies)):
            mapping[expert, : copies[expert]] = np.flatnonzero(row == expert)
        ordinals = (
            token_hashes[:, None].astype(np.uint64)
            % copies[logical_ids].astype(np.uint64)
        ).astype(np.intp)
        destinations = mapping[logical_ids, ordinals] // slots_per_rank
        masks = np.bitwise_or.reduce(1 << destinations, axis=1)
        seen = (masks[:, None] & rank_bits[None, :]) != 0
        per_step = []
        for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            traffic = np.stack(
                [
                    seen[start:stop][source[start:stop]].sum(axis=0)
                    for source in source_masks
                ]
            )
            fanout = traffic.sum() / lengths[index]
            np.fill_diagonal(traffic, 0)
            bottleneck = (traffic.sum(axis=0) + traffic.sum(axis=1)).max()
            per_step.append(bottleneck / lengths[index] + fanout_weight * fanout)
        return float(np.mean(per_step))

    rows = placement.reshape(num_ranks, slots_per_rank).copy()
    current = score(rows.reshape(-1))
    for _ in range(max_rounds):
        best = current - 1e-9
        best_swap = None
        for first in range(num_ranks):
            for second in range(first + 1, num_ranks):
                rows[[first, second]] = rows[[second, first]]
                candidate = score(rows.reshape(-1))
                rows[[first, second]] = rows[[second, first]]
                if candidate < best:
                    best = candidate
                    best_swap = (first, second)
        if best_swap is None:
            break
        first, second = best_swap
        rows[[first, second]] = rows[[second, first]]
        current = best
    return rows.reshape(-1)


def hash_aware_pair_score(
    placement: np.ndarray,
    weight: np.ndarray,
    lower_pair_counts: np.ndarray,
    num_ranks: int,
    overlap_curvature: float,
    mean_weight: float,
) -> tuple[float, np.ndarray]:
    """Estimate unique destinations with the router's shared replica hash."""
    num_experts = len(weight)
    copies = np.bincount(placement, minlength=num_experts)
    ordinal = np.zeros(len(placement), dtype=np.int32)
    for expert in range(num_experts):
        ordinal[np.flatnonzero(placement == expert)] = np.arange(copies[expert])
    pair_counts = np.tril(lower_pair_counts, k=-1)
    pair_counts = pair_counts + pair_counts.T
    slots_per_rank = len(placement) // num_ranks
    pressures = np.zeros(num_ranks, dtype=np.float64)
    for rank in range(num_ranks):
        start = rank * slots_per_rank
        stop = start + slots_per_rank
        row = placement[start:stop]
        row_copies = copies[row]
        row_ordinals = ordinal[start:stop]
        divisor = np.gcd(row_copies[:, None], row_copies[None, :])
        probability = np.where(
            (row_ordinals[:, None] - row_ordinals[None, :]) % divisor == 0,
            divisor / (row_copies[:, None] * row_copies[None, :]),
            0.0,
        )
        assignments = (weight[row] / row_copies).sum()
        overlap = np.triu(pair_counts[np.ix_(row, row)] * probability, k=1).sum()
        pressures[rank] = (
            assignments - overlap + overlap_curvature * overlap * overlap / assignments
        )
    return float(pressures.max() + mean_weight * pressures.mean()), pressures


def refine_by_hash_aware_pairs(
    placement: np.ndarray,
    weight: np.ndarray,
    lower_pair_counts: np.ndarray,
    num_ranks: int,
    *,
    steps: int,
    proposals: int,
    overlap_curvature: float,
    mean_weight: float,
    seed: int,
) -> np.ndarray:
    """Swap copies using an overlap score conditioned on replica hash indices."""
    return _refine_by_swaps(
        placement,
        num_ranks,
        steps=steps,
        proposals=proposals,
        seed=seed,
        scorer=lambda row: hash_aware_pair_score(
            row,
            weight,
            lower_pair_counts,
            num_ranks,
            overlap_curvature,
            mean_weight,
        ),
    )


def refine_by_wave_pressure(
    placement: np.ndarray,
    wave_loads: np.ndarray,
    wave_pairs: np.ndarray,
    wave_tokens: np.ndarray,
    num_ranks: int,
    *,
    steps: int,
    proposals: int,
    overlap_curvature: float,
    mean_weight: float,
    wave_weight_power: float,
    seed: int,
) -> np.ndarray:
    """Improve mean per-wave bottleneck pressure from training waves."""
    copies = np.bincount(placement, minlength=wave_loads.shape[1])
    lower = np.tril(wave_pairs, k=-1)
    pair_overlap = (lower + lower.transpose(0, 2, 1)) / (
        copies[:, None] * copies[None, :]
    )
    copy_loads = wave_loads / copies[None, :]
    wave_weights = wave_tokens.astype(np.float64) ** wave_weight_power
    slots_per_rank = len(placement) // num_ranks

    def scorer(row_placement: np.ndarray) -> tuple[float, np.ndarray]:
        rows = row_placement.reshape(num_ranks, slots_per_rank)
        pressures = np.empty((len(wave_tokens), num_ranks), dtype=np.float64)
        for rank, row in enumerate(rows):
            assignments = copy_loads[:, row].sum(axis=1)
            overlap = np.triu(pair_overlap[:, row][:, :, row], k=1).sum(axis=(1, 2))
            pressures[:, rank] = (
                assignments
                - overlap
                + overlap_curvature * overlap * overlap / np.maximum(assignments, 1.0)
            )
        normalized = pressures / wave_tokens[:, None]
        per_wave = normalized.max(axis=1) + mean_weight * normalized.mean(axis=1)
        return float(np.average(per_wave, weights=wave_weights)), np.average(
            normalized, axis=0, weights=wave_weights
        )

    return _refine_by_swaps(
        placement,
        num_ranks,
        steps=steps,
        proposals=proposals,
        seed=seed,
        scorer=scorer,
    )


def _guided_swap_pairs(
    placement: np.ndarray,
    num_ranks: int,
    loads: np.ndarray,
    pair_counts: np.ndarray,
    max_candidates: int,
) -> list[tuple[int, int]]:
    """Rank cross-GPU swaps by pair overlap, penalizing assignment load growth."""
    slots = len(placement) // num_ranks
    rows = placement.reshape(num_ranks, slots)
    copies = np.bincount(placement, minlength=len(loads))
    copy_load = loads / copies
    affinity = np.tril(pair_counts, k=-1)
    affinity = (affinity + affinity.T) / np.outer(copies, copies)
    rank_load = copy_load[rows].sum(axis=1)
    current_max = rank_load.max()
    row_affinity = [affinity[:, row].sum(axis=1) for row in rows]
    proposals: list[tuple[float, int, int]] = []
    for first_rank in range(num_ranks):
        first = rows[first_rank]
        for second_rank in range(first_rank + 1, num_ranks):
            second = rows[second_rank]
            cross = affinity[np.ix_(first, second)]
            first_gain = (
                row_affinity[first_rank][second][None, :]
                - row_affinity[first_rank][first][:, None]
                - cross
            )
            second_gain = (
                row_affinity[second_rank][first][:, None]
                - row_affinity[second_rank][second][None, :]
                - cross
            )
            new_first = (
                rank_load[first_rank]
                - copy_load[first][:, None]
                + copy_load[second][None, :]
            )
            new_second = (
                rank_load[second_rank]
                - copy_load[second][None, :]
                + copy_load[first][:, None]
            )
            max_growth = np.maximum(np.maximum(new_first, new_second) - current_max, 0)
            score = first_gain + second_gain - 0.5 * max_growth
            valid = ~np.isin(first, second)[:, None] & ~np.isin(second, first)[None, :]
            score = np.where(valid, score, -np.inf)
            top = min(max_candidates, score.size)
            flat = np.argpartition(score.ravel(), -top)[-top:]
            for index in flat:
                value = float(score.ravel()[index])
                if np.isfinite(value):
                    first_slot, second_slot = divmod(int(index), slots)
                    proposals.append(
                        (
                            value,
                            first_rank * slots + first_slot,
                            second_rank * slots + second_slot,
                        )
                    )
    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(first, second) for _, first, second in proposals[:max_candidates]]


def _refine_by_swaps(
    placement: np.ndarray,
    num_ranks: int,
    *,
    steps: int,
    proposals: int,
    seed: int,
    scorer: Callable[[np.ndarray], tuple[float, np.ndarray]],
    replica_move_fraction: float = 0.0,
    cycle_move_fraction: float = 0.0,
    block_swap_fraction: float = 0.0,
    guided_swap_fraction: float = 0.0,
    guided_replica_fraction: float = 0.0,
    replica_pair_counts: np.ndarray | None = None,
    replica_candidate_loads: np.ndarray | None = None,
) -> np.ndarray:
    if len(placement) % num_ranks:
        raise ValueError("Placement must have equal slots on every rank")
    slots_per_rank = len(placement) // num_ranks
    current = placement.copy()
    current_score, arrivals = scorer(current)
    rng = np.random.default_rng(seed)
    if replica_move_fraction:
        assert replica_candidate_loads is not None
        replacement_pool = np.argsort(-replica_candidate_loads)[:128]
    if guided_replica_fraction or guided_swap_fraction:
        assert replica_pair_counts is not None
        assert replica_candidate_loads is not None
        affinity = np.tril(replica_pair_counts, k=-1)
        affinity = affinity + affinity.T
    for _ in range(steps):
        best_score = current_score - 1e-9
        best = None
        best_arrivals = None
        busiest = int(np.argmax(arrivals))
        if guided_swap_fraction:
            guided_swaps = _guided_swap_pairs(
                current,
                num_ranks,
                replica_candidate_loads,
                replica_pair_counts,
                max_candidates=min(proposals, 128),
            )
            guided_swap_index = 0
        if guided_replica_fraction:
            guided_pools = []
            for rank in range(num_ranks):
                row = current[rank * slots_per_rank : (rank + 1) * slots_per_rank]
                scores = affinity[:, row].sum(axis=1) / np.sqrt(
                    np.maximum(replica_candidate_loads, 1)
                )
                scores[row] = -np.inf
                guided_pools.append(np.argsort(-scores, kind="stable")[:32])
        for proposal in range(proposals):
            if replica_move_fraction and rng.random() < replica_move_fraction:
                copies = np.bincount(current, minlength=len(replica_candidate_loads))
                redundant_positions = np.flatnonzero(copies[current] > 1)
                if not len(redundant_positions):
                    continue
                on_busiest = redundant_positions[
                    redundant_positions // slots_per_rank == busiest
                ]
                if len(on_busiest) and rng.random() < 0.5:
                    redundant_positions = on_busiest
                first = int(rng.choice(redundant_positions))
                rank = first // slots_per_rank
                if guided_replica_fraction and rng.random() < guided_replica_fraction:
                    target = int(rng.choice(guided_pools[rank]))
                else:
                    target = int(rng.choice(replacement_pool))
                row = current[rank * slots_per_rank : (rank + 1) * slots_per_rank]
                if target in row:
                    continue
                original = current[first]
                current[first] = target
                score, candidate_arrivals = scorer(current)
                current[first] = original
                if score < best_score:
                    best_score = score
                    best = ("replace", first, target)
                    best_arrivals = candidate_arrivals
                continue
            if cycle_move_fraction and rng.random() < cycle_move_fraction:
                ranks = rng.choice(num_ranks, size=3, replace=False)
                positions = [
                    int(rank * slots_per_rank + rng.integers(slots_per_rank))
                    for rank in ranks
                ]
                first, second, third = positions
                first_row = current[
                    ranks[0] * slots_per_rank : (ranks[0] + 1) * slots_per_rank
                ]
                second_row = current[
                    ranks[1] * slots_per_rank : (ranks[1] + 1) * slots_per_rank
                ]
                third_row = current[
                    ranks[2] * slots_per_rank : (ranks[2] + 1) * slots_per_rank
                ]
                if (
                    current[third] in first_row
                    or current[first] in second_row
                    or current[second] in third_row
                ):
                    continue
                current[first], current[second], current[third] = (
                    current[third],
                    current[first],
                    current[second],
                )
                score, candidate_arrivals = scorer(current)
                current[first], current[second], current[third] = (
                    current[second],
                    current[third],
                    current[first],
                )
                if score < best_score:
                    best_score = score
                    best = ("cycle", first, second, third)
                    best_arrivals = candidate_arrivals
                continue
            if block_swap_fraction and rng.random() < block_swap_fraction:
                if proposal % 2 == 0:
                    first_rank = busiest
                    second_rank = int(rng.integers(num_ranks - 1))
                    if second_rank >= first_rank:
                        second_rank += 1
                else:
                    first_rank, second_rank = rng.choice(
                        num_ranks, size=2, replace=False
                    )
                first_slots = rng.choice(slots_per_rank, size=2, replace=False)
                second_slots = rng.choice(slots_per_rank, size=2, replace=False)
                first_positions = first_rank * slots_per_rank + first_slots
                second_positions = second_rank * slots_per_rank + second_slots
                first_row = current[
                    first_rank * slots_per_rank : (first_rank + 1) * slots_per_rank
                ]
                second_row = current[
                    second_rank * slots_per_rank : (second_rank + 1) * slots_per_rank
                ]
                first_rest = np.delete(first_row, first_slots)
                second_rest = np.delete(second_row, second_slots)
                if np.isin(current[second_positions], first_rest).any() or np.isin(
                    current[first_positions], second_rest
                ).any():
                    continue
                positions = np.concatenate((first_positions, second_positions))
                originals = current[positions].copy()
                current[positions] = originals[[2, 3, 0, 1]]
                score, candidate_arrivals = scorer(current)
                current[positions] = originals
                if score < best_score:
                    best_score = score
                    best = ("block", *positions.tolist())
                    best_arrivals = candidate_arrivals
                continue
            if (
                guided_swap_fraction
                and guided_swaps
                and rng.random() < guided_swap_fraction
            ):
                first, second = guided_swaps[guided_swap_index % len(guided_swaps)]
                guided_swap_index += 1
            else:
                if proposal % 2 == 0:
                    first_rank = busiest
                    second_rank = int(rng.integers(num_ranks - 1))
                    if second_rank >= first_rank:
                        second_rank += 1
                else:
                    first_rank, second_rank = rng.choice(
                        num_ranks, size=2, replace=False
                    )
                first_slot = int(rng.integers(slots_per_rank))
                second_slot = int(rng.integers(slots_per_rank))
                first = first_rank * slots_per_rank + first_slot
                second = second_rank * slots_per_rank + second_slot
            first_rank = first // slots_per_rank
            second_rank = second // slots_per_rank
            first_expert = current[first]
            second_expert = current[second]
            if (
                first_expert == second_expert
                or second_expert
                in current[
                    first_rank * slots_per_rank : (first_rank + 1) * slots_per_rank
                ]
                or first_expert
                in current[
                    second_rank * slots_per_rank : (second_rank + 1) * slots_per_rank
                ]
            ):
                continue
            current[first], current[second] = second_expert, first_expert
            score, candidate_arrivals = scorer(current)
            current[first], current[second] = first_expert, second_expert
            if score < best_score:
                best_score = score
                best = ("swap", first, second)
                best_arrivals = candidate_arrivals
        if best is None:
            break
        kind, first, second, *rest = best
        if kind == "swap":
            current[first], current[second] = current[second], current[first]
        elif kind == "cycle":
            third = rest[0]
            current[first], current[second], current[third] = (
                current[third],
                current[first],
                current[second],
            )
        elif kind == "block":
            positions = np.array([first, second, *rest], dtype=np.intp)
            current[positions] = current[positions][[2, 3, 0, 1]]
        else:
            current[first] = second
        current_score = best_score
        assert best_arrivals is not None
        arrivals = best_arrivals
    return current
