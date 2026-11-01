# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .abstract import AbstractEplbPolicy


class CorrAwareEplbPolicy(AbstractEplbPolicy):
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
    ) -> torch.Tensor:
        raise NotImplementedError("Correlation-aware EPLB policy is not implemented")
