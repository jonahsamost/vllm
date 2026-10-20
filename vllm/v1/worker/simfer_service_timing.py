# SPDX-License-Identifier: Apache-2.0
"""Exact, profiler-free scheduler-step service timing for SimFer evidence."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

_Result = TypeVar("_Result")


@dataclass(frozen=True)
class ServiceTimingObservation:
    started_ns: int
    finished_ns: int

    @property
    def duration_ns(self) -> int:
        return self.finished_ns - self.started_ns


class ServiceTimingSelector:
    """Claim a bounded ordinal sample set from exact scheduler token shapes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timed_steps = 0
        self._matches = 0

    def claim(self, scheduler_output: Any) -> int | None:
        raw_counts = os.environ.get("VLLM_SIMFER_SERVICE_TIMING_TOKEN_COUNTS")
        if raw_counts is None:
            return None
        try:
            token_counts = {
                int(value.strip()) for value in raw_counts.split(",") if value.strip()
            }
            maximum = int(os.environ.get("VLLM_SIMFER_SERVICE_TIMING_MAX_STEPS", "1"))
            skip_matches = int(
                os.environ.get("VLLM_SIMFER_SERVICE_TIMING_SKIP_MATCHES", "0")
            )
        except ValueError as error:
            raise ValueError("invalid SimFer service-timing step selection") from error
        if not token_counts:
            raise ValueError("VLLM_SIMFER_SERVICE_TIMING_TOKEN_COUNTS cannot be empty")
        if any(value < 0 for value in token_counts) or maximum <= 0 or skip_matches < 0:
            raise ValueError(
                "SimFer service-timing token counts must be nonnegative; "
                "maximum must be positive"
            )
        scheduled = getattr(scheduler_output, "total_num_scheduled_tokens", None)
        with self._lock:
            if scheduled not in token_counts:
                return None
            self._matches += 1
            if self._matches <= skip_matches or self._timed_steps >= maximum:
                return None
            sample_index = self._timed_steps
            self._timed_steps += 1
        return sample_index


_SELECTOR = ServiceTimingSelector()


def claim_service_timing(scheduler_output: Any) -> int | None:
    return _SELECTOR.claim(scheduler_output)


def observe_service_time(
    execute: Callable[[], _Result],
) -> tuple[_Result, ServiceTimingObservation]:
    """Execute one step between device-complete monotonic host timestamps."""
    torch.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    result = execute()
    torch.cuda.synchronize()
    finished_ns = time.perf_counter_ns()
    return result, ServiceTimingObservation(started_ns, finished_ns)
