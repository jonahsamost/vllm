# SPDX-License-Identifier: Apache-2.0
"""Opt-in, capture-independent engine-step service timing for SimFer.

This module deliberately has no dependency on ``simfer_capture`` or a device
profiler.  A service-timing process writes to its own directory while
``VLLM_SIMFER_CAPTURE_DIR`` is absent.  The timer encloses scheduling, model
execution, sampling, and scheduler reconciliation at the EngineCore boundary.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_OUTPUT_ENV = "VLLM_SIMFER_SERVICE_TIMING_DIR"
_CAPTURE_ID_ENV = "VLLM_SIMFER_SERVICE_TIMING_CAPTURE_ID"
_TOKEN_COUNTS_ENV = "VLLM_SIMFER_SERVICE_TIMING_TOKEN_COUNTS"
_MAX_STEPS_ENV = "VLLM_SIMFER_SERVICE_TIMING_MAX_STEPS"


@dataclass(frozen=True)
class ServiceTimingSelection:
    sample_index: int
    source_step_index: int
    started_ns: int
    scheduled_tokens: int
    scheduled_requests: int


class ServiceTimingRecorder:
    """Select exact scheduler shapes and append immutable engine-step samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source_steps = 0
        self._samples = 0
        self._armed = False
        self._iteration_matches = 0
        self._iteration_skip_matches = 0
        self._token_counts: frozenset[int] = frozenset()

    @property
    def enabled(self) -> bool:
        return bool(os.environ.get(_OUTPUT_ENV))

    def begin(self) -> int | None:
        if not self.enabled:
            return None
        if os.environ.get("VLLM_SIMFER_CAPTURE_DIR"):
            raise RuntimeError(
                "production service timing and structural capture require "
                "separate processes"
            )
        with self._lock:
            if not self._armed:
                return None
        return time.perf_counter_ns()

    def arm_iteration(self, *, skip_matches: int) -> None:
        """Select one exact-shape step in the next synthetic iteration."""

        if not self.enabled:
            raise RuntimeError("service timing is not enabled")
        if skip_matches < 0:
            raise ValueError("service-timing iteration skip cannot be negative")
        token_counts, maximum = _selection_config()
        with self._lock:
            if self._armed:
                raise RuntimeError("previous service-timing iteration is incomplete")
            if self._samples >= maximum:
                raise RuntimeError("service-timing sample budget is exhausted")
            self._armed = True
            self._iteration_matches = 0
            self._iteration_skip_matches = skip_matches
            self._token_counts = token_counts

    def finish_iteration(self) -> None:
        """Require the armed iteration to have produced its selected step."""

        with self._lock:
            if self._armed:
                self._armed = False
                raise RuntimeError(
                    "synthetic iteration emitted no selected service-timing step"
                )

    def claim(
        self, scheduler_output: Any, *, started_ns: int | None
    ) -> ServiceTimingSelection | None:
        if started_ns is None:
            return None
        scheduled_tokens = int(scheduler_output.total_num_scheduled_tokens)
        scheduled_requests = len(scheduler_output.num_scheduled_tokens)
        with self._lock:
            source_step_index = self._source_steps
            self._source_steps += 1
            if not self._armed or scheduled_tokens not in self._token_counts:
                return None
            self._iteration_matches += 1
            if self._iteration_matches <= self._iteration_skip_matches:
                return None
            sample_index = self._samples
            self._samples += 1
            self._armed = False
        return ServiceTimingSelection(
            sample_index=sample_index,
            source_step_index=source_step_index,
            started_ns=started_ns,
            scheduled_tokens=scheduled_tokens,
            scheduled_requests=scheduled_requests,
        )

    def finish(self, selection: ServiceTimingSelection | None) -> None:
        if selection is None:
            return
        finished_ns = time.perf_counter_ns()
        if finished_ns <= selection.started_ns:
            raise RuntimeError("service-timing monotonic clock did not advance")
        root = Path(os.environ[_OUTPUT_ENV]).resolve()
        capture_id = os.environ.get(_CAPTURE_ID_ENV)
        if not capture_id:
            raise ValueError(f"{_CAPTURE_ID_ENV} is required")
        root.mkdir(parents=True, exist_ok=True)
        rank = _rank()
        event = {
            "schema_version": 1,
            "kind": "engine_core_step_service_timing",
            "capture_id": capture_id,
            "rank": rank,
            **asdict(selection),
            "finished_ns": finished_ns,
            "duration_ns": finished_ns - selection.started_ns,
            "timer": "engine_core_step_perf_counter",
            "clock_scope": "single_host_monotonic",
            "instrumentation": "none",
        }
        encoded = (json.dumps(event, sort_keys=True) + "\n").encode()
        descriptor = os.open(
            root / f"rank-{rank}.jsonl",
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("service-timing event write made no progress")
                remaining = remaining[written:]
        finally:
            os.close(descriptor)


def _selection_config() -> tuple[frozenset[int], int]:
    raw_counts = os.environ.get(_TOKEN_COUNTS_ENV)
    if raw_counts is None:
        raise ValueError(f"{_TOKEN_COUNTS_ENV} is required")
    try:
        token_counts = frozenset(
            int(value.strip()) for value in raw_counts.split(",") if value.strip()
        )
        maximum = int(os.environ.get(_MAX_STEPS_ENV, "1"))
    except ValueError as error:
        raise ValueError("invalid SimFer service-timing selection") from error
    if (
        not token_counts
        or any(value < 0 for value in token_counts)
        or maximum <= 0
    ):
        raise ValueError("invalid SimFer service-timing selection bounds")
    return token_counts, maximum


def _rank() -> int:
    value = int(os.environ.get("RANK", "0"))
    if value < 0:
        raise ValueError("service-timing rank cannot be negative")
    return value


RECORDER = ServiceTimingRecorder()


__all__ = ["RECORDER", "ServiceTimingRecorder", "ServiceTimingSelection"]
