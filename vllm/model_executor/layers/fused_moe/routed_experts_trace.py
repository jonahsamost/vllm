# SPDX-License-Identifier: Apache-2.0
"""Opt-in, worker-local routed-expert traces for offline MoE experiments."""

from __future__ import annotations

import json
import os
import queue
import struct
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class _TraceEntry:
    metadata: dict
    host: torch.Tensor | None
    event: torch.cuda.Event | None
    snapshot: torch.Tensor | None


class RoutedExpertsTraceWriter:
    def __init__(self, model: torch.nn.Module, config: VllmConfig, root: Path):
        from vllm.model_executor.layers.fused_moe.layer import MoERunner

        self.active_file = root / "active"
        self.rank = dist.get_rank()
        self.rank_dir = root / f"rank-{self.rank:02d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.capturer = RoutedExpertsCapturer(
            config.scheduler_config.max_num_batched_tokens, config
        )
        bind_routed_experts_capturer(model, self.capturer)
        self.layer_ids = sorted(
            {m.layer_id for m in model.modules() if isinstance(m, MoERunner)}
        )
        if not self.layer_ids:
            raise ValueError("No MoE layers found for routed-expert tracing")
        self.next_step = 0
        self._entries: queue.Queue[_TraceEntry | None] = queue.Queue(maxsize=8)
        self._error: BaseException | None = None
        (self.rank_dir / "config.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "world_rank": self.rank,
                    "dp_rank": config.parallel_config.data_parallel_rank,
                    "tp_size": config.parallel_config.tensor_parallel_size,
                    "layer_ids": self.layer_ids,
                    "top_k": self.capturer.shape_per_token[1],
                    "num_logical_experts": config.model_config.get_num_experts(),
                    "dtype": self.capturer.output_dtype_name,
                    "compression": "zlib-1",
                    "model": config.model_config.model,
                },
                indent=2,
            )
            + "\n"
        )
        self._thread = threading.Thread(target=self._write, daemon=True)
        self._thread.start()

    def record(
        self,
        num_tokens: int,
        *,
        has_forward: bool,
        request_ids: list[str] | None = None,
        query_start_loc: list[int] | None = None,
    ) -> None:
        if self._error is not None:
            raise RuntimeError("Routed-expert trace writer failed") from self._error
        if not self.active_file.exists():
            return
        if num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative")
        metadata = {
            "step": self.next_step,
            "num_tokens": num_tokens,
            "has_forward": has_forward,
            "request_ids": request_ids or [],
            "query_start_loc": query_start_loc or [],
        }
        self.next_step += 1

        host = event = snapshot = None
        if num_tokens:
            if not has_forward:
                raise ValueError("A step without a forward cannot have tokens")
            snapshot = self.capturer.device_buffer[
                :num_tokens, self.layer_ids, :
            ].to(self.capturer.output_dtype)
            host = torch.empty(
                snapshot.shape,
                dtype=snapshot.dtype,
                device="cpu",
                pin_memory=True,
            )
            host.copy_(snapshot, non_blocking=True)
            event = torch.cuda.Event(blocking=True)
            event.record(torch.cuda.current_stream())
        self._enqueue(_TraceEntry(metadata, host, event, snapshot))

    def _enqueue(self, entry: _TraceEntry | None) -> None:
        while True:
            if self._error is not None:
                raise RuntimeError("Routed-expert trace writer failed") from self._error
            try:
                self._entries.put(entry, timeout=1)
                return
            except queue.Full:
                continue

    def _write(self) -> None:
        try:
            with (
                (self.rank_dir / "steps.jsonl").open("w") as metadata_file,
                (self.rank_dir / "routes.bin").open("wb") as routes_file,
            ):
                while (entry := self._entries.get()) is not None:
                    if entry.event is not None:
                        entry.event.synchronize()
                    raw = (
                        entry.host.numpy().tobytes()
                        if entry.host is not None
                        else b""
                    )
                    compressed = zlib.compress(raw, level=1)
                    entry.metadata["offset"] = routes_file.tell()
                    entry.metadata["compressed_bytes"] = len(compressed)
                    entry.metadata["raw_bytes"] = len(raw)
                    routes_file.write(struct.pack("<I", len(compressed)))
                    routes_file.write(compressed)
                    metadata_file.write(json.dumps(entry.metadata) + "\n")
                    routes_file.flush()
                    metadata_file.flush()
                    self._entries.task_done()
                metadata_file.flush()
                routes_file.flush()
                self._entries.task_done()
        except BaseException as exc:
            self._error = exc

    def close(self) -> None:
        self._enqueue(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Routed-expert trace writer failed") from self._error


def maybe_create_routed_experts_trace(
    model: torch.nn.Module, config: VllmConfig
) -> RoutedExpertsTraceWriter | None:
    root = os.getenv("VLLM_ROUTED_EXPERT_TRACE_DIR")
    if not root:
        return None
    return RoutedExpertsTraceWriter(model, config, Path(root))
