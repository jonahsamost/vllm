# SPDX-License-Identifier: Apache-2.0
"""Exact SimFer operation provenance for CUDA graph replay.

The feature is opt-in.  During capture, external CUDA events bracket each
instrumented operation.  Once capture finishes, PyTorch's graph metadata maps
those boundary events to CUPTI graph-node IDs.  Replay therefore needs no
Python operation hooks and introduces no marker kernels.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

_ENABLED_ENV = "VLLM_SIMFER_GRAPH_PROVENANCE"
_ACTIVE = threading.local()
_LOCK = threading.Lock()
_NEXT_GRAPH = 0
_SITE_OCCURRENCES: dict[tuple[int, str], int] = {}
_CAPTURED_BINDINGS: set[tuple[int, int, str, int]] = set()


def enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "").lower() in {"1", "true", "yes"}


@dataclass
class _OperationBoundary:
    binding: dict[str, Any]
    begin: torch.cuda.Event
    end: torch.cuda.Event


@dataclass
class GraphCapture:
    graph: torch.cuda.CUDAGraph
    graph_instance_id: str
    runtime_mode: str
    batch_descriptor: str
    operations: list[_OperationBoundary] = field(default_factory=list)

    def finish(self) -> None:
        self.graph.instantiate()
        if not self.operations:
            return
        graph_data = self.graph.get_graph_data()
        nodes = graph_data.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise RuntimeError("SimFer CUDA graph provenance found no graph nodes")
        raw_nodes = _raw_graph_nodes(self.graph)
        if len(raw_nodes) != len(nodes):
            raise RuntimeError("SimFer CUDA graph metadata node counts disagree")
        event_nodes = _event_node_indices(raw_nodes)
        ownership: dict[int, dict[str, Any]] = {}
        for operation in self.operations:
            begin = event_nodes.get(operation.begin.cuda_event)
            end = event_nodes.get(operation.end.cuda_event)
            if begin is None or end is None:
                raise RuntimeError(
                    "SimFer operation boundary event is absent from graph"
                )
            for index in _nodes_between(nodes, begin, end):
                prior = ownership.setdefault(index, operation.binding)
                if prior != operation.binding:
                    raise RuntimeError("SimFer CUDA graph operation scopes overlap")

        from vllm.v1.worker.simfer_capture import record_event

        record_event(
            "cuda_graph_provenance",
            graph_instance_id=self.graph_instance_id,
            runtime_mode=self.runtime_mode,
            batch_descriptor=self.batch_descriptor,
            exec_graph_id=graph_data.get("exec_graph_id"),
            nodes=tuple(
                {
                    "index": item["index"],
                    "node_type": item["node_type"],
                    "graph_id": item["graph_id"],
                    "graph_node_id": item["node_id"],
                    "dependencies": tuple(item["dependencies"]),
                    "operation_correlation": ownership.get(item["index"]),
                }
                for item in nodes
            ),
        )


@contextmanager
def capture_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    runtime_mode: Any,
    batch_descriptor: Any,
) -> Iterator[GraphCapture | None]:
    """Activate operation scopes for one CUDA graph capture."""
    if not enabled():
        yield None
        return
    if getattr(_ACTIVE, "capture", None) is not None:
        raise RuntimeError("nested SimFer CUDA graph capture is not supported")
    global _NEXT_GRAPH
    with _LOCK:
        index = _NEXT_GRAPH
        _NEXT_GRAPH += 1
    capture = GraphCapture(
        graph=graph,
        graph_instance_id=f"rank-{_rank()}:cuda-graph-{index}",
        runtime_mode=getattr(runtime_mode, "value", str(runtime_mode)),
        batch_descriptor=str(batch_descriptor),
    )
    _ACTIVE.capture = capture
    try:
        yield capture
    except BaseException:
        raise
    else:
        capture.finish()
    finally:
        _ACTIVE.capture = None


@contextmanager
def operation_scope(instrumentation_site: str) -> Iterator[bool]:
    """Bracket an instrumented operation when CUDA capture is active."""
    capture = getattr(_ACTIVE, "capture", None)
    if capture is None:
        yield False
        return
    binding, coordinate = _graph_binding(instrumentation_site)
    begin = torch.cuda.Event(enable_timing=False, external=True)
    end = torch.cuda.Event(enable_timing=False, external=True)
    begin.record()
    try:
        yield True
    finally:
        end.record()
        capture.operations.append(_OperationBoundary(binding, begin, end))
        _CAPTURED_BINDINGS.add(coordinate)


def bindings_covered(rank: int, step_index: int) -> bool:
    """Return whether capture observed every planned binding for a step."""
    from vllm.v1.worker.simfer_capture import operation_plan_coordinates

    expected = operation_plan_coordinates(rank, step_index)
    return bool(expected) and expected.issubset(_CAPTURED_BINDINGS)


def _graph_binding(
    instrumentation_site: str,
) -> tuple[dict[str, Any], tuple[int, int, str, int]]:
    from vllm.v1.worker.simfer_capture import graph_operation_bindings

    rank = _rank()
    bindings = graph_operation_bindings(rank, instrumentation_site)
    if not bindings:
        raise RuntimeError(
            "SimFer graph provenance has no binding for "
            f"rank={rank}, site={instrumentation_site!r}"
        )
    key = (rank, instrumentation_site)
    occurrence = _SITE_OCCURRENCES.get(key, 0) % len(bindings)
    _SITE_OCCURRENCES[key] = _SITE_OCCURRENCES.get(key, 0) + 1
    coordinate, binding = bindings[occurrence]
    return binding, coordinate


def _raw_graph_nodes(graph: torch.cuda.CUDAGraph) -> list[Any]:
    from cuda.bindings import runtime

    raw_graph = graph.raw_cuda_graph()
    error, _, count = runtime.cudaGraphGetNodes(raw_graph)
    _check_cuda(error, "cudaGraphGetNodes(count)")
    error, nodes, actual = runtime.cudaGraphGetNodes(raw_graph, count)
    _check_cuda(error, "cudaGraphGetNodes(nodes)")
    if actual != count:
        raise RuntimeError("CUDA graph node count changed during introspection")
    return list(nodes)


def _event_node_indices(nodes: list[Any]) -> dict[int, int]:
    from cuda.bindings import runtime

    result = {}
    for index, node in enumerate(nodes):
        error, node_type = runtime.cudaGraphNodeGetType(node)
        _check_cuda(error, "cudaGraphNodeGetType")
        if "EventRecord" not in str(node_type):
            continue
        error, event = runtime.cudaGraphEventRecordNodeGetEvent(node)
        _check_cuda(error, "cudaGraphEventRecordNodeGetEvent")
        result[_pointer_value(event)] = index
    return result


def _nodes_between(nodes: list[dict[str, Any]], begin: int, end: int) -> set[int]:
    dependencies = {
        int(node["index"]): {int(value) for value in node["dependencies"]}
        for node in nodes
    }
    dependents: dict[int, set[int]] = {index: set() for index in dependencies}
    for index, parents in dependencies.items():
        for parent in parents:
            dependents[parent].add(index)

    after_begin = _reachable(dependents, begin)
    before_end = _reachable(dependencies, end)
    return (after_begin & before_end) - {begin, end}


def _reachable(edges: dict[int, set[int]], start: int) -> set[int]:
    seen: set[int] = set()
    pending = list(edges.get(start, ()))
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        pending.extend(edges.get(value, ()))
    return seen


def _pointer_value(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if "0x" not in text:
        raise RuntimeError(f"cannot read CUDA handle {text!r}")
    return int(text.split("0x", 1)[1].rstrip(">"), 16)


def _check_cuda(error: Any, operation: str) -> None:
    if int(error) != 0:
        raise RuntimeError(f"{operation} failed with {error}")


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


__all__ = ["bindings_covered", "capture_graph", "enabled", "operation_scope"]
