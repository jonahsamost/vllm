# SPDX-License-Identifier: Apache-2.0
"""Exact SimFer operation provenance for CUDA graph replay.

The feature is opt-in.  During capture, external CUDA events bracket each
instrumented operation.  Once capture finishes, PyTorch's graph metadata maps
those boundary events to CUPTI graph-node IDs.  Replay therefore needs no
Python operation hooks and introduces no marker kernels.
"""

from __future__ import annotations

import logging
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
_NEXT_EVENT = 0
_SITE_OCCURRENCES: dict[tuple[int, str], int] = {}
_CAPTURED_BINDINGS: set[tuple[int, int, str, int]] = set()
_COMPILED_BOUNDARIES: list[_OperationBoundary] = []
_COMPILED_EVENTS: dict[int, torch.cuda.Event] = {}
_DEBUG_STATES: set[tuple[str, str, bool, bool]] = set()
logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "").lower() in {"1", "true", "yes"}


def debug_operation_site(
    instrumentation_site: str, name: str, *, compiling: bool
) -> None:
    """Log distinct instrumented execution states when explicitly enabled."""
    if not os.environ.get("VLLM_SIMFER_DEBUG_PROVENANCE"):
        return
    state = (
        instrumentation_site,
        name,
        compiling,
        getattr(_ACTIVE, "capture", None) is not None,
    )
    if state in _DEBUG_STATES:
        return
    _DEBUG_STATES.add(state)
    logger.warning(
        "SimFer operation site=%s name=%s compiling=%s graph_capture=%s",
        *state,
    )


@dataclass
class _OperationBoundary:
    correlation_id: str
    operation_correlation: dict[str, Any]
    coordinate: tuple[int, int, str, int]
    begin: torch.cuda.Event
    end: torch.cuda.Event


@torch.library.custom_op("vllm::simfer_record_external_event", mutates_args=("anchor",))
def _record_external_event(anchor: torch.Tensor, token: int) -> None:
    """Record an external event from an opaque compiled-graph operation."""
    _COMPILED_EVENTS[token].record()


@_record_external_event.register_fake
def _record_external_event_fake(anchor: torch.Tensor, token: int) -> None:
    return None


@dataclass
class GraphCapture:
    graph: torch.cuda.CUDAGraph
    graph_instance_id: str
    runtime_mode: str
    batch_descriptor: str
    operations: list[_OperationBoundary] = field(default_factory=list)

    def finish(self) -> None:
        self.graph.instantiate()
        graph_data = self.graph.get_graph_data()
        nodes = graph_data.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise RuntimeError("SimFer CUDA graph provenance found no graph nodes")
        raw_nodes = _raw_graph_nodes(self.graph)
        if len(raw_nodes) != len(nodes):
            raise RuntimeError("SimFer CUDA graph metadata node counts disagree")
        event_nodes = _event_node_indices(raw_nodes)
        if os.environ.get("VLLM_SIMFER_DEBUG_PROVENANCE"):
            logger.warning(
                "SimFer graph=%s direct_operations=%d compiled_boundaries=%d "
                "event_nodes=%d",
                self.graph_instance_id,
                len(self.operations),
                len(_COMPILED_BOUNDARIES),
                len(event_nodes),
            )
        operations = [*self.operations, *_compiled_operations(event_nodes)]
        if not operations:
            return
        self.operations = operations
        ownership: dict[int, _OperationBoundary] = {}
        for operation in operations:
            begin = event_nodes.get(operation.begin.cuda_event)
            end = event_nodes.get(operation.end.cuda_event)
            if begin is None or end is None:
                raise RuntimeError(
                    "SimFer operation boundary event is absent from graph"
                )
            operation_nodes = _nodes_between(nodes, begin, end)
            if not operation_nodes:
                raise RuntimeError(
                    "SimFer operation scope captured no CUDA graph nodes"
                )
            for index in operation_nodes:
                prior = ownership.setdefault(index, operation)
                if prior.correlation_id != operation.correlation_id:
                    raise RuntimeError("SimFer CUDA graph operation scopes overlap")
            _CAPTURED_BINDINGS.add(operation.coordinate)

        from vllm.v1.worker.simfer_capture import record_event

        record_event(
            "cuda_graph_provenance",
            graph_instance_id=self.graph_instance_id,
            runtime_mode=self.runtime_mode,
            batch_descriptor=self.batch_descriptor,
            exec_graph_id=graph_data.get("exec_graph_id"),
            nodes=tuple(
                _node_record(item, ownership.get(item["index"])) for item in nodes
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
    active_sites = getattr(_ACTIVE, "operation_sites", set())
    if instrumentation_site in active_sites:
        yield True
        return
    boundary = _new_operation_boundary(instrumentation_site)
    _ACTIVE.operation_sites = {*active_sites, instrumentation_site}
    boundary.begin.record()
    try:
        yield True
    finally:
        boundary.end.record()
        capture.operations.append(boundary)
        _ACTIVE.operation_sites = active_sites


def begin_compiled_operation(
    instrumentation_site: str, anchor: torch.Tensor
) -> _OperationBoundary:
    """Insert a begin marker that survives Dynamo/Inductor compilation."""
    boundary = _new_operation_boundary(instrumentation_site)
    _record_external_event(anchor, _register_event(boundary.begin))
    _COMPILED_BOUNDARIES.append(boundary)
    return boundary


def end_compiled_operation(boundary: _OperationBoundary, anchor: torch.Tensor) -> None:
    """Insert the matching compiled end marker after the operation output."""
    _record_external_event(anchor, _register_event(boundary.end))


def _new_operation_boundary(instrumentation_site: str) -> _OperationBoundary:
    correlation_id, operation_correlation, coordinate = _graph_binding(
        instrumentation_site
    )
    return _OperationBoundary(
        correlation_id=correlation_id,
        operation_correlation=operation_correlation,
        coordinate=coordinate,
        begin=torch.cuda.Event(enable_timing=False, external=True),
        end=torch.cuda.Event(enable_timing=False, external=True),
    )


def _register_event(event: torch.cuda.Event) -> int:
    global _NEXT_EVENT
    with _LOCK:
        token = _NEXT_EVENT
        _NEXT_EVENT += 1
        _COMPILED_EVENTS[token] = event
    return token


def _compiled_operations(event_nodes: dict[int, int]) -> list[_OperationBoundary]:
    operations = []
    for operation in _COMPILED_BOUNDARIES:
        begin = operation.begin.cuda_event in event_nodes
        end = operation.end.cuda_event in event_nodes
        if begin != end:
            raise RuntimeError("SimFer compiled operation boundary is incomplete")
        if begin:
            operations.append(operation)
    return operations


def bindings_covered(rank: int, step_index: int) -> bool:
    """Return whether capture observed every planned binding for a step."""
    from vllm.v1.worker.simfer_capture import graph_operation_plan

    expected = {key for key, _item in graph_operation_plan(rank, step_index)}
    return bool(expected) and expected.issubset(_CAPTURED_BINDINGS)


def _graph_binding(
    instrumentation_site: str,
) -> tuple[str, dict[str, Any], tuple[int, int, str, int]]:
    from vllm.v1.worker.simfer_capture import graph_operation_plan

    rank = _rank()
    step_index = int(os.environ.get("VLLM_SIMFER_OPERATION_STEP_INDEX", "0"))
    bindings = tuple(
        (key, item)
        for key, item in graph_operation_plan(rank, step_index)
        if key[2] == instrumentation_site
    )
    if not bindings:
        raise RuntimeError(
            "SimFer graph provenance has no binding for "
            f"rank={rank}, site={instrumentation_site!r}"
        )
    key = (rank, instrumentation_site)
    occurrence = _SITE_OCCURRENCES.get(key, 0) % len(bindings)
    _SITE_OCCURRENCES[key] = _SITE_OCCURRENCES.get(key, 0) + 1
    coordinate, item = bindings[occurrence]
    operation_correlation = {
        "schema_version": 1,
        "physical_operation_id": item["physical_operation_id"],
        "semantic_operation_ids": tuple(item["semantic_operation_ids"]),
        "timing_key_id": item["timing_key_id"],
        "dependency_correlation_ids": tuple(item["dependency_correlation_ids"]),
    }
    return item["correlation_id"], operation_correlation, coordinate


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


def _node_record(
    node: dict[str, Any], operation: _OperationBoundary | None
) -> dict[str, Any]:
    return {
        "index": node["index"],
        "node_type": node["node_type"],
        "graph_id": node["graph_id"],
        "graph_node_id": node["node_id"],
        "dependencies": tuple(node["dependencies"]),
        "correlation_id": None if operation is None else operation.correlation_id,
        "operation_correlation": (
            None if operation is None else operation.operation_correlation
        ),
    }


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


__all__ = [
    "begin_compiled_operation",
    "bindings_covered",
    "capture_graph",
    "debug_operation_site",
    "enabled",
    "end_compiled_operation",
    "operation_scope",
]
