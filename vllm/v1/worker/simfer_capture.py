# SPDX-License-Identifier: Apache-2.0
"""Opt-in structural and composition evidence for SimFer conformance.

Structural capture is diagnostic evidence. It does not produce production
service-time observations: those are emitted independently at the engine-core
boundary, so enabling this module can never contaminate authoritative timing.
Set ``VLLM_SIMFER_CAPTURE_DIR`` to emit one JSONL stream per worker rank.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import torch

_SCHEMA_VERSION = 2
_LOCK = threading.Lock()
_EVENT_INDICES: dict[Path, int] = {}
_CORRELATION_INDEX = 0
_NEXT_STEP = 0
_PROFILED_STEPS = 0
_PROFILE_MATCHES = 0
_ACTIVE_STEP: int | None = None
_ACTIVE_STEP_CORRELATION: str | None = None
_ACTIVE_OPERATION_STEP: int | None = None
_ACTIVE_PHASE = threading.local()
_INITIALIZED_STREAMS: set[Path] = set()
_OPERATION_PLAN_PATH: Path | None = None
_OPERATION_PLAN: dict[tuple[int, int, str, int], dict[str, Any]] = {}
_OPERATION_OCCURRENCES: dict[tuple[int, int, str], int] = {}
_CLAIMED_OPERATION_BINDINGS: set[tuple[int, int, str, int]] = set()
_PROVENANCE_ENV = {
    "capture_id": "VLLM_SIMFER_CAPTURE_ID",
    "vllm_commit": "VLLM_SIMFER_VLLM_COMMIT",
    "model_config_sha256": "VLLM_SIMFER_MODEL_CONFIG_SHA256",
}


@dataclass(frozen=True)
class OperationCorrelation:
    """Exact SimFer identity for one explicitly instrumented operation."""

    physical_operation_id: str
    semantic_operation_ids: tuple[str, ...]
    timing_key_id: str
    dependency_correlation_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported SimFer operation-correlation schema")
        if not self.physical_operation_id or not self.timing_key_id:
            raise ValueError("SimFer operation correlation is incomplete")
        for name, values in (
            ("semantic operation IDs", self.semantic_operation_ids),
            ("operation dependency IDs", self.dependency_correlation_ids),
        ):
            if values != tuple(sorted(set(values))) or any(
                not value for value in values
            ):
                raise ValueError(f"SimFer {name} must be nonempty canonical strings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def enabled() -> bool:
    return bool(os.environ.get("VLLM_SIMFER_CAPTURE_DIR"))


def record_event(
    kind: str,
    *,
    correlation_id: str | None = None,
    parent_correlation_id: str | None = None,
    **payload: Any,
) -> None:
    if not enabled():
        return
    if not kind:
        raise ValueError("SimFer capture event kind cannot be empty")
    provenance = _capture_provenance()
    rank = _rank()
    with _LOCK:
        path = Path(os.environ["VLLM_SIMFER_CAPTURE_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        stream = path / f"rank-{rank}.jsonl"
        if stream not in _INITIALIZED_STREAMS:
            if stream.exists() and stream.stat().st_size:
                raise ValueError(
                    f"SimFer capture stream already exists and is not empty: {stream}"
                )
            _INITIALIZED_STREAMS.add(stream)
        event_index = _EVENT_INDICES.get(stream, 0)
        _EVENT_INDICES[stream] = event_index + 1
        event = {
            "schema_version": _SCHEMA_VERSION,
            **provenance,
            "event_index": event_index,
            "kind": kind,
            "rank": rank,
            "step_index": _ACTIVE_STEP,
            "timestamp_ns": time.monotonic_ns(),
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "correlation_id": correlation_id,
            "parent_correlation_id": parent_correlation_id,
            "payload": _jsonable(payload),
        }
        with stream.open("a") as output:
            output.write(json.dumps(event, sort_keys=True) + "\n")


def record_runtime_dispatch(
    scheduler_output: Any,
    cudagraph_mode: Any,
    batch_descriptor: Any,
    *,
    num_requests: int,
    should_ubatch: bool,
    num_tokens_across_dp: Any,
) -> None:
    if not enabled():
        return
    record_event(
        "runtime_dispatch",
        correlation_id=_ACTIVE_STEP_CORRELATION,
        request_ids=tuple(scheduler_output.num_scheduled_tokens),
        scheduled_tokens=tuple(scheduler_output.num_scheduled_tokens.items()),
        actual_num_tokens=scheduler_output.total_num_scheduled_tokens,
        actual_num_requests=num_requests,
        cudagraph_mode=getattr(cudagraph_mode, "value", str(cudagraph_mode)),
        batch_descriptor=batch_descriptor,
        should_ubatch=should_ubatch,
        num_tokens_across_dp=num_tokens_across_dp,
    )


def capture_step(function: Any) -> Any:
    """Record one worker API span and explicitly selected timing evidence.

    Ordinary structural capture remains asynchronous. Explicit profiler
    selection is diagnostic and intentionally observes device completion.
    """

    @wraps(function)
    def wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        if not enabled():
            return function(self, scheduler_output, *args, **kwargs)
        correlation = _begin_step(scheduler_output)
        outcome = "completed"
        profile_path = None
        try:
            profile_path = _claim_profile_path(scheduler_output)
            _activate_operation_step(profile_path)
            if profile_path is None:
                with _profiler_marker("worker_step", correlation, None):
                    return function(self, scheduler_output, *args, **kwargs)
            with torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ),
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                with _profiler_marker("worker_step", correlation, None):
                    result = function(self, scheduler_output, *args, **kwargs)
                    # Profiling mode intentionally observes device completion.
                    # Keep the sync correlated to the worker step; profiler
                    # setup/teardown synchronization remains tool overhead.
                    torch.cuda.synchronize()
            profiler.export_chrome_trace(str(profile_path))
            record_event(
                "profiler_artifact",
                correlation_id=correlation,
                tool="pytorch-profiler",
                path=profile_path.name,
                schema="kineto-chrome-trace",
            )
            return result
        except BaseException:
            outcome = "error"
            raise
        finally:
            _end_step(
                correlation,
                outcome,
                device_completion_observed=profile_path is not None,
            )

    return wrapped


@contextmanager
def record_phase(name: str, *, resource: str = "host", **metadata: Any) -> Any:
    """Record a nested host phase suitable for profiler correlation."""

    if not enabled():
        yield None
        return
    if _ACTIVE_STEP is None or _ACTIVE_STEP_CORRELATION is None:
        raise RuntimeError("SimFer phase range requires an active worker step")
    correlation = _new_correlation_id("phase")
    with _record_range(name, resource, correlation, metadata, None):
        yield correlation


@contextmanager
def record_step_phase(
    name: str, *, resource: str = "host", **metadata: Any
) -> Any:
    """Record a phase only when execution belongs to a scheduler step.

    Model execution also occurs during initialization, memory profiling,
    compilation, graph capture, and warmup. Those calls are valid capture
    lifecycle work but have no scheduler-step parent. Keep them in the
    enclosing lifecycle timers instead of fabricating a step correlation or
    weakening ``record_phase``'s fail-closed contract for callers that require
    an active step.
    """
    if not enabled() or _ACTIVE_STEP is None or _ACTIVE_STEP_CORRELATION is None:
        yield None
        return
    with record_phase(name, resource=resource, **metadata) as correlation:
        yield correlation


@contextmanager
def record_operation(
    name: str,
    binding: OperationCorrelation,
    *,
    correlation_id: str,
    resource: str,
    **metadata: Any,
) -> Any:
    """Emit one exact operation range into capture and profiler evidence.

    Callers must provide identities exported by the matching SimFer structural
    run.  This API intentionally has no symbol-name or ordering fallback: an
    unbound operation remains whole-step evidence rather than being assigned
    a plausible but false identity.
    """

    if not enabled():
        yield None
        return
    if _ACTIVE_STEP is None or _ACTIVE_STEP_CORRELATION is None:
        raise RuntimeError("SimFer operation range requires an active worker step")
    if not name or not resource or not correlation_id:
        raise ValueError(
            "SimFer operation range requires name, resource, and correlation ID"
        )
    operation = binding.to_dict()
    event_metadata = {"operation_correlation": operation, **metadata}
    with _record_range(
        name,
        resource,
        correlation_id,
        event_metadata,
        {"operation_correlation": operation},
        repeat_metadata_on_end=True,
    ):
        yield correlation_id


@contextmanager
def record_bound_operation(
    instrumentation_site: str,
    name: str,
    *,
    resource: str = "gpu",
    **metadata: Any,
) -> Any:
    """Record the next exact structural binding for one pinned call site.

    The content-addressed plan is produced before observation.  A call site is
    never matched by symbol name, shape, or best effort: rank, worker step,
    site, and occurrence must all agree.  If no plan is configured the helper
    is a zero-overhead semantic no-op apart from entering this Python context.
    """

    if (
        not enabled()
        or not os.environ.get("VLLM_SIMFER_OPERATION_PLAN")
        or _ACTIVE_OPERATION_STEP is None
    ):
        yield None
        return
    if not instrumentation_site:
        raise ValueError("SimFer instrumentation site cannot be empty")
    binding, correlation_id = _claim_operation_binding(instrumentation_site)
    with record_operation(
        name,
        binding,
        correlation_id=correlation_id,
        resource=resource,
        instrumentation_site=instrumentation_site,
        **metadata,
    ):
        yield correlation_id


@contextmanager
def record_cuda_graph_replay(graph_instance_id: str | None) -> Any:
    """Mark one graph launch so profiler evidence can join it to capture."""
    if (
        not enabled()
        or graph_instance_id is None
        or _ACTIVE_STEP is None
        or _ACTIVE_STEP_CORRELATION is None
    ):
        yield None
        return
    correlation = _new_correlation_id("cuda-graph-replay")
    with _record_range(
        "cuda_graph_replay",
        "gpu",
        correlation,
        {"graph_instance_id": graph_instance_id},
        {"cuda_graph_replay": {"graph_instance_id": graph_instance_id}},
    ):
        yield correlation


def graph_operation_plan(
    rank: int, step_index: int
) -> tuple[tuple[tuple[int, int, str, int], dict[str, Any]], ...]:
    """Return one rank/step slice for graph-capture provenance."""
    _load_operation_plan()
    return tuple(
        (key, item)
        for key, item in sorted(_OPERATION_PLAN.items())
        if key[0] == rank and key[1] == step_index
    )


def _claim_operation_binding(
    instrumentation_site: str,
) -> tuple[OperationCorrelation, str]:
    if _ACTIVE_OPERATION_STEP is None:
        raise RuntimeError("SimFer bound operation requires an active profiled step")
    _load_operation_plan()
    rank = _rank()
    occurrence_key = (rank, _ACTIVE_OPERATION_STEP, instrumentation_site)
    occurrence = _OPERATION_OCCURRENCES.get(occurrence_key, 0)
    key = (*occurrence_key, occurrence)
    try:
        item = _OPERATION_PLAN[key]
    except KeyError:
        raise RuntimeError(
            "SimFer operation plan has no binding for "
            f"rank={rank}, step={_ACTIVE_OPERATION_STEP}, "
            f"site={instrumentation_site!r}, "
            f"occurrence={occurrence}"
        ) from None
    _OPERATION_OCCURRENCES[occurrence_key] = occurrence + 1
    _CLAIMED_OPERATION_BINDINGS.add(key)
    return (
        OperationCorrelation(
            physical_operation_id=item["physical_operation_id"],
            semantic_operation_ids=tuple(item["semantic_operation_ids"]),
            timing_key_id=item["timing_key_id"],
            dependency_correlation_ids=tuple(item["dependency_correlation_ids"]),
        ),
        item["correlation_id"],
    )


def _load_operation_plan() -> None:
    global _OPERATION_PLAN_PATH, _OPERATION_PLAN
    global _OPERATION_OCCURRENCES, _CLAIMED_OPERATION_BINDINGS
    path = Path(os.environ["VLLM_SIMFER_OPERATION_PLAN"]).resolve()
    if _OPERATION_PLAN_PATH == path:
        return
    value = json.loads(path.read_text())
    if set(value) != {
        "id",
        "schema_version",
        "validation_plan_id",
        "expected_capture_id",
        "bindings",
    }:
        raise ValueError("SimFer operation-range plan fields disagree with schema")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported SimFer operation-range plan schema")
    identity = {key: item for key, item in value.items() if key != "id"}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    expected_id = "operation-range-plan-v1:" + hashlib.sha256(encoded).hexdigest()
    if value["id"] != expected_id:
        raise ValueError("SimFer operation-range plan ID does not match content")
    expected_capture = os.environ.get("VLLM_SIMFER_CAPTURE_ID")
    if expected_capture and value.get("expected_capture_id") != expected_capture:
        raise ValueError("SimFer operation plan targets another capture")
    required = {
        "rank",
        "step_index",
        "instrumentation_site",
        "occurrence",
        "correlation_id",
        "physical_operation_id",
        "semantic_operation_ids",
        "timing_key_id",
        "dependency_correlation_ids",
    }
    indexed = {}
    coordinates = []
    correlations = set()
    for item in value.get("bindings", ()):
        if set(item) != required:
            raise ValueError("SimFer operation binding fields disagree with schema")
        key = (
            item["rank"],
            item["step_index"],
            item["instrumentation_site"],
            item["occurrence"],
        )
        numeric_coordinates = (*key[:2], key[3])
        if (
            not all(
                isinstance(coordinate, int) for coordinate in numeric_coordinates
            )
            or any(coordinate < 0 for coordinate in numeric_coordinates)
            or not isinstance(key[2], str)
            or not key[2]
        ):
            raise ValueError("SimFer operation binding coordinates are invalid")
        for field in (
            "correlation_id",
            "physical_operation_id",
            "timing_key_id",
        ):
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError("SimFer operation binding identity is incomplete")
        for field in (
            "semantic_operation_ids",
            "dependency_correlation_ids",
        ):
            values = item[field]
            if (
                not isinstance(values, list)
                or values != sorted(set(values))
                or any(not isinstance(entry, str) or not entry for entry in values)
            ):
                raise ValueError("SimFer operation binding values are not canonical")
        if key in indexed:
            raise ValueError("SimFer operation plan repeats a call-site binding")
        if item["correlation_id"] in correlations:
            raise ValueError("SimFer operation plan repeats a correlation ID")
        indexed[key] = item
        coordinates.append(key)
        correlations.add(item["correlation_id"])
    if coordinates != sorted(coordinates):
        raise ValueError("SimFer operation plan bindings are not canonical")
    _OPERATION_PLAN_PATH = path
    _OPERATION_PLAN = indexed
    _OPERATION_OCCURRENCES = {}
    _CLAIMED_OPERATION_BINDINGS = set()


def _verify_step_operation_bindings(step_index: int | None) -> None:
    if not os.environ.get("VLLM_SIMFER_OPERATION_PLAN"):
        return
    if step_index is None:
        return
    _load_operation_plan()
    rank = _rank()
    if os.environ.get("VLLM_SIMFER_GRAPH_PROVENANCE"):
        from vllm.v1.worker.simfer_graph_provenance import bindings_covered

        if bindings_covered(rank, step_index):
            return
    expected = {
        key for key in _OPERATION_PLAN if key[0] == rank and key[1] == step_index
    }
    missing = expected - _CLAIMED_OPERATION_BINDINGS
    if missing:
        sites = sorted((key[2], key[3]) for key in missing)
        raise RuntimeError(
            f"SimFer operation plan left unclaimed bindings for rank={rank}, "
            f"step={step_index}: {sites}"
        )


@contextmanager
def _record_range(
    name: str,
    resource: str,
    correlation_id: str,
    event_metadata: dict[str, Any],
    marker_metadata: dict[str, Any] | None,
    *,
    repeat_metadata_on_end: bool = False,
) -> Any:
    parent = getattr(_ACTIVE_PHASE, "correlation", None)
    if parent is None:
        parent = _ACTIVE_STEP_CORRELATION
    previous = getattr(_ACTIVE_PHASE, "correlation", None)
    _ACTIVE_PHASE.correlation = correlation_id
    payload = {"name": name, "resource": resource, **event_metadata}
    record_event(
        "phase_start",
        correlation_id=correlation_id,
        parent_correlation_id=parent,
        **payload,
    )
    try:
        with _profiler_marker(
            name,
            correlation_id,
            parent,
            metadata=marker_metadata,
        ):
            yield
    finally:
        end_payload = payload if repeat_metadata_on_end else {
            "name": name,
            "resource": resource,
        }
        record_event(
            "phase_end",
            correlation_id=correlation_id,
            parent_correlation_id=parent,
            **end_payload,
        )
        _ACTIVE_PHASE.correlation = previous


def _begin_step(scheduler_output: Any) -> str:
    global _ACTIVE_STEP, _ACTIVE_STEP_CORRELATION, _NEXT_STEP
    with _LOCK:
        step = _NEXT_STEP
        _NEXT_STEP += 1
    _ACTIVE_STEP = step
    correlation = _new_correlation_id("step")
    _ACTIVE_STEP_CORRELATION = correlation
    record_event(
        "step_start",
        correlation_id=correlation,
        total_num_scheduled_tokens=getattr(
            scheduler_output, "total_num_scheduled_tokens", None
        ),
        boundary="host_api_entry",
    )
    return correlation


def _end_step(
    correlation: str,
    outcome: str,
    *,
    device_completion_observed: bool = False,
) -> None:
    global _ACTIVE_STEP, _ACTIVE_STEP_CORRELATION, _ACTIVE_OPERATION_STEP
    operation_step_index = _ACTIVE_OPERATION_STEP
    binding_error: RuntimeError | None = None
    if outcome == "completed":
        try:
            _verify_step_operation_bindings(operation_step_index)
        except RuntimeError as error:
            binding_error = error
    record_event(
        "step_end",
        correlation_id=correlation,
        outcome=outcome,
        boundary="host_api_return",
        device_completion_observed=device_completion_observed,
        operation_bindings_complete=(
            None if operation_step_index is None else binding_error is None
        ),
    )
    _ACTIVE_STEP = None
    _ACTIVE_STEP_CORRELATION = None
    _ACTIVE_OPERATION_STEP = None
    if binding_error is not None:
        raise binding_error


def _activate_operation_step(profile_path: Path | None) -> None:
    """Bind operation obligations only to an intentionally profiled step.

    Runtime step numbers include unprofiled warmup and selection misses.  A
    range plan instead uses a capture-local coordinate supplied explicitly by
    the probe.  Keeping those namespaces separate prevents warmup count from
    silently changing operation identity.
    """

    global _ACTIVE_OPERATION_STEP
    _ACTIVE_OPERATION_STEP = None
    if profile_path is None or not os.environ.get("VLLM_SIMFER_OPERATION_PLAN"):
        return
    try:
        step_index = int(os.environ.get("VLLM_SIMFER_OPERATION_STEP_INDEX", "0"))
        maximum = int(os.environ.get("VLLM_SIMFER_PROFILE_MAX_STEPS", "1"))
    except ValueError as error:
        raise ValueError("invalid SimFer operation-step selection") from error
    if step_index < 0:
        raise ValueError("VLLM_SIMFER_OPERATION_STEP_INDEX cannot be negative")
    if maximum != 1:
        raise ValueError(
            "operation-bound capture requires VLLM_SIMFER_PROFILE_MAX_STEPS=1"
        )
    _ACTIVE_OPERATION_STEP = step_index


def _claim_profile_path(scheduler_output: Any) -> Path | None:
    """Select a bounded set of exact scheduler steps for device profiling."""

    global _PROFILED_STEPS, _PROFILE_MATCHES
    root_value = os.environ.get("VLLM_SIMFER_PROFILE_DIR")
    if not root_value:
        return None
    raw_counts = os.environ.get("VLLM_SIMFER_PROFILE_TOKEN_COUNTS", "")
    try:
        token_counts = {
            int(value.strip()) for value in raw_counts.split(",") if value.strip()
        }
        maximum = int(os.environ.get("VLLM_SIMFER_PROFILE_MAX_STEPS", "1"))
        skip_matches = int(
            os.environ.get("VLLM_SIMFER_PROFILE_SKIP_MATCHES", "0")
        )
    except ValueError as error:
        raise ValueError("invalid SimFer profiler step selection") from error
    if not token_counts:
        raise ValueError("VLLM_SIMFER_PROFILE_TOKEN_COUNTS cannot be empty")
    if (
        any(value < 0 for value in token_counts)
        or maximum <= 0
        or skip_matches < 0
    ):
        raise ValueError(
            "SimFer profiler token counts must be nonnegative; maximum must "
            "be positive"
        )
    scheduled = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    with _LOCK:
        if scheduled not in token_counts:
            return None
        _PROFILE_MATCHES += 1
        if _PROFILE_MATCHES <= skip_matches or _PROFILED_STEPS >= maximum:
            return None
        _PROFILED_STEPS += 1
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"rank-{_rank()}-step-{_ACTIVE_STEP}.pt.trace.json"


def _rank() -> int:
    """Resolve ranks created internally by vLLM as well as torchrun ranks."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _new_correlation_id(kind: str) -> str:
    global _CORRELATION_INDEX
    rank = _rank()
    with _LOCK:
        index = _CORRELATION_INDEX
        _CORRELATION_INDEX += 1
    return f"rank-{rank}:{kind}-{index}"


@contextmanager
def _profiler_marker(
    name: str,
    correlation_id: str,
    parent_correlation_id: str | None,
    *,
    metadata: dict[str, Any] | None = None,
) -> Any:
    mode = os.environ.get("VLLM_SIMFER_CAPTURE_MARKERS", "torch").lower()
    if mode not in {"none", "torch", "nvtx", "both"}:
        raise ValueError(
            "VLLM_SIMFER_CAPTURE_MARKERS must be none, torch, nvtx, or both"
        )
    rank = _rank()
    payload_value = {
            "correlation_id": correlation_id,
            "name": name,
            "parent_correlation_id": parent_correlation_id,
            "rank": rank,
            "step_index": _ACTIVE_STEP,
        }
    if metadata:
        overlap = set(payload_value).intersection(metadata)
        if overlap:
            raise ValueError(
                "SimFer profiler marker metadata shadows reserved fields: "
                + ", ".join(sorted(overlap))
            )
        payload_value.update(_jsonable(metadata))
    payload = json.dumps(
        payload_value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    marker = "simfer:v1:" + encoded
    with ExitStack() as stack:
        if mode in {"torch", "both"}:
            stack.enter_context(torch.profiler.record_function(marker))
        if mode in {"nvtx", "both"}:
            stack.enter_context(torch.cuda.nvtx.range(marker))
        yield


def record_fx_graph(
    graph: torch.fx.GraphModule,
    *,
    graph_index: int,
    compile_range: Any,
) -> None:
    if not enabled():
        return
    nodes = tuple(
        {
            "name": node.name,
            "op": node.op,
            "target": _target_name(node.target),
            "inputs": tuple(input_node.name for input_node in node.all_input_nodes),
            "output": _value_signature(node.meta.get("example_value")),
        }
        for node in graph.graph.nodes
    )
    record_event(
        "fx_graph",
        graph_index=graph_index,
        compile_range=compile_range,
        nodes=nodes,
    )


def record_model_memory(model: torch.nn.Module, consumed_memory: int) -> None:
    if not enabled():
        return
    tensors = tuple(
        (f"parameter:{name}", tensor)
        for name, tensor in model.named_parameters(remove_duplicate=False)
    ) + tuple(
        (f"buffer:{name}", tensor)
        for name, tensor in model.named_buffers(remove_duplicate=False)
    )
    record_event(
        "model_memory",
        consumed_memory=consumed_memory,
        **_storage_inventory(tensors),
    )


def record_kv_memory(
    kv_caches: list[torch.Tensor],
    kv_cache_config: Any,
) -> None:
    if not enabled():
        return
    tensors = tuple(
        (f"kv_cache:{index}", tensor) for index, tensor in enumerate(kv_caches)
    )
    record_event(
        "kv_memory",
        num_blocks=kv_cache_config.num_blocks,
        cache_groups=tuple(
            {
                "layer_names": tuple(group.layer_names),
                "spec": type(group.kv_cache_spec).__name__,
            }
            for group in kv_cache_config.kv_cache_groups
        ),
        cache_tensors=tuple(
            {
                "size": tensor.size,
                "layers": tuple(tensor.layers),
                "layer_stride": tensor.layer_stride,
                "block_stride": tensor.block_stride,
                "host_resident": tensor.host_resident,
            }
            for tensor in kv_cache_config.kv_cache_tensors
        ),
        **_storage_inventory(tensors),
    )


def record_backend_selection(model_runner: Any) -> None:
    if not enabled():
        return
    attention = []
    for cache_group_index, groups in enumerate(model_runner.attn_groups):
        for attention_group_index, group in enumerate(groups):
            attention.append(
                {
                    "cache_group_index": cache_group_index,
                    "attention_group_index": attention_group_index,
                    "backend": group.backend.full_cls_name(),
                    "layers": tuple(group.layer_names),
                    "kv_cache_spec": type(group.kv_cache_spec).__name__,
                    "metadata_builders": tuple(
                        f"{type(builder).__module__}.{type(builder).__qualname__}"
                        for builder in group.metadata_builders
                    ),
                }
            )
    parallel = model_runner.parallel_config
    compilation = model_runner.compilation_config
    record_event(
        "backend_selection",
        attention=attention,
        all2all_backend=parallel.all2all_backend,
        disable_custom_all_reduce=parallel.disable_custom_all_reduce,
        compilation_backend=compilation.backend,
        compilation_mode=getattr(compilation.mode, "value", str(compilation.mode)),
        cudagraph_mode=getattr(
            compilation.cudagraph_mode,
            "value",
            str(compilation.cudagraph_mode),
        ),
    )


def record_collective(
    kind: str,
    group: Any,
    input_value: Any,
    *,
    dimension: int | None = None,
    sizes: list[int] | None = None,
) -> None:
    if not enabled() or group.world_size == 1:
        return
    communicator = group.device_communicator
    record_event(
        "collective",
        collective_kind=kind,
        group_name=group.unique_name,
        ranks=tuple(group.ranks),
        world_size=group.world_size,
        backend=(
            None
            if communicator is None
            else f"{type(communicator).__module__}.{type(communicator).__qualname__}"
        ),
        all2all_backend=getattr(communicator, "all2all_backend", None),
        input=_value_signature(input_value),
        input_logical_bytes=_value_nbytes(input_value),
        dimension=dimension,
        sizes=sizes,
    )


def _storage_inventory(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> dict[str, object]:
    storage_ids: dict[tuple[str, int, int], int] = {}
    storages: list[dict[str, object]] = []
    tensors: list[dict[str, object]] = []
    for name, tensor in named_tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storage_index = storage_ids.get(key)
        if storage_index is None:
            storage_index = len(storages)
            storage_ids[key] = storage_index
            storages.append(
                {
                    "storage_index": storage_index,
                    "device": str(tensor.device),
                    "bytes": storage.nbytes(),
                }
            )
        tensors.append(
            {
                "name": name,
                "storage_index": storage_index,
                **_tensor_signature(tensor),
            }
        )
    return {
        "storages": storages,
        "tensors": tensors,
        "unique_storage_bytes": sum(int(item["bytes"]) for item in storages),
    }


def _tensor_signature(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "element_size": tensor.element_size(),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "storage_offset": tensor.storage_offset(),
    }


def _value_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, (tuple, list)):
        return tuple(_value_signature(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _value_signature(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _value_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_value_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_value_nbytes(item) for item in value.values())
    return 0


def _capture_provenance() -> dict[str, str]:
    values = {
        field: os.environ.get(environment, "")
        for field, environment in _PROVENANCE_ENV.items()
    }
    missing = [field for field, value in values.items() if not value]
    if missing:
        raise ValueError(
            "SimFer capture requires provenance environment variables for: "
            + ", ".join(missing)
        )
    for field in ("vllm_commit", "model_config_sha256"):
        value = values[field]
        expected_length = 40 if field == "vllm_commit" else 64
        if len(value) != expected_length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(
                f"SimFer capture {field} must be a lowercase "
                f"{expected_length}-character hexadecimal digest"
            )
    return {"source": "instrumented_pinned_vllm", **values}


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(target)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    start = getattr(value, "start", None)
    end = getattr(value, "end", None)
    if isinstance(start, int) and isinstance(end, int):
        return {"start": start, "end": end}
    return str(value)
