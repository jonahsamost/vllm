# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pybase64 as base64
import pytest
import torch

import vllm.v1.worker.simfer_capture as simfer_capture
import vllm.v1.worker.simfer_graph_provenance as graph_provenance
from vllm.utils.simfer_capture import capture_operation_site
from vllm.v1.worker.simfer_capture import (
    OperationCorrelation,
    capture_step,
    record_event,
    record_fx_graph,
    record_model_memory,
    record_operation,
    record_phase,
    record_step_phase,
)


def _operation_plan(path: Path, capture_id: str) -> Path:
    binding = {
        "rank": 0,
        "step_index": 0,
        "instrumentation_site": "glm.kda",
        "occurrence": 0,
        "correlation_id": "simfer-operation-v1:one",
        "physical_operation_id": "physical:kda:0",
        "semantic_operation_ids": ["semantic:kda:0"],
        "timing_key_id": "measurement-key-v3:kda",
        "dependency_correlation_ids": [],
    }
    identity = {
        "schema_version": 1,
        "validation_plan_id": "validation-plan-v1:test",
        "expected_capture_id": capture_id,
        "bindings": [binding],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    value = {
        "id": "operation-range-plan-v1:" + hashlib.sha256(encoded).hexdigest(),
        **identity,
    }
    path.write_text(json.dumps(value))
    return path


def test_graph_provenance_selects_nodes_between_event_boundaries() -> None:
    nodes = [
        {"index": 0, "dependencies": []},
        {"index": 1, "dependencies": [0]},
        {"index": 2, "dependencies": [1]},
        {"index": 3, "dependencies": [2]},
        {"index": 4, "dependencies": [3]},
    ]

    assert graph_provenance._nodes_between(nodes, 1, 4) == {2, 3}


def test_shared_linear_boundaries_publish_stable_simfer_site_metadata() -> None:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )

    for linear_type in (
        ReplicatedLinear,
        ColumnParallelLinear,
        RowParallelLinear,
        GateLinear,
    ):
        assert linear_type.forward.__simfer_instrumentation_site__ == "physical.linear"
        assert linear_type.forward.__simfer_operation_name__ == "linear"


def test_graph_provenance_excludes_unrelated_branch() -> None:
    nodes = [
        {"index": 0, "dependencies": []},
        {"index": 1, "dependencies": [0]},
        {"index": 2, "dependencies": [1]},
        {"index": 3, "dependencies": [0]},
        {"index": 4, "dependencies": [2, 3]},
    ]

    assert graph_provenance._nodes_between(nodes, 1, 4) == {2}


def test_graph_provenance_requires_capture_output(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMFER_GRAPH_PROVENANCE", "1")
    monkeypatch.delenv("VLLM_SIMFER_CAPTURE_DIR", raising=False)

    assert not graph_provenance.enabled()

    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", "/capture")
    assert graph_provenance.enabled()


def test_graph_provenance_serializes_cupti_graph_node_id() -> None:
    node = {
        "index": 7,
        "node_type": "kernel",
        "graph_id": 11,
        "node_id": 7,
        "dependencies": [6],
    }

    record = graph_provenance._node_record(node, None)

    assert record["index"] == 7
    assert record["graph_node_id"] == (11 << 32) | 7


def test_operation_decorator_inserts_boundaries_during_compilation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_GRAPH_PROVENANCE", "1")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    calls: list[tuple[Any, ...]] = []
    boundary = object()

    def begin(site, anchor):
        calls.append(("begin", site, anchor))
        return boundary

    def end(value, anchor):
        calls.append(("end", value, anchor))

    monkeypatch.setattr(graph_provenance, "begin_compiled_operation", begin)
    monkeypatch.setattr(graph_provenance, "end_compiled_operation", end)

    @capture_operation_site("glm.kda", "kda")
    def kda(value):
        calls.append(("operation", value))
        return value + 1

    value = torch.ones(1)
    result = kda(value)

    assert torch.equal(result, torch.full((1,), 2.0))
    assert calls[0] == ("begin", "glm.kda", value)
    assert calls[1] == ("operation", value)
    assert calls[2][0:2] == ("end", boundary)
    assert torch.equal(calls[2][2], result)


def test_operation_decorator_coalesces_nested_compiled_same_site(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_GRAPH_PROVENANCE", "1")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    calls: list[tuple[Any, ...]] = []
    boundary = object()

    def begin(site, anchor):
        calls.append(("begin", site, anchor))
        return boundary

    def end(value, anchor):
        calls.append(("end", value, anchor))

    monkeypatch.setattr(graph_provenance, "begin_compiled_operation", begin)
    monkeypatch.setattr(graph_provenance, "end_compiled_operation", end)

    @capture_operation_site("physical.linear", "linear")
    def inner(value):
        calls.append(("inner", value))
        return value + 1

    @capture_operation_site("physical.linear", "linear")
    def outer(value):
        calls.append(("outer", value))
        return inner(value)

    value = torch.ones(1)
    result = outer(value)

    assert torch.equal(result, torch.full((1,), 2.0))
    assert [call[0] for call in calls] == ["begin", "outer", "inner", "end"]


def test_compiled_boundaries_require_both_event_nodes(monkeypatch) -> None:
    begin = type("Event", (), {"cuda_event": 10})()
    end = type("Event", (), {"cuda_event": 20})()
    boundary = graph_provenance._OperationBoundary(
        correlation_id="correlation",
        operation_correlation={},
        coordinate=(0, 0, "glm.kda", 0),
        begin=begin,
        end=end,
    )
    monkeypatch.setattr(graph_provenance, "_COMPILED_BOUNDARIES", [boundary])

    assert graph_provenance._compiled_operations({10: 1, 20: 3}) == [boundary]
    with pytest.raises(RuntimeError, match="boundary is incomplete"):
        graph_provenance._compiled_operations({10: 1})


def test_graph_provenance_coalesces_nested_same_site(monkeypatch) -> None:
    recorded = []
    event = type("Event", (), {"record": lambda self: recorded.append(self)})
    boundary = graph_provenance._OperationBoundary(
        correlation_id="correlation",
        operation_correlation={},
        coordinate=(0, 0, "glm.kda", 0),
        begin=event(),
        end=event(),
    )
    capture = type("Capture", (), {"operations": []})()
    monkeypatch.setattr(graph_provenance._ACTIVE, "capture", capture, raising=False)
    monkeypatch.setattr(
        graph_provenance, "_new_operation_boundary", lambda _site: boundary
    )

    with (
        graph_provenance.operation_scope("glm.kda") as outer,
        graph_provenance.operation_scope("glm.kda") as inner,
    ):
        assert outer and inner

    assert recorded == [boundary.begin, boundary.end]
    assert capture.operations == [boundary]


def _enable_fake_profile(tmp_path: Path, monkeypatch) -> None:
    class FakeProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def export_chrome_trace(self, path):
            Path(path).write_text('{"traceEvents": []}')

    monkeypatch.setenv("VLLM_SIMFER_PROFILE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_TOKEN_COUNTS", "4")
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_MAX_STEPS", "1")
    monkeypatch.setattr(simfer_capture, "_PROFILED_STEPS", 0)
    monkeypatch.setattr(simfer_capture, "_PROFILE_MATCHES", 0)
    monkeypatch.setattr(simfer_capture, "_ACTIVE_OPERATION_STEP", None)
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: FakeProfiler())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


class TiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 3))
        self.alias = self.weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


def test_capture_uses_initialized_distributed_rank(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-rank")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    record_event("rank_probe")

    event = json.loads((tmp_path / "rank-3.jsonl").read_text())
    assert event["rank"] == 3
    assert event["kind"] == "rank_probe"


def test_capture_deduplicates_storage_and_records_fx_graph(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-1")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    model = TiedModel()

    record_model_memory(model, consumed_memory=123)
    record_fx_graph(
        torch.fx.symbolic_trace(model),
        graph_index=0,
        compile_range=type("Range", (), {"start": 1, "end": 8})(),
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "rank-0.jsonl").read_text().splitlines()
    ]
    memory, graph = events
    assert memory["capture_id"] == "capture-1"
    assert graph["vllm_commit"] == "a" * 40
    assert memory["kind"] == "model_memory"
    assert memory["payload"]["unique_storage_bytes"] == 24
    assert len(memory["payload"]["storages"]) == 1
    assert len(memory["payload"]["tensors"]) == 2
    assert graph["kind"] == "fx_graph"
    assert graph["payload"]["compile_range"] == {"start": 1, "end": 8}
    assert any(node["op"] == "call_function" for node in graph["payload"]["nodes"])


def test_capture_records_correlated_host_boundaries(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-2")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    markers = []

    @contextmanager
    def record_function(name):
        markers.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record_function)

    class SchedulerOutput:
        total_num_scheduled_tokens = 4

    @capture_step
    def execute(_self, _scheduler_output):
        with record_phase("model_forward"):
            return "done"

    assert execute(None, SchedulerOutput()) == "done"
    events = [
        json.loads(line)
        for line in (tmp_path / "rank-0.jsonl").read_text().splitlines()
    ]

    assert [event["kind"] for event in events] == [
        "step_start",
        "phase_start",
        "phase_end",
        "step_end",
    ]
    assert all(event["schema_version"] == 2 for event in events)
    assert events[0]["correlation_id"] == events[-1]["correlation_id"]
    assert events[1]["parent_correlation_id"] == events[0]["correlation_id"]
    assert [event["timestamp_ns"] for event in events] == sorted(
        event["timestamp_ns"] for event in events
    )
    assert events[-1]["payload"]["device_completion_observed"] is False
    assert len(markers) == 2
    marker_payloads = [
        json.loads(
            base64.urlsafe_b64decode(
                name.removeprefix("simfer:v1:")
                + "=" * (-len(name.removeprefix("simfer:v1:")) % 4)
            )
        )
        for name in markers
    ]
    assert marker_payloads[0]["name"] == "worker_step"
    assert marker_payloads[1]["name"] == "model_forward"
    assert marker_payloads[1]["parent_correlation_id"] == events[0]["correlation_id"]


def test_capture_emits_exact_operation_identity_to_stream_and_marker(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-operation")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    markers = []

    @contextmanager
    def record_function(name):
        markers.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record_function)

    class SchedulerOutput:
        total_num_scheduled_tokens = 4

    binding = OperationCorrelation(
        physical_operation_id="physical:attention:0",
        semantic_operation_ids=("semantic:attention:0",),
        timing_key_id="measurement-key-v3:attention",
        dependency_correlation_ids=(),
    )

    @capture_step
    def execute(_self, _scheduler_output):
        with record_operation(
            "attention",
            binding,
            correlation_id="operation:attention:0",
            resource="gpu:compute",
        ):
            return "done"

    assert execute(None, SchedulerOutput()) == "done"
    events = [
        json.loads(line)
        for line in (tmp_path / "rank-0.jsonl").read_text().splitlines()
    ]
    operation_events = events[1:3]
    assert [event["kind"] for event in operation_events] == [
        "phase_start",
        "phase_end",
    ]
    assert (
        operation_events[0]["correlation_id"] == operation_events[1]["correlation_id"]
    )
    expected = {
        **binding.to_dict(),
        "semantic_operation_ids": ["semantic:attention:0"],
        "dependency_correlation_ids": [],
    }
    assert operation_events[0]["payload"]["operation_correlation"] == expected

    marker = next(value for value in markers if value != markers[0])
    encoded = marker.removeprefix("simfer:v1:")
    payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert payload["correlation_id"] == operation_events[0]["correlation_id"]
    assert payload["operation_correlation"] == expected


def test_operation_correlation_rejects_noncanonical_dependencies() -> None:
    with pytest.raises(ValueError, match="canonical strings"):
        OperationCorrelation(
            physical_operation_id="physical:attention:0",
            semantic_operation_ids=("semantic:attention:0",),
            timing_key_id="measurement-key-v3:attention",
            dependency_correlation_ids=("second", "first"),
        )


def test_operation_range_requires_active_worker_step(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    binding = OperationCorrelation(
        physical_operation_id="physical:attention:0",
        semantic_operation_ids=("semantic:attention:0",),
        timing_key_id="measurement-key-v3:attention",
    )

    with (
        pytest.raises(RuntimeError, match="active worker step"),
        record_operation(
            "attention",
            binding,
            correlation_id="operation:attention:0",
            resource="gpu:compute",
        ),
    ):
        pass


def test_step_phase_ignores_capture_lifecycle_work_outside_worker_step(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))

    with record_step_phase("model_forward", resource="gpu") as correlation:
        assert correlation is None

    assert list(tmp_path.iterdir()) == []


def test_warmup_sample_tokens_uses_only_lifecycle_aware_phases() -> None:
    """Every nested range in vLLM's warmup entry point must tolerate no step."""
    source = (
        Path(simfer_capture.__file__).parent / "gpu" / "model_runner.py"
    ).read_text()
    module = ast.parse(source)
    sample_tokens = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "sample_tokens"
    )
    strict_phase_calls = [
        node
        for node in ast.walk(sample_tokens)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_phase"
    ]

    assert strict_phase_calls == []


def test_step_phase_records_work_inside_worker_step(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-step-phase")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)

    class SchedulerOutput:
        total_num_scheduled_tokens = 4

    @capture_step
    def run_step(_worker, _scheduler_output):
        with record_step_phase("model_forward", resource="gpu") as correlation:
            assert correlation is not None

    run_step(None, SchedulerOutput())

    events = [
        json.loads(line)
        for line in (tmp_path / "rank-0.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "step_start",
        "phase_start",
        "phase_end",
        "step_end",
    ]


def test_bound_operation_claims_exact_plan_entry_and_closes_step(
    tmp_path, monkeypatch
) -> None:
    capture = tmp_path / "capture"
    plan = _operation_plan(tmp_path / "operation-plan.json", "capture-bound")
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(capture))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-bound")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    monkeypatch.setenv("VLLM_SIMFER_OPERATION_PLAN", str(plan))
    monkeypatch.setattr(simfer_capture, "_NEXT_STEP", 0)
    monkeypatch.setattr(simfer_capture, "_OPERATION_PLAN_PATH", None)
    monkeypatch.setattr(simfer_capture, "_OPERATION_PLAN", {})
    monkeypatch.setattr(simfer_capture, "_OPERATION_OCCURRENCES", {})
    monkeypatch.setattr(simfer_capture, "_CLAIMED_OPERATION_BINDINGS", set())
    _enable_fake_profile(tmp_path, monkeypatch)

    class SchedulerOutput:
        total_num_scheduled_tokens = 4

    @capture_operation_site("glm.kda", "kda")
    def kda():
        return "done"

    @capture_step
    def execute(_self, _scheduler_output):
        return kda()

    assert execute(None, SchedulerOutput()) == "done"
    events = [
        json.loads(line) for line in (capture / "rank-0.jsonl").read_text().splitlines()
    ]
    assert events[1]["payload"]["operation_correlation"] == {
        "schema_version": 1,
        "physical_operation_id": "physical:kda:0",
        "semantic_operation_ids": ["semantic:kda:0"],
        "timing_key_id": "measurement-key-v3:kda",
        "dependency_correlation_ids": [],
    }
    assert events[-1]["payload"]["operation_bindings_complete"] is True


def test_bound_operation_fails_step_when_exact_binding_is_unclaimed(
    tmp_path, monkeypatch
) -> None:
    capture = tmp_path / "capture"
    plan = _operation_plan(tmp_path / "operation-plan.json", "capture-missing")
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(capture))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-missing")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    monkeypatch.setenv("VLLM_SIMFER_OPERATION_PLAN", str(plan))
    monkeypatch.setattr(simfer_capture, "_NEXT_STEP", 0)
    monkeypatch.setattr(simfer_capture, "_OPERATION_PLAN_PATH", None)
    monkeypatch.setattr(simfer_capture, "_OPERATION_PLAN", {})
    monkeypatch.setattr(simfer_capture, "_OPERATION_OCCURRENCES", {})
    monkeypatch.setattr(simfer_capture, "_CLAIMED_OPERATION_BINDINGS", set())
    _enable_fake_profile(tmp_path, monkeypatch)

    class SchedulerOutput:
        total_num_scheduled_tokens = 4

    @capture_step
    def execute(_self, _scheduler_output):
        return "done"

    with pytest.raises(RuntimeError, match="unclaimed bindings"):
        execute(None, SchedulerOutput())


def test_bound_operation_ignores_unprofiled_match_then_uses_local_step_zero(
    tmp_path, monkeypatch
) -> None:
    capture = tmp_path / "capture"
    plan = _operation_plan(tmp_path / "operation-plan.json", "capture-selected")
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(capture))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-selected")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    monkeypatch.setenv("VLLM_SIMFER_OPERATION_PLAN", str(plan))
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_SKIP_MATCHES", "1")
    monkeypatch.setattr(simfer_capture, "_NEXT_STEP", 0)
    monkeypatch.setattr(simfer_capture, "_OPERATION_PLAN_PATH", None)
    monkeypatch.setattr(simfer_capture, "_OPERATION_PLAN", {})
    monkeypatch.setattr(simfer_capture, "_OPERATION_OCCURRENCES", {})
    monkeypatch.setattr(simfer_capture, "_CLAIMED_OPERATION_BINDINGS", set())
    _enable_fake_profile(tmp_path, monkeypatch)

    class SchedulerOutput:
        total_num_scheduled_tokens = 4

    @capture_operation_site("glm.kda", "kda")
    def kda():
        return "done"

    @capture_step
    def execute(_self, _scheduler_output):
        return kda()

    assert execute(None, SchedulerOutput()) == "done"
    assert execute(None, SchedulerOutput()) == "done"
    events = [
        json.loads(line) for line in (capture / "rank-0.jsonl").read_text().splitlines()
    ]
    operation_starts = [
        event
        for event in events
        if event["kind"] == "phase_start" and event["payload"]["name"] == "kda"
    ]
    assert len(operation_starts) == 1
    assert operation_starts[0]["step_index"] == 1
    assert operation_starts[0]["correlation_id"] == "simfer-operation-v1:one"
    step_ends = [event for event in events if event["kind"] == "step_end"]
    assert step_ends[0]["payload"]["operation_bindings_complete"] is None
    assert step_ends[1]["payload"]["operation_bindings_complete"] is True


def test_capture_profiles_only_selected_step_and_observes_device_completion(
    tmp_path, monkeypatch
) -> None:
    capture = tmp_path / "capture"
    traces = tmp_path / "traces"
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(capture))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-profile")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_DIR", str(traces))
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_TOKEN_COUNTS", "8")
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_MAX_STEPS", "1")
    monkeypatch.setattr(simfer_capture, "_PROFILED_STEPS", 0)
    monkeypatch.setattr(simfer_capture, "_PROFILE_MATCHES", 0)
    synchronized = []

    class FakeProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def export_chrome_trace(self, path):
            Path(path).write_text('{"traceEvents": []}')

    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: FakeProfiler())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(True))

    class SchedulerOutput:
        total_num_scheduled_tokens = 8

    @capture_step
    def execute(_self, _scheduler_output):
        return "done"

    assert execute(None, SchedulerOutput()) == "done"
    events = [
        json.loads(line) for line in (capture / "rank-0.jsonl").read_text().splitlines()
    ]

    assert synchronized == [True]
    assert [event["kind"] for event in events] == [
        "step_start",
        "profiler_artifact",
        "step_end",
    ]
    assert events[-1]["payload"]["device_completion_observed"] is True
    assert len(tuple(traces.glob("rank-0-step-*.pt.trace.json"))) == 1


def test_capture_can_skip_matching_warmup_step(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path / "capture"))
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_ID", "capture-warmup")
    monkeypatch.setenv("VLLM_SIMFER_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("VLLM_SIMFER_MODEL_CONFIG_SHA256", "b" * 64)
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_TOKEN_COUNTS", "8")
    monkeypatch.setenv("VLLM_SIMFER_PROFILE_SKIP_MATCHES", "1")
    monkeypatch.setattr(simfer_capture, "_PROFILED_STEPS", 0)
    monkeypatch.setattr(simfer_capture, "_PROFILE_MATCHES", 0)

    class FakeProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def export_chrome_trace(self, path):
            Path(path).write_text('{"traceEvents": []}')

    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: FakeProfiler())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class SchedulerOutput:
        total_num_scheduled_tokens = 8

    @capture_step
    def execute(_self, _scheduler_output):
        return "done"

    execute(None, SchedulerOutput())
    execute(None, SchedulerOutput())

    assert len(tuple((tmp_path / "traces").glob("*.pt.trace.json"))) == 1
