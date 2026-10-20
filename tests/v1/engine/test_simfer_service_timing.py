import json
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import vllm.v1.engine.core as engine_core
import vllm.v1.engine.simfer_service_timing as service_timing


class SchedulerOutput:
    def __init__(self, tokens: int, requests: int = 2) -> None:
        self.total_num_scheduled_tokens = tokens
        self.num_scheduled_tokens = {str(index): 1 for index in range(requests)}


def _configure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMFER_SERVICE_TIMING_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SIMFER_SERVICE_TIMING_CAPTURE_ID", "capture:test")
    monkeypatch.setenv("VLLM_SIMFER_SERVICE_TIMING_TOKEN_COUNTS", "4")
    monkeypatch.setenv("VLLM_SIMFER_SERVICE_TIMING_MAX_STEPS", "2")
    monkeypatch.setenv("RANK", "3")


def test_records_only_selected_clean_engine_steps(tmp_path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    timestamps = iter((10, 30, 47, 50, 79))
    monkeypatch.setattr(
        service_timing.time, "perf_counter_ns", lambda: next(timestamps)
    )
    recorder = service_timing.ServiceTimingRecorder()

    recorder.arm_iteration(skip_matches=1)
    skipped = recorder.claim(SchedulerOutput(4), started_ns=recorder.begin())
    assert skipped is None
    selected = recorder.claim(SchedulerOutput(4), started_ns=recorder.begin())
    recorder.finish(selected)
    recorder.finish_iteration()
    recorder.arm_iteration(skip_matches=0)
    second = recorder.claim(SchedulerOutput(4), started_ns=recorder.begin())
    recorder.finish(second)
    recorder.finish_iteration()
    events = [
        json.loads(line)
        for line in (tmp_path / "rank-3.jsonl").read_text().splitlines()
    ]
    assert [event["sample_index"] for event in events] == [0, 1]
    assert [event["duration_ns"] for event in events] == [17, 29]
    assert all(event["instrumentation"] == "none" for event in events)
    assert all(event["timer"] == "engine_core_step_perf_counter" for event in events)
    assert all(event["capture_id"] == "capture:test" for event in events)


def test_refuses_to_time_inside_structural_capture(tmp_path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SIMFER_CAPTURE_DIR", str(tmp_path / "capture"))

    with pytest.raises(RuntimeError, match="separate processes"):
        service_timing.ServiceTimingRecorder().begin()


def test_invalid_selection_fails_before_writing(tmp_path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SIMFER_SERVICE_TIMING_TOKEN_COUNTS", "")
    recorder = service_timing.ServiceTimingRecorder()

    with pytest.raises(ValueError, match="selection bounds"):
        recorder.arm_iteration(skip_matches=0)
    assert not tuple(tmp_path.glob("*.jsonl"))


def test_retries_partial_event_writes(tmp_path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    timestamps = iter((10, 30))
    monkeypatch.setattr(
        service_timing.time, "perf_counter_ns", lambda: next(timestamps)
    )
    real_write = service_timing.os.write
    writes = []

    def partial_write(descriptor, value):
        prefix = value[: max(1, len(value) // 2)]
        writes.append(len(prefix))
        return real_write(descriptor, prefix)

    monkeypatch.setattr(service_timing.os, "write", partial_write)
    recorder = service_timing.ServiceTimingRecorder()
    recorder.arm_iteration(skip_matches=0)
    selection = recorder.claim(SchedulerOutput(4), started_ns=recorder.begin())
    recorder.finish(selection)

    event = json.loads((tmp_path / "rank-3.jsonl").read_text())
    assert event["duration_ns"] == 20
    assert len(writes) > 1


def test_iteration_must_emit_exact_selected_shape(tmp_path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    recorder = service_timing.ServiceTimingRecorder()
    recorder.arm_iteration(skip_matches=0)

    assert recorder.claim(SchedulerOutput(8), started_ns=recorder.begin()) is None
    with pytest.raises(RuntimeError, match="no selected"):
        recorder.finish_iteration()


def test_engine_core_step_encloses_schedule_execution_and_reconciliation(
    monkeypatch,
) -> None:
    events = []
    output = SchedulerOutput(4)

    class Recorder:
        def begin(self):
            events.append("begin")
            return 10

        def claim(self, scheduler_output, *, started_ns):
            assert scheduler_output is output
            assert started_ns == 10
            events.append("claim")
            return "selection"

        def finish(self, selection):
            assert selection == "selection"
            events.append("finish")

    class Scheduler:
        def has_requests(self):
            events.append("has-requests")
            return True

        def schedule(self, _throttle):
            events.append("schedule")
            return output

        def get_grammar_bitmask(self, _output):
            return None

        def update_from_output(self, _scheduler_output, _model_output):
            events.append("reconcile")
            return {}

    class Future:
        def result(self):
            events.append("model-result")
            return object()

    class Executor:
        def execute_model(self, _scheduler_output, *, non_block):
            assert non_block is True
            events.append("execute")
            return Future()

    monkeypatch.setattr(engine_core, "_SIMFER_SERVICE_TIMING", Recorder())
    core = SimpleNamespace(
        scheduler=Scheduler(),
        model_executor=Executor(),
        capture_iteration_details=lambda _output: nullcontext(None),
        log_error_detail=lambda _output: nullcontext(),
        _process_aborts_queue=lambda: None,
        _attach_iteration_details=lambda _outputs, _details: None,
        _should_throttle_prefills=lambda: False,
    )

    outputs, executed = engine_core.EngineCore.step(core)

    assert outputs == {}
    assert executed is True
    assert events == [
        "begin",
        "has-requests",
        "schedule",
        "claim",
        "execute",
        "model-result",
        "reconcile",
        "finish",
    ]


def test_engine_core_batch_queue_finishes_the_matching_scheduled_selection(
    monkeypatch,
) -> None:
    events = []
    output = SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"request": 4},
        pending_structured_output_tokens=False,
    )

    class Recorder:
        def begin(self):
            events.append("begin")
            return 10

        def claim(self, scheduler_output, *, started_ns):
            assert scheduler_output is output
            assert started_ns == 10
            events.append("claim")
            return "queued-selection"

        def finish(self, selection):
            assert selection == "queued-selection"
            events.append("finish")

    class Scheduler:
        requests = True

        def has_requests(self):
            return self.requests

        def schedule(self, _throttle):
            events.append("schedule")
            self.requests = False
            return output

        def get_grammar_bitmask(self, _output):
            return None

        def update_from_output(self, scheduler_output, _model_output):
            assert scheduler_output is output
            events.append("reconcile")
            return {}

    class Future:
        def result(self):
            events.append("model-result")
            return object()

    class Executor:
        def execute_model(self, scheduler_output, *, non_block):
            assert scheduler_output is output
            assert non_block is True
            events.append("execute")
            return Future()

        def sample_tokens(self, _grammar_output, *, non_block):
            assert non_block is True
            return Future()

    monkeypatch.setattr(engine_core, "_SIMFER_SERVICE_TIMING", Recorder())
    core = SimpleNamespace(
        batch_queue=deque(maxlen=2),
        batch_queue_size=2,
        scheduler=Scheduler(),
        model_executor=Executor(),
        is_ec_consumer=True,
        is_pooling_model=False,
        check_for_draft_tokens=False,
        capture_iteration_details=lambda _output: nullcontext(None),
        log_error_detail=lambda _output: nullcontext(),
        _process_aborts_queue=lambda: None,
        _attach_iteration_details=lambda _outputs, _details: None,
        _should_throttle_prefills=lambda: False,
    )

    queued_outputs, executed = engine_core.EngineCore.step_with_batch_queue(core)
    assert queued_outputs is None
    assert executed is True
    assert events == ["begin", "schedule", "claim", "execute"]
    assert core.batch_queue[-1][3] == "queued-selection"

    completed_outputs, executed = engine_core.EngineCore.step_with_batch_queue(core)
    assert completed_outputs == {}
    assert executed is False
    assert events == [
        "begin",
        "schedule",
        "claim",
        "execute",
        "model-result",
        "reconcile",
        "finish",
    ]
