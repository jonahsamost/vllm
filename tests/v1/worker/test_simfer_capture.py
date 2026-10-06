import json

import torch

from vllm.v1.worker.simfer_capture import (
    record_fx_graph,
    record_model_memory,
)


class TiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 3))
        self.alias = self.weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


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
