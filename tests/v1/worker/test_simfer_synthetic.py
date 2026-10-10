import hashlib
import json
from pathlib import Path

import pytest
import torch

from vllm.v1.worker.simfer_synthetic import (
    exact_moe_routing,
    override_speculative_acceptance,
    reset_for_test,
    validate_dummy_loader,
)


def _write_manifest(
    tmp_path: Path, decisions: list[dict], *, world_size: int = 1
) -> Path:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    config = model / "config.json"
    config.write_text(json.dumps({"model_type": "test"}))
    payload = {
        "schema_version": 1,
        "model": "test/model",
        "model_config_path": str(model),
        "model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "vllm_revision": "vllm@test",
        "seed": 7,
        "parallel": {
            "tensor_parallel_size": world_size,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "sequence_parallel": False,
        },
        "steps": [
            {
                "id": "decode",
                "request_count": 2,
                "scheduled_tokens": 2,
                "context_lengths": [8, 8],
                "graph_mode": "eager",
                "graph_bucket_tokens": None,
            }
        ],
        "decisions": decisions,
        "load_format": "dummy",
        "allow_network": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["id"] = "synthetic-v1:" + hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    reset_for_test()
    monkeypatch.delenv("VLLM_SIMFER_SYNTHETIC_MANIFEST", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    yield
    reset_for_test()


def test_dummy_loader_policy_and_exact_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_manifest(
        tmp_path,
        [
            {
                "kind": "moe_routing",
                "ordinal": 0,
                "rank": None,
                "values": {"expert_ids": [[0, 2], [1, 3]]},
            }
        ],
    )
    monkeypatch.setenv("VLLM_SIMFER_SYNTHETIC_MANIFEST", str(path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    validate_dummy_loader("dummy")
    weights, ids = exact_moe_routing(
        num_tokens=2,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        indices_type=torch.int32,
    )
    assert ids.tolist() == [[0, 2], [1, 3]]
    assert weights.tolist() == [[0.5, 0.5], [0.5, 0.5]]
    with pytest.raises(ValueError, match="missing synthetic decision"):
        exact_moe_routing(
            num_tokens=2,
            num_experts=4,
            top_k=2,
            device=torch.device("cpu"),
            indices_type=torch.int32,
        )


def test_exact_acceptance_reconciles_ragged_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_manifest(
        tmp_path,
        [
            {
                "kind": "accepted_draft_tokens",
                "ordinal": 0,
                "rank": None,
                "values": [2, 0],
            }
        ],
    )
    monkeypatch.setenv("VLLM_SIMFER_SYNTHETIC_MANIFEST", str(path))
    sampled = torch.tensor([[10, 11, 12, -1], [20, 21, -1, -1]])
    draft = torch.tensor([1, 2, 3, 4, 5])

    output, lengths = override_speculative_acceptance(
        sampled=sampled,
        draft_sampled=draft,
        cu_num_logits=[0, 4, 6],
        has_logprobs=False,
    )
    assert output.tolist() == [[1, 2, 0, -1], [0, -1, -1, -1]]
    assert lengths.tolist() == [3, 1]


def test_manifest_requires_offline_dummy_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_manifest(tmp_path, [])
    monkeypatch.setenv("VLLM_SIMFER_SYNTHETIC_MANIFEST", str(path))
    with pytest.raises(ValueError, match="load_format=dummy"):
        validate_dummy_loader("safetensors")


def test_compact_generated_routing_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_manifest(
        tmp_path,
        [
            {
                "kind": "moe_routing",
                "ordinal": 0,
                "rank": None,
                "values": {
                    "_repeat": True,
                    "value": {
                        "mode": "skewed",
                        "seed": 3,
                        "hot_experts": 2,
                    },
                },
            }
        ],
    )
    monkeypatch.setenv("VLLM_SIMFER_SYNTHETIC_MANIFEST", str(path))
    _, ids = exact_moe_routing(
        num_tokens=3,
        num_experts=8,
        top_k=2,
        device=torch.device("cpu"),
        indices_type=torch.int64,
    )
    assert ids.tolist() == [[1, 0], [1, 0], [1, 0]]
    repeated_weights, repeated = exact_moe_routing(
        num_tokens=1,
        num_experts=8,
        top_k=2,
        device=torch.device("cpu"),
        indices_type=torch.int64,
    )
    assert repeated.tolist() == [[1, 0]]
    same_weights, same_ids = exact_moe_routing(
        num_tokens=1,
        num_experts=8,
        top_k=2,
        device=torch.device("cpu"),
        indices_type=torch.int64,
    )
    assert same_weights is repeated_weights
    assert same_ids is repeated


def test_synthetic_decisions_use_initialized_distributed_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_manifest(
        tmp_path,
        [
            {
                "kind": "moe_routing",
                "ordinal": 0,
                "rank": 3,
                "values": {"expert_ids": [[2]]},
            }
        ],
        world_size=4,
    )
    monkeypatch.setenv("VLLM_SIMFER_SYNTHETIC_MANIFEST", str(path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    _, ids = exact_moe_routing(
        num_tokens=1,
        num_experts=4,
        top_k=1,
        device=torch.device("cpu"),
        indices_type=torch.int64,
    )
    assert ids.tolist() == [[2]]
