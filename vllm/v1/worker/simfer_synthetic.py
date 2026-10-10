# SPDX-License-Identifier: Apache-2.0
"""Strict profiling-only inputs for synthetic-weight SimFer captures.

This module intentionally has no dependency on the private SimFer package. The
standalone probe writes the versioned manifest, and pinned vLLM consumes it at
the places where data-dependent decisions become physical work.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


_ENV = "VLLM_SIMFER_SYNTHETIC_MANIFEST"
_SCHEMA_VERSION = 1
_lock = threading.Lock()
_store: _DecisionStore | None = None
_loaded_path: str | None = None


class _DecisionStore:
    def __init__(self, manifest: dict[str, Any], rank: int):
        self.manifest = manifest
        self.rank = rank
        self._records: dict[tuple[int | None, str, int], Any] = {}
        self._cursors: dict[str, int] = {}
        self.tensor_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        for record in manifest.get("decisions", ()):
            key = (record.get("rank"), record["kind"], int(record["ordinal"]))
            if key in self._records:
                raise ValueError(f"duplicate synthetic decision {key}")
            self._records[key] = record["values"]

    def has(self, kind: str) -> bool:
        return any(
            record_kind == kind and rank in {None, self.rank}
            for rank, record_kind, _ in self._records
        )

    def next(self, kind: str) -> Any | None:
        if not self.has(kind):
            return None
        ordinal = self._cursors.get(kind, 0)
        rank_key = (self.rank, kind, ordinal)
        global_key = (None, kind, ordinal)
        repeating = self._records.get((self.rank, kind, 0))
        if repeating is None:
            repeating = self._records.get((None, kind, 0))
        if (
            isinstance(repeating, dict)
            and repeating.get("_repeat") is True
            and set(repeating) == {"_repeat", "value"}
        ):
            return repeating["value"]
        if rank_key in self._records:
            value = self._records[rank_key]
        elif global_key in self._records:
            value = self._records[global_key]
        else:
            raise ValueError(
                f"missing synthetic decision rank={self.rank} "
                f"kind={kind!r} ordinal={ordinal}"
            )
        self._cursors[kind] = ordinal + 1
        return value


def synthetic_manifest_enabled() -> bool:
    return bool(os.environ.get(_ENV))


def synthetic_decisions_enabled(kind: str) -> bool:
    store = _decision_store()
    return store is not None and store.has(kind)


def validate_dummy_loader(load_format: Any) -> None:
    """Prove a requested synthetic run cannot enter a checkpoint loader."""

    store = _decision_store()
    if store is None:
        return
    value = getattr(load_format, "value", load_format)
    if str(value).lower() != "dummy":
        raise ValueError("SimFer synthetic execution requires load_format=dummy")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("SimFer synthetic execution requires HF_HUB_OFFLINE=1")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("SimFer synthetic execution requires TRANSFORMERS_OFFLINE=1")

    manifest = store.manifest
    model_path = Path(manifest["model_config_path"])
    if not model_path.is_absolute():
        assert _loaded_path is not None
        model_path = Path(_loaded_path).resolve().parent / model_path
    config = model_path / "config.json"
    if not config.is_file():
        raise ValueError("synthetic model config.json is unavailable")
    if _sha256(config) != manifest["model_config_sha256"]:
        raise ValueError("synthetic model config digest changed after planning")


def exact_moe_routing(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: Any,
    indices_type: Any,
) -> tuple[Any, Any] | None:
    store = _decision_store()
    if store is None:
        return None
    value = store.next("moe_routing")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("MoE routing decision must be an object")
    cache_key = (
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        num_tokens,
        num_experts,
        top_k,
        str(device),
        str(indices_type),
    )
    cached = store.tensor_cache.get(cache_key)
    if cached is not None:
        return cached
    expert_ids = value.get("expert_ids")
    weights = value.get("weights")
    expected = (num_tokens, top_k)
    if expert_ids is None and value.get("mode") in {"balanced", "skewed"}:
        expert_ids = _generated_expert_ids(
            mode=value["mode"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            seed=int(value.get("seed", 0)),
            hot_experts=int(value.get("hot_experts", top_k)),
        )
    if _shape(expert_ids) != expected:
        raise ValueError(
            f"MoE routing IDs have shape {_shape(expert_ids)}, expected {expected}"
        )
    flattened = tuple(expert for row in expert_ids for expert in row)
    if any(
        isinstance(expert, bool)
        or not isinstance(expert, int)
        or expert < 0
        or expert >= num_experts
        for expert in flattened
    ):
        raise ValueError("MoE routing contains an out-of-range expert")
    if any(len(set(row)) != len(row) for row in expert_ids):
        raise ValueError("MoE routing repeats an expert for one token")
    if weights is None:
        weights = [[1.0 / top_k] * top_k for _ in range(num_tokens)]
    if _shape(weights) != expected:
        raise ValueError("MoE routing weights do not match expert IDs")

    import torch

    result = (
        torch.tensor(weights, dtype=torch.float32, device=device),
        torch.tensor(expert_ids, dtype=indices_type, device=device),
    )
    store.tensor_cache[cache_key] = result
    return result


def override_moe_routing(
    topk_weights: Any, topk_ids: Any, num_experts: int
) -> tuple[Any, Any] | None:
    """Replace routing outputs after the native router has executed."""

    return exact_moe_routing(
        num_tokens=topk_ids.shape[0],
        num_experts=num_experts,
        top_k=topk_ids.shape[1],
        device=topk_ids.device,
        indices_type=topk_ids.dtype,
    )


def override_speculative_acceptance(
    *,
    sampled: Any,
    draft_sampled: Any,
    cu_num_logits: Any,
    has_logprobs: bool,
) -> tuple[Any, Any] | None:
    """Force exact accepted draft lengths while retaining real verification work."""

    store = _decision_store()
    if store is None:
        return None
    values = store.next("accepted_draft_tokens")
    if values is None:
        return None
    if has_logprobs:
        raise ValueError(
            "exact synthetic acceptance does not support returned logprobs"
        )
    if not isinstance(values, list) or len(values) != sampled.shape[0]:
        raise ValueError("one accepted draft length is required per request")

    import torch

    boundaries = [int(value) for value in cu_num_logits]
    available_by_request = [
        boundaries[index + 1] - boundaries[index] - 1
        for index in range(len(values))
    ]
    for accepted, available in zip(values, available_by_request, strict=True):
        if (
            isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or accepted < 0
            or accepted > available
        ):
            raise ValueError(
                f"accepted draft length {accepted!r} is outside [0, {available}]"
            )
    accepted = torch.tensor(values, dtype=torch.long, device=sampled.device)
    starts = torch.tensor(
        boundaries[:-1], dtype=torch.long, device=sampled.device
    )
    positions = torch.arange(sampled.shape[1], device=sampled.device)
    gather = starts[:, None] + positions[None, :]
    gather = gather.clamp_max(max(0, draft_sampled.numel() - 1))
    draft_rows = draft_sampled[gather]
    output = torch.full_like(sampled, -1)
    output = torch.where(positions[None, :] < accepted[:, None], draft_rows, output)
    output = torch.where(
        positions[None, :] == accepted[:, None], torch.zeros_like(output), output
    )
    num_sampled = (accepted + 1).to(torch.int32)
    return output, num_sampled


def reset_for_test() -> None:
    global _loaded_path, _store
    with _lock:
        _loaded_path = None
        _store = None


def _rank() -> int:
    """Resolve ranks created internally by vLLM as well as torchrun ranks."""

    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except ImportError:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _decision_store() -> _DecisionStore | None:
    global _loaded_path, _store
    path = os.environ.get(_ENV)
    if not path:
        return None
    with _lock:
        if _store is not None and _loaded_path == path:
            return _store
        manifest = json.loads(Path(path).read_text())
        _validate_manifest(manifest)
        rank = _rank()
        world_size = int(manifest["parallel"]["tensor_parallel_size"]) * int(
            manifest["parallel"]["data_parallel_size"]
        )
        if rank < 0 or rank >= world_size:
            raise ValueError("runtime rank is outside synthetic manifest world")
        _store = _DecisionStore(manifest, rank)
        _loaded_path = path
        return _store


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported SimFer synthetic manifest schema")
    if manifest.get("load_format") != "dummy" or manifest.get("allow_network"):
        raise ValueError("unsafe SimFer synthetic execution policy")
    identity = dict(manifest)
    manifest_id = identity.pop("id", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    expected = "synthetic-v1:" + hashlib.sha256(encoded).hexdigest()
    if manifest_id != expected:
        raise ValueError("SimFer synthetic manifest content ID is invalid")


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shapes = {_shape(item) for item in value}
    if len(child_shapes) != 1:
        return (-1,)
    return (len(value), *child_shapes.pop())


def _generated_expert_ids(
    *,
    mode: str,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int,
    hot_experts: int,
) -> list[list[int]]:
    if top_k > num_experts:
        raise ValueError("MoE top-k exceeds the expert domain")
    domain = num_experts
    if mode == "skewed":
        if hot_experts < top_k or hot_experts > num_experts:
            raise ValueError("skewed routing hot_experts is outside valid bounds")
        domain = hot_experts
    return [
        [((token * top_k) + offset + seed) % domain for offset in range(top_k)]
        for token in range(num_tokens)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
