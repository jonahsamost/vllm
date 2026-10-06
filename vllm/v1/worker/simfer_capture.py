# SPDX-License-Identifier: Apache-2.0
"""Opt-in structural evidence for SimFer differential conformance.

This module deliberately records metadata only. It does not synchronize a
device, time kernels, install module hooks, retain tensors, or alter dispatch.
Set ``VLLM_SIMFER_CAPTURE_DIR`` to emit one JSONL stream per worker rank.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


_SCHEMA_VERSION = 1
_LOCK = threading.Lock()
_EVENT_INDEX = 0
_ACTIVE_STEP: int | None = None
_INITIALIZED_STREAMS: set[Path] = set()
_PROVENANCE_ENV = {
    "capture_id": "VLLM_SIMFER_CAPTURE_ID",
    "vllm_commit": "VLLM_SIMFER_VLLM_COMMIT",
    "model_config_sha256": "VLLM_SIMFER_MODEL_CONFIG_SHA256",
}


def enabled() -> bool:
    return bool(os.environ.get("VLLM_SIMFER_CAPTURE_DIR"))


def record_event(kind: str, **payload: Any) -> None:
    if not enabled():
        return
    if not kind:
        raise ValueError("SimFer capture event kind cannot be empty")
    provenance = _capture_provenance()
    global _EVENT_INDEX
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    with _LOCK:
        event_index = _EVENT_INDEX
        _EVENT_INDEX += 1
        path = Path(os.environ["VLLM_SIMFER_CAPTURE_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        stream = path / f"rank-{rank}.jsonl"
        if stream not in _INITIALIZED_STREAMS:
            if stream.exists() and stream.stat().st_size:
                raise ValueError(
                    f"SimFer capture stream already exists and is not empty: {stream}"
                )
            _INITIALIZED_STREAMS.add(stream)
        event = {
            "schema_version": _SCHEMA_VERSION,
            **provenance,
            "event_index": event_index,
            "kind": kind,
            "rank": rank,
            "step_index": _ACTIVE_STEP,
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
    global _ACTIVE_STEP
    _ACTIVE_STEP = 0 if _ACTIVE_STEP is None else _ACTIVE_STEP + 1
    record_event(
        "runtime_dispatch",
        request_ids=tuple(scheduler_output.num_scheduled_tokens),
        scheduled_tokens=tuple(scheduler_output.num_scheduled_tokens.items()),
        actual_num_tokens=scheduler_output.total_num_scheduled_tokens,
        actual_num_requests=num_requests,
        cudagraph_mode=getattr(cudagraph_mode, "value", str(cudagraph_mode)),
        batch_descriptor=batch_descriptor,
        should_ubatch=should_ubatch,
        num_tokens_across_dp=num_tokens_across_dp,
    )


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
