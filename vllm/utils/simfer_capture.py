# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level, opt-in SimFer instrumentation hooks.

Model and provider modules must not import the v1 worker package while their
classes are being defined. This bridge keeps that dependency lazy: when
capture is disabled the wrapper performs no worker import, and when enabled it
enters the worker-owned correlation context at the exact call boundary.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import wraps
from typing import Any, Protocol, cast

import torch


class _InstrumentedCallable(Protocol):
    __simfer_instrumentation_site__: str
    __simfer_operation_name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    values: Iterable[Any]
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        raise ValueError("SimFer compiled operation has no tensor boundary")
    for item in values:
        try:
            return _first_tensor(item)
        except ValueError:
            pass
    raise ValueError("SimFer compiled operation has no tensor boundary")


def capture_operation_site(
    instrumentation_site: str,
    name: str,
    *,
    resource: str = "gpu",
    compile_boundary: bool = True,
    eager_range: bool = True,
) -> Any:
    """Decorate an exact operation boundary with a frozen SimFer binding."""
    if not instrumentation_site or not name or not resource:
        raise ValueError("SimFer operation-site decorator arguments cannot be empty")

    def decorate(function: Any) -> Any:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not os.environ.get("VLLM_SIMFER_CAPTURE_DIR"):
                return function(*args, **kwargs)
            from vllm.v1.worker.simfer_graph_provenance import enabled

            compiling = torch.compiler.is_compiling()
            if os.environ.get("VLLM_SIMFER_DEBUG_PROVENANCE"):
                from vllm.v1.worker.simfer_graph_provenance import (
                    debug_operation_site,
                )

                debug_operation_site(instrumentation_site, name, compiling=compiling)

            if compile_boundary and enabled() and compiling:
                from vllm.v1.worker.simfer_graph_provenance import (
                    begin_compiled_operation,
                    compiled_operation_scope,
                    end_compiled_operation,
                )

                with compiled_operation_scope(instrumentation_site) as publish:
                    if not publish:
                        return function(*args, **kwargs)
                    boundary = begin_compiled_operation(
                        instrumentation_site, _first_tensor((args, kwargs))
                    )
                    result = function(*args, **kwargs)
                    end_compiled_operation(boundary, _first_tensor(result))
                    return result

            from vllm.v1.worker.simfer_graph_provenance import operation_scope

            with operation_scope(instrumentation_site) as captured:
                if captured:
                    return function(*args, **kwargs)
            if not eager_range:
                return function(*args, **kwargs)
            from vllm.v1.worker.simfer_capture import record_bound_operation

            with record_bound_operation(
                instrumentation_site,
                name,
                resource=resource,
            ):
                return function(*args, **kwargs)

        instrumented = cast(_InstrumentedCallable, wrapped)
        instrumented.__simfer_instrumentation_site__ = instrumentation_site
        instrumented.__simfer_operation_name__ = name
        return instrumented

    return decorate


__all__ = ["capture_operation_site"]
