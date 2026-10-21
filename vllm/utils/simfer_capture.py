# SPDX-License-Identifier: Apache-2.0
"""Low-level, opt-in SimFer instrumentation hooks.

Model and provider modules must not import the v1 worker package while their
classes are being defined. This bridge keeps that dependency lazy: when
capture is disabled the wrapper performs no worker import, and when enabled it
enters the worker-owned correlation context at the exact call boundary.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any

import torch


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
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

            if enabled():
                from vllm.compilation.breakable_cudagraph import (
                    BreakableCUDAGraphCapture,
                )

                breakable_capture = BreakableCUDAGraphCapture.current()
                if (
                    breakable_capture is not None
                    and not breakable_capture._capturing
                ):
                    breakable_capture.bind_eager_operation(instrumentation_site)

            if compile_boundary and enabled() and torch.compiler.is_compiling():
                from vllm.v1.worker.simfer_graph_provenance import (
                    begin_compiled_operation,
                    end_compiled_operation,
                )

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

        return wrapped

    return decorate


__all__ = ["capture_operation_site"]
