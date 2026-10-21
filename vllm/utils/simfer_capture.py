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


def capture_operation_site(
    instrumentation_site: str,
    name: str,
    *,
    resource: str = "gpu",
) -> Any:
    """Decorate an exact operation boundary with a frozen SimFer binding."""

    if not instrumentation_site or not name or not resource:
        raise ValueError("SimFer operation-site decorator arguments cannot be empty")

    def decorate(function: Any) -> Any:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not os.environ.get("VLLM_SIMFER_CAPTURE_DIR"):
                return function(*args, **kwargs)
            from vllm.v1.worker.simfer_graph_provenance import operation_scope

            with operation_scope(instrumentation_site) as captured:
                if captured:
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
