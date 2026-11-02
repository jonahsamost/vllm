# SPDX-License-Identifier: Apache-2.0
"""Summarize GPU durations of DeepEP HT dispatch and combine kernels."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import ijson


def _phase(name: str) -> str | None:
    lower = name.lower()
    if "_eplb_corr_map_and_record_i32_kernel" in lower:
        return "corr_eplb_map_and_record"
    if "_eplb_map_and_record_i32_kernel" in lower:
        return "default_eplb_map_and_record"
    if "deep_ep" not in lower and "deepep" not in lower:
        return None
    if "get_dispatch_layout" in lower:
        return "dispatch_layout"
    if "notify_dispatch" in lower:
        return "dispatch_notify"
    if "notify_combine" in lower:
        return "combine_notify"
    if "dispatch" in lower:
        return "dispatch"
    if "combine" in lower:
        return "combine"
    return None


def _events(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        yield from ijson.items(file, "traceEvents.item")


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(root: Path) -> list[dict]:
    rows = []
    for variant in ("baseline", "default", "corr_aware"):
        trace_paths = sorted((root / variant).rglob("*.pt.trace.json*"))
        if not trace_paths:
            raise FileNotFoundError(f"No profiler traces under {root / variant}")
        timings: dict[tuple[str, str], list[float]] = defaultdict(list)
        for path in trace_paths:
            for event in _events(path):
                if event.get("ph") != "X" or "kernel" not in event.get("cat", ""):
                    continue
                name = event.get("name", "")
                phase = _phase(name)
                if phase is not None:
                    timings[(phase, name)].append(float(event["dur"]))

        by_phase: dict[str, list[float]] = defaultdict(list)
        for (phase, _), durations in timings.items():
            by_phase[phase].extend(durations)
        for phase, durations in sorted(by_phase.items()):
            rows.append(
                {
                    "variant": variant,
                    "phase": phase,
                    "kernel": "ALL_SPECIALIZATIONS",
                    "trace_files": len(trace_paths),
                    "launches": len(durations),
                    "mean_us": statistics.mean(durations),
                    "p50_us": _percentile(durations, 50),
                    "p95_us": _percentile(durations, 95),
                    "total_gpu_ms": sum(durations) / 1000,
                }
            )
        for (phase, name), durations in sorted(timings.items()):
            rows.append(
                {
                    "variant": variant,
                    "phase": phase,
                    "kernel": name,
                    "trace_files": len(trace_paths),
                    "launches": len(durations),
                    "mean_us": statistics.mean(durations),
                    "p50_us": _percentile(durations, 50),
                    "p95_us": _percentile(durations, 95),
                    "total_gpu_ms": sum(durations) / 1000,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.profile_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"kernels": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
