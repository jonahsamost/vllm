# SPDX-License-Identifier: Apache-2.0
"""Profile and analyze the EPLB sweep after the unprofiled matrix completes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ("baseline", "default", "corr_aware")


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    return not proc_stat.exists() or proc_stat.read_text().split()[2] != "Z"


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, required=True)
    args = parser.parse_args()

    while _is_running(args.wait_pid):
        time.sleep(30)

    expected = {
        (variant, concurrency)
        for variant in VARIANTS
        for concurrency in args.concurrencies
    }
    found = set()
    for path in args.results_root.glob("*/c*/summary.json"):
        summary = json.loads(path.read_text())
        measured = summary["measured"]
        if measured["failed"] or measured["requests"] != 128:
            raise RuntimeError(f"Invalid throughput run: {path}")
        found.add((path.parent.parent.name, summary["concurrency"]))
    if found != expected:
        raise RuntimeError(
            f"Throughput matrix incomplete: missing={sorted(expected - found)}"
        )

    _run(
        [
            sys.executable,
            str(HERE / "analyze.py"),
            "--results-root",
            str(args.results_root),
            "--output-dir",
            str(args.analysis_dir),
        ]
    )
    _run(
        [
            sys.executable,
            str(HERE / "run_matrix.py"),
            "--input",
            str(args.input),
            "--output-root",
            str(args.profile_root),
            "--concurrencies",
            "32",
            "--warmup-requests",
            "32",
            "--measure-requests",
            "32",
            "--profile",
        ]
    )
    _run(
        [
            sys.executable,
            str(HERE / "analyze_profiles.py"),
            "--profile-root",
            str(args.profile_root),
            "--output",
            str(args.analysis_dir / "deepep_collective_kernels.csv"),
        ]
    )
    print("Throughput sweep, profiler pass, and graphs complete.", flush=True)


if __name__ == "__main__":
    main()
