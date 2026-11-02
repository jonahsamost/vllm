# SPDX-License-Identifier: Apache-2.0
"""Run paired EPLB variants against one frozen session-trace request set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE / "server.py"
REPLAY = HERE / "replay.py"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
VARIANTS = ("baseline", "default", "corr_aware")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _replay(
    *,
    input_path: Path,
    output_dir: Path,
    port: int,
    concurrency: int,
    skip_requests: int,
    requests: int,
    profile: bool,
    profile_seconds: float,
) -> None:
    command = [
        sys.executable,
        str(REPLAY),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        MODEL,
        "--concurrency",
        str(concurrency),
        "--skip-requests",
        str(skip_requests),
        "--warmup-requests",
        "0",
        "--measure-requests",
        str(requests),
        "--reset-prefix-cache",
    ]
    if profile:
        command.extend(("--profile", "--profile-seconds", str(profile_seconds)))
    _run(command)
    summary = json.loads((output_dir / "summary.json").read_text())
    if summary["measured"]["failed"]:
        raise RuntimeError(
            f"{summary['measured']['failed']} replay requests failed in {output_dir}"
        )


def run(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_files = [
        HERE / "prepare_trace.py",
        REPLAY,
        SERVER,
        Path(__file__),
        HERE.parents[1] / "vllm/distributed/eplb/eplb_state.py",
        HERE.parents[1] / "vllm/distributed/eplb/policy/corr_aware.py",
        HERE.parents[1]
        / "vllm/model_executor/layers/fused_moe/router/corr_aware_eplb_kernel.py",
    ]
    manifest = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "concurrencies": args.concurrencies,
        "variants": args.variants,
        "profile": args.profile,
        "profile_seconds": args.profile_seconds if args.profile else None,
        "warmup_requests": args.warmup_requests,
        "measure_requests": args.measure_requests,
        "source_sha256": {
            str(path.relative_to(HERE.parents[1])): _sha256(path)
            for path in source_files
        },
    }
    (args.output_root / "matrix.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    for variant in args.variants:
        run_dir = args.output_root / variant
        run_dir.mkdir(exist_ok=True)
        server_args = [
            sys.executable,
            str(SERVER),
            "--output-dir",
            str(run_dir),
            "--port",
            str(args.port),
        ]
        started = False
        try:
            start_args = server_args + ["--variant", variant]
            if args.profile:
                start_args.append("--profile")
            _run([*start_args[:2], "start", *start_args[2:]])
            started = True
            _run([*server_args[:2], "wait-ready", *server_args[2:]])

            warmup_dir = run_dir / "warmup"
            _replay(
                input_path=args.input,
                output_dir=warmup_dir,
                port=args.port,
                concurrency=args.warmup_concurrency,
                skip_requests=0,
                requests=args.warmup_requests,
                profile=False,
                profile_seconds=args.profile_seconds,
            )
            log = (run_dir / "server.log").read_text(errors="replace")
            if "Using DeepEPHTAll2AllManager all2all manager" not in log:
                raise RuntimeError("Server never selected DeepEP HT")

            for concurrency in args.concurrencies:
                output_dir = run_dir / f"c{concurrency:03d}"
                _replay(
                    input_path=args.input,
                    output_dir=output_dir,
                    port=args.port,
                    concurrency=concurrency,
                    skip_requests=args.warmup_requests,
                    requests=args.measure_requests,
                    profile=args.profile,
                    profile_seconds=args.profile_seconds,
                )
        finally:
            if started:
                _run([*server_args[:2], "stop", *server_args[2:]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--concurrencies", nargs="+", type=int, required=True)
    parser.add_argument("--port", type=int, default=19080)
    parser.add_argument("--warmup-concurrency", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=400)
    parser.add_argument("--measure-requests", type=int, default=500)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-seconds", type=float, default=10)
    args = parser.parse_args()
    if (
        min(args.concurrencies) < 1
        or args.warmup_requests < 1
        or args.measure_requests < 1
        or args.warmup_concurrency < 1
        or args.profile_seconds <= 0
    ):
        parser.error("Concurrency and request counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
