# SPDX-License-Identifier: Apache-2.0
"""Collect logical MoE routes from one frozen-traffic DEP8 DeepEP HT run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from server import SERVED_MODEL, command

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_arg(args: list[str], name: str, value: str) -> None:
    args[args.index(name) + 1] = value


def _wait_ready(process: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode}")
        try:
            if httpx.get(url, timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready within {timeout}s")


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output / "routes"
    trace_dir.mkdir(exist_ok=True)
    marker = trace_dir / "active"
    if marker.exists() or any(trace_dir.glob("rank-*")):
        raise FileExistsError(f"Capture directory is not empty: {trace_dir}")

    input_manifest = json.loads(args.input.with_suffix(".manifest.json").read_text())
    request_count = args.requests or input_manifest["requests"]
    if request_count > input_manifest["requests"]:
        raise ValueError("Requested more rows than the frozen trace contains")
    if _sha256(args.input) != input_manifest["input_sha256"]:
        raise ValueError("Frozen request file does not match its manifest")

    serve_command = command("baseline", args.port, None)
    _set_arg(serve_command, "--tensor-parallel-size", "1")
    _set_arg(serve_command, "--data-parallel-size", "8")
    _set_arg(serve_command, "--data-parallel-size-local", "8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": "/model-cache/huggingface",
            "VLLM_USE_RUST_FRONTEND": "0",
            "VLLM_USE_FASTOKENS": "0",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": str(args.timeout),
            "VLLM_ROUTED_EXPERT_TRACE_DIR": str(trace_dir.resolve()),
        }
    )
    environment["PATH"] = (
        f"{Path(sys.executable).parent}:/usr/local/cuda/bin:"
        + environment.get("PATH", "")
    )
    source_files = [
        HERE / "server.py",
        HERE / "replay.py",
        Path(__file__),
        REPO / "vllm/model_executor/layers/fused_moe/routed_experts_trace.py",
        REPO / "vllm/v1/worker/gpu/model_runner.py",
    ]
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "input_sha256": _sha256(args.input),
                "requests": request_count,
                "concurrency": args.concurrency,
                "command": serve_command,
                "source_sha256": {
                    str(path.relative_to(REPO)): _sha256(path) for path in source_files
                },
            },
            indent=2,
        )
        + "\n"
    )

    process = None
    with (args.output / "server.log").open("w") as log:
        try:
            process = subprocess.Popen(
                serve_command,
                cwd=REPO,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (args.output / "server.pid").write_text(f"{process.pid}\n")
            _wait_ready(process, args.port, args.timeout)
            marker.touch()
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "replay.py"),
                    "--input",
                    str(args.input),
                    "--output-dir",
                    str(args.output / "replay"),
                    "--base-url",
                    f"http://127.0.0.1:{args.port}",
                    "--model",
                    SERVED_MODEL,
                    "--concurrency",
                    str(args.concurrency),
                    "--warmup-requests",
                    "0",
                    "--measure-requests",
                    str(request_count),
                    "--reset-prefix-cache",
                ],
                cwd=REPO,
                check=True,
            )
            replay_summary = json.loads(
                (args.output / "replay" / "summary.json").read_text()
            )
            if replay_summary["measured"]["failed"]:
                raise RuntimeError(
                    f"Replay had {replay_summary['measured']['failed']} failed "
                    "requests"
                )
        finally:
            marker.unlink(missing_ok=True)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    subprocess.run(
        [sys.executable, str(HERE / "validate_routes.py"), str(trace_dir)],
        cwd=REPO,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--port", type=int, default=19081)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
