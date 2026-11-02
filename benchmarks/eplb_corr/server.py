# SPDX-License-Identifier: Apache-2.0
"""Launch one comparable DSV4.1 TEP8 server for EPLB benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SERVED_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
EPLB_CONFIG = {
    "communicator": "nixl",
    "log_balancedness": True,
    "log_balancedness_interval": 100,
    "num_redundant_experts": 8,
    "step_interval": 3000,
    "use_async": True,
    "window_size": 1000,
}
ATTENTION_CONFIG = {
    "backend": "FLASHINFER_MLA_SPARSE_DSV41",
    "indexer_kv_dtype": "mxfp4",
    "indexer_sparse_logits": True,
}


def command(variant: str, port: int, profile_dir: Path | None) -> list[str]:
    args = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tokenizer-mode",
        "deepseek_v41",
        "--tool-call-parser",
        "deepseek_v41",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_v41",
        "--tensor-parallel-size",
        "4",
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--enable-expert-parallel",
        "--all2all-backend",
        "deepep_high_throughput",
        "--kernel-config",
        json.dumps(
            {
                "enable_flashinfer_autotune": True,
                "moe_backend": "marlin",
                "linear_backend": "marlin",
            }
        ),
        "--attention-config",
        json.dumps(ATTENTION_CONFIG),
        "--engram-config",
        json.dumps({"cpu_offload": True}),
        "--kv-cache-dtype",
        "fp8",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "4096",
        "--max-model-len",
        "1048576",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--disable-uvicorn-access-log",
    ]
    if variant != "baseline":
        config = {**EPLB_CONFIG, "policy": variant}
        args.extend(("--enable-eplb", "--eplb-config", json.dumps(config)))
    if profile_dir is not None:
        args.extend(
            (
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": str(profile_dir),
                        "torch_profiler_with_stack": False,
                        "ignore_frontend": True,
                    }
                ),
            )
        )
    return args


def start(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = args.output_dir / "profiles" if args.profile else None
    if profile_dir is not None:
        profile_dir.mkdir(exist_ok=True)
    cmd = command(args.variant, args.port, profile_dir)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": "/model-cache/huggingface",
            "VLLM_USE_RUST_FRONTEND": "0",
            "VLLM_USE_FASTOKENS": "0",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        }
    )
    environment["PATH"] = (
        f"{Path(sys.executable).parent}:/usr/local/cuda/bin:"
        + environment.get("PATH", "")
    )
    (args.output_dir / "launch.json").write_text(
        json.dumps(
            {
                "command": cmd,
                "environment": {
                    key: environment[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "CUDA_HOME",
                        "HF_HOME",
                        "VLLM_USE_RUST_FRONTEND",
                        "VLLM_USE_FASTOKENS",
                        "VLLM_USE_BREAKABLE_CUDAGRAPH",
                        "VLLM_SERVER_DEV_MODE",
                        "VLLM_ENGINE_READY_TIMEOUT_S",
                    )
                },
                "variant": args.variant,
                "profile": args.profile,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with (args.output_dir / "server.log").open("w") as log:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    (args.output_dir / "server.pid").write_text(str(process.pid) + "\n")
    print(process.pid)


def wait_ready(args: argparse.Namespace) -> None:
    pid = int((args.output_dir / "server.pid").read_text())
    url = f"http://127.0.0.1:{args.port}/health"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5)
            if response.status_code == 200:
                print("ready")
                return
        except httpx.HTTPError:
            pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError(f"Server process {pid} exited") from exc
        time.sleep(5)
    raise TimeoutError(f"Server did not become ready within {args.timeout}s")


def stop(args: argparse.Namespace) -> None:
    pid = int((args.output_dir / "server.pid").read_text())
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(2)
    os.killpg(pid, signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "wait-ready", "stop"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("baseline", "default", "corr_aware"),
        default="baseline"
    )
    parser.add_argument("--port", type=int, default=19080)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.action == "start":
        start(args)
    elif args.action == "wait-ready":
        wait_ready(args)
    else:
        stop(args)


if __name__ == "__main__":
    main()
