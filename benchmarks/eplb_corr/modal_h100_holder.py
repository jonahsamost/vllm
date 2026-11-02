# /// script
# requires-python = ">=3.12"
# dependencies = ["modal>=1.5.4"]
# ///
"""Retain one eight-H100 Modal worker for the EPLB placement sweep."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "eplb-corr-h100-sweep-20261102"
VOLUME_NAME = "eplb-corr-h100-20261102"
BASE_IMAGE = (
    "vllm/vllm-openai@"
    "sha256:7db4b696b45eea23757a1649200d81a13049e1bfd29a8ea392e506dc704f4b6a"
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "curl", "git", "tar")
)


@app.function(
    image=image,
    gpu="H100!:8",
    cpu=32,
    memory=128 * 1024,
    timeout=24 * 60 * 60,
    volumes={"/workspace": volume},
)
def hold() -> None:
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 8 or any("H100" not in name for name in names):
        raise RuntimeError(f"Expected eight H100 GPUs, got {names}")
    print(f"Retaining eight H100 GPUs in {APP_NAME}", flush=True)
    status = Path("/workspace/status.json")

    def run_stage(name: str, command: list[str]) -> None:
        status.write_text(json.dumps({"stage": name, "state": "running"}) + "\n")
        volume.commit()
        with (Path("/workspace") / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        status.write_text(json.dumps({"stage": name, "state": "complete"}) + "\n")
        volume.commit()

    try:
        run_stage(
            "setup",
            [
                "bash",
                "-lc",
                (
                    "mkdir -p /opt/jonah_vllm && "
                    "tar -xzf /workspace/source.tar.gz -C /opt/jonah_vllm && "
                    "bash /opt/jonah_vllm/benchmarks/eplb_corr/setup_h100_worker.sh"
                ),
            ],
        )
        run_stage(
            "smoke",
            [
                "/opt/jonah_vllm/.venv/bin/python",
                "/opt/jonah_vllm/benchmarks/eplb_corr/replay_captured_deepep.py",
                "--input",
                "/opt/jonah_vllm/benchmarks/eplb_corr/data/dep8-h100-weight-sweep",
                "--output",
                "/workspace/results/quick",
                "--quick",
                "--warmups",
                "1",
                "--repeats",
                "1",
            ],
        )
        run_stage(
            "sweep",
            ["bash", "/opt/jonah_vllm/benchmarks/eplb_corr/run_h100_weight_sweep.sh"],
        )
    except Exception as exc:
        stage = json.loads(status.read_text())["stage"]
        status.write_text(
            json.dumps({"stage": stage, "state": "failed", "error": str(exc)}) + "\n"
        )
        volume.commit()
        print(f"H100 setup or sweep failed; holder remains active: {exc}", flush=True)
    while True:
        time.sleep(60)
        volume.commit()
