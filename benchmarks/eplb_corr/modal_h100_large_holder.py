# /// script
# requires-python = ">=3.12"
# dependencies = ["modal>=1.5.4"]
# ///
"""Retain an eight-H100 worker with the cached DSV4.1 checkpoint mounted."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "eplb-corr-h100-large-20261102"
app = modal.App(APP_NAME)
workspace = modal.Volume.from_name("eplb-corr-h100-20261102")
model_cache = modal.Volume.from_name("vllm-experiment-cache")
image = (
    modal.Image.from_registry(
        "vllm/vllm-openai@"
        "sha256:7db4b696b45eea23757a1649200d81a13049e1bfd29a8ea392e506dc704f4b6a",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("build-essential", "curl", "git", "tar")
)


@app.function(
    image=image,
    gpu="H100!:8",
    cpu=32,
    memory=384 * 1024,
    timeout=24 * 60 * 60,
    volumes={"/workspace": workspace, "/model-cache": model_cache},
)
def hold() -> None:
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 8 or any("H100" not in name for name in names):
        raise RuntimeError(f"Expected eight H100 GPUs, got {names}")
    status = Path("/workspace/status_large.json")
    try:
        status.write_text(json.dumps({"stage": "setup", "state": "running"}) + "\n")
        workspace.commit()
        with Path("/workspace/setup_large.log").open("w") as log:
            subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        "mkdir -p /opt/jonah_vllm && "
                        "tar -xzf /workspace/source.tar.gz -C /opt/jonah_vllm && "
                        "bash /opt/jonah_vllm/benchmarks/eplb_corr/"
                        "setup_h100_worker.sh"
                    ),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        source = Path("/opt/jonah_vllm")
        for override in Path("/workspace/overrides").glob("*.py"):
            target = source / "benchmarks/eplb_corr" / override.name
            target.write_bytes(override.read_bytes())
        system_package = Path("/usr/local/lib/python3.12/dist-packages/vllm")
        for library in system_package.rglob("*.so"):
            target = source / "vllm" / library.relative_to(system_package)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(library)
        subprocess.run(
            [
                str(source / ".venv/bin/python"),
                str(source / "benchmarks/eplb_corr/repair_h100_worker.py"),
            ],
            check=True,
        )
        status.write_text(json.dumps({"stage": "setup", "state": "complete"}) + "\n")
        workspace.commit()
    except Exception as exc:
        status.write_text(
            json.dumps({"stage": "setup", "state": "failed", "error": str(exc)})
            + "\n"
        )
        workspace.commit()
        print(f"Setup failed; holder remains active: {exc}", flush=True)
    while True:
        time.sleep(60)
        workspace.commit()
