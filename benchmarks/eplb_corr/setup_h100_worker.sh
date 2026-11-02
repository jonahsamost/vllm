#!/usr/bin/env bash
set -euo pipefail

source_root=/opt/jonah_vllm
mkdir -p "$source_root/benchmarks/eplb_corr/data"
tar -xzf /workspace/source.tar.gz -C "$source_root"
tar -xzf /workspace/data.tar.gz -C "$source_root/benchmarks/eplb_corr/data"

curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
uv_bin=/root/.local/bin/uv
"$uv_bin" venv --python /usr/bin/python3 --system-site-packages "$source_root/.venv"
"$uv_bin" pip install --python "$source_root/.venv/bin/python" matplotlib orjson

cd "$source_root"
./.venv/bin/python - <<'PY'
import importlib.util
import json
import torch

print(json.dumps({
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "deep_ep_available": importlib.util.find_spec("deep_ep") is not None,
}, indent=2))
PY
