#!/usr/bin/env bash
set -euo pipefail

cd /opt/jonah_vllm
.venv/bin/python benchmarks/eplb_corr/collect_dep8_routes.py \
  --input /workspace/large-input/requests.jsonl \
  --output /workspace/captures/h100-large-v1 \
  --concurrency 8 \
  --wave-size 8 \
  --max-num-batched-tokens 16384 \
  --max-model-len 65536 \
  --attention-config '{"backend":"FLASHMLA_SPARSE_DSV41","indexer_kv_dtype":"fp8","indexer_sparse_logits":false}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --no-prefix-caching \
  --timeout 3600
