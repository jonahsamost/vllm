#!/usr/bin/env bash
set -euo pipefail

cd /opt/jonah_vllm
prepared=benchmarks/eplb_corr/data/dep8-h100-weight-sweep
results=/workspace/results/h100-weight-sweep
mkdir -p "$results"

./.venv/bin/python benchmarks/eplb_corr/replay_captured_deepep.py \
    --input "$prepared" \
    --output "$results" \
    --num-processes 8 \
    --warmups 2 \
    --repeats 5

./.venv/bin/python benchmarks/eplb_corr/analyze_h100_weight_sweep.py \
    --results "$results" \
    --prepared "$prepared"
