#!/usr/bin/env bash
set -euo pipefail

cd /opt/jonah_vllm
export PYTHONPATH=/opt/jonah_vllm${PYTHONPATH:+:$PYTHONPATH}
prepared=/workspace/prepared/h100-large-v3
results=/workspace/results/h100-large-v3
.venv/bin/python benchmarks/eplb_corr/prepare_h100_large_replay.py \
  --capture /workspace/captures/h100-large-v3/routes \
  --placements benchmarks/eplb_corr/data/dep8-h100-weight-sweep \
  --output "$prepared"
.venv/bin/python benchmarks/eplb_corr/replay_captured_deepep.py \
  --input "$prepared" \
  --output /workspace/results/h100-large-v3-quick \
  --buffer-gib 2 \
  --quick --warmups 1 --repeats 1
.venv/bin/python benchmarks/eplb_corr/replay_captured_deepep.py \
  --input "$prepared" \
  --output "$results" \
  --buffer-gib 2 \
  --warmups 2 --repeats 5
.venv/bin/python benchmarks/eplb_corr/analyze_h100_large.py \
  --results "$results" --prepared "$prepared"
