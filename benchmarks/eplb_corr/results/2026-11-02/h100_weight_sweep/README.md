# H100 captured-route correlation-weight sweep

Eight H100 GPUs replayed the same captured DEP8 logical top-k routes for each placement. The placement training window was forwards 2000–2999. Evaluation used 210 later forwards with 8k–16k total source tokens across eight ranks, across eight routed layers. One 16k–32k forward was measured but excluded from the primary aggregate. Each case used two warmups and five timed repetitions; `cases.csv` contains the median repetition time. `summary.json` and `weight_sweep.svg` summarize the results.

The timed path is DeepEP `get_dispatch_layout` → `dispatch` → `combine` with dummy BF16 activations. It does not include expert GEMM or an end-to-end vLLM engine. `total_ms` is the maximum synchronized wall time across the eight ranks.

| Placement | Mean DeepEP time (ms) | Change vs default | Median destination ranks/token | Median busiest/mean destination |
| --- | ---: | ---: | ---: | ---: |
| No EPLB | 1.5470 | +2.42% | 4.444 | 1.247 |
| Default EPLB | 1.5105 | reference | 4.456 | 1.075 |
| Original correlation score | 1.5146 | +0.27% | 4.113 | 1.125 |
| Normalized λ=0.3 | 1.5112 | +0.05% | 4.216 | 1.112 |
| Normalized λ=0.6 | 1.6187 | +7.16% | 3.572 | 1.841 |
| Normalized λ=1.0 | 1.6702 | +10.58% | 3.685 | 1.940 |

λ=0.3 was the fastest *weighted* placement, but it did not improve on default EPLB. A bootstrap over the 210 forwards put its paired mean difference from default at 0.0007 ms, with a 95% interval of approximately −0.0030 to +0.0044 ms. High correlation weights lowered fan-out while creating substantial destination skew; DeepEP time increased. This run supports keeping destination balance in the placement objective. It does not establish a best weight for larger forwards because the capture had only one forward above 16k tokens.

The retained Modal app is `eplb-corr-h100-sweep-20261102`, with results on volume `eplb-corr-h100-20261102` at `/results/h100-weight-sweep`. The holder was left running after the experiment.
