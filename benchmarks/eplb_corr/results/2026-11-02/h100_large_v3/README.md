# H100 DEP8 large-route replay

Full DeepSeek-V4.1-Flash-1017 inference on eight H100s captured logical top-k
routes with DeepEP HT and no EPLB. The server used a 16,384-token per-rank
batched-token cap, and the capture reached 16,384 tokens on one rank and
131,072 tokens across all ranks in one synchronized forward. All 40 MoE layers
were captured and validated. Of 160 fixed production prompts, 153 succeeded;
seven exceeded the configured 65,536-token context limit and were rejected.

The replay uses eight selected layers (0, 5, ..., 35), the same captured routes
and dummy activations for every placement, and the fixed placement files from
the preceding H100 weight sweep. The default and correlation-aware placements
each have eight redundant experts; the no-EPLB placement has none. Timings are
synchronized wall time for DeepEP layout, dispatch, and combine, taking the
slowest rank for each case. They exclude expert computation and server overhead.

| Total source tokens per forward | Forwards | No EPLB | Default EPLB | Corr aware λ=0.3 | Default vs no EPLB | Corr vs no EPLB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8k–16k | 10 | 3.042 ms | 2.944 ms | 2.956 ms | −3.22% | −2.81% |
| 16k–32k | 11 | 4.772 ms | 4.524 ms | 4.492 ms | −5.20% | −5.88% |
| 32k–64k | 6 | 4.761 ms | 4.461 ms | 4.479 ms | −6.30% | −5.94% |
| 64k–128k | 13 | 6.066 ms | 5.694 ms | 5.732 ms | −6.13% | −5.51% |

At 64k–128k, correlation-aware λ=0.3 was 0.66% slower than default EPLB.
A forward-level paired bootstrap gave an approximate 95% interval of
0.09%–1.26% for that difference. This experiment does not show a growing
advantage for the correlation-aware placement at larger token counts.

- [Graph](h100-large-v3/large_route_sweep.svg)
- [Band summary](h100-large-v3/summary.json)
- [Paired bootstrap](h100-large-v3/paired_bootstrap.json)
- [Per-case measurements](h100-large-v3/cases.csv)
- Full route corpus: `benchmarks/eplb_corr/data/dep8-h100-large-v3/routes`
- Fixed input prompts: `benchmarks/eplb_corr/data/dep8-h100-large/input/requests.jsonl`
