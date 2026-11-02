# Captured-route DeepEP placement replay

Eight B300 GPUs replayed the same DSV4.1 logical top-k routes under three expert
placements: no EPLB (384 physical experts), default EPLB (392), and correlation-aware
EPLB (392). Dummy BF16 activations were 5,120 elements wide. Each case timed
`get_dispatch_layout`, `dispatch`, and `combine` with CUDA synchronization; the
reported time is the slowest rank's wall time. Expert computation was not included.

Both EPLB placements were trained on captured forwards 2,000–2,999. Evaluation used
32 later forwards from each of five traffic strata and eight spaced routed layers,
giving 1,280 matched step/layer cases. Every case had two warmups and ten timed
repetitions per placement. The replica hash was checked against vLLM's CUDA router.
See [preparation_manifest.json](preparation_manifest.json) and
[manifest.json](manifest.json) for the exact selected steps and settings.

## Results

Negative percentages mean lower DeepEP time. Each entry below is the median of the
matched case-level percentage changes within that stratum.

| Traffic stratum | Default vs no EPLB | Corr aware vs no EPLB | Corr aware vs default |
| --- | ---: | ---: | ---: |
| Decode | +5.06% | +4.34% | -0.72% |
| Medium volume | -0.07% | -1.17% | -1.61% |
| High volume | -3.51% | -4.77% | -1.70% |
| 1–3 idle source ranks | +5.42% | +4.84% | -0.70% |
| 4–7 idle source ranks | +5.69% | +4.72% | -1.01% |

For high-volume cases, median destination-rank fan-out was 4.44 with no EPLB, 4.46
with default EPLB, and 4.11 with correlation-aware EPLB. Median busiest-to-mean
destination load was 1.26, 1.09, and 1.14 respectively. Correlation-aware placement
reduced fan-out while retaining most of default EPLB's load balance.

Weighting the sampled strata by their frequency among the 10,253 eligible later
forwards gives an estimated mean communication time of 0.484 ms without EPLB,
0.478 ms with default EPLB, and 0.471 ms with correlation-aware EPLB. That is a
1.50% reduction for correlation-aware versus default. A stratified resampling of
the selected steps gives a descriptive 95% interval of 1.17%–1.81% lower time;
the sampling was deterministic and adjacent model steps are not independent.

## Figures and data

- [Paired effects by traffic stratum](paired_effects.svg)
- [Percent DeepEP time change by source volume, baseline skew, fan-out, and idle ranks](captured_sweeps.svg)
- [Matched fan-out change versus DeepEP time change](fanout_time_change.svg)
- [Paired change heatmaps by source volume and baseline destination skew](captured_heatmaps.svg)
- [Per-case paired data](paired.csv), [per-repetition combined data](combined.csv),
  and [machine-readable summary](summary.json)

The comparison uses one fixed placement per policy, trained at the first rebalance
window. It is a communication-only counterfactual on routes captured without EPLB.
It excludes expert computation, policy computation, and expert weight migration.
The no-EPLB placement has eight fewer physical experts than the two EPLB placements;
the correlation-aware versus default comparison holds physical expert count fixed.
