# SPDX-License-Identifier: Apache-2.0
"""Compare three placements by actual total tokens in a DEP8 route capture."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze(results: Path, prepared: Path) -> None:
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    run = json.loads((results / "manifest.json").read_text())
    variants = tuple(prepared_manifest["variants"])
    if tuple(run["variants"]) != variants:
        raise ValueError("Measured variants differ from prepared placements")

    grouped: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    with ExitStack() as stack:
        readers = [
            csv.DictReader(
                stack.enter_context((results / f"rank-{rank:02d}.csv").open())
            )
            for rank in range(prepared_manifest["num_ranks"])
        ]
        for rank_rows in zip(*readers, strict=True):
            key = tuple(
                rank_rows[0][part]
                for part in ("step", "layer", "variant", "repetition")
            )
            if any(
                tuple(row[part] for part in ("step", "layer", "variant", "repetition"))
                != key
                for row in rank_rows[1:]
            ):
                raise ValueError(f"Rank rows are misaligned at {key}")
            source = sum(int(row["source_tokens"]) for row in rank_rows)
            destination = np.array(
                [[int(x) for x in row["dest_counts"].split(";")] for row in rank_rows]
            ).sum(axis=0)
            sends = int(destination.sum())
            grouped[int(key[0]), int(key[1]), key[2]].append(
                {
                    "source_tokens": source,
                    "fanout": sends / source,
                    "destination_skew": max(destination) * len(rank_rows) / sends,
                    "total_ms": max(float(row["total_ms"]) for row in rank_rows),
                }
            )
    expected = (
        len(run["steps"])
        * len(run["layers"])
        * len(variants)
        * run["repeats"]
    )
    if sum(map(len, grouped.values())) != expected:
        raise ValueError("Rank result count does not match run manifest")

    cases = []
    for (step, layer, variant), rows in grouped.items():
        if len(rows) != run["repeats"]:
            raise ValueError(f"Missing repetitions for {step=}, {layer=}, {variant=}")
        cases.append(
            {
                "step": step,
                "layer": layer,
                "variant": variant,
                "source_tokens": rows[0]["source_tokens"],
                "fanout": rows[0]["fanout"],
                "destination_skew": rows[0]["destination_skew"],
                "total_ms": float(np.median([row["total_ms"] for row in rows])),
            }
        )
    cases.sort(key=lambda row: (row["step"], row["layer"], row["variant"]))
    with (results / "cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    bins = [(name, set(steps)) for name, steps in prepared_manifest["strata"].items()]
    summary = {}
    for name, steps in bins:
        rows = [row for row in cases if row["step"] in steps]
        if not rows:
            summary[name] = {"steps": 0, "variants": {}}
            continue
        values = {}
        for variant in variants:
            subset = [row for row in rows if row["variant"] == variant]
            values[variant] = {
                "mean_ms": float(np.mean([row["total_ms"] for row in subset])),
                "median_fanout": float(np.median([row["fanout"] for row in subset])),
                "median_destination_skew": float(
                    np.median([row["destination_skew"] for row in subset])
                ),
            }
        baseline = values["baseline"]["mean_ms"]
        for variant in variants:
            values[variant]["time_change_vs_no_eplb_pct"] = 100 * (
                values[variant]["mean_ms"] / baseline - 1
            )
        summary[name] = {
            "steps": len(steps),
            "source_tokens_min": min(row["source_tokens"] for row in rows),
            "source_tokens_max": max(row["source_tokens"] for row in rows),
            "variants": values,
        }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Bootstrap whole forwards, keeping their eight measured layers together.
    # Layer timings within one forward are not independent observations.
    step_means = defaultdict(lambda: defaultdict(list))
    for row in cases:
        step_means[row["step"]][row["variant"]].append(row["total_ms"])
    rng = np.random.default_rng(47)
    paired = {}
    for name, steps in bins:
        if not steps:
            continue
        values = np.array(
            [
                [
                    np.mean(step_means[step][variant])
                    for variant in ("baseline", "default", "w03")
                ]
                for step in sorted(steps)
            ]
        )
        sample_indices = rng.integers(0, len(values), size=(20_000, len(values)))
        bootstrap_means = values[sample_indices].mean(axis=1)
        means = values.mean(axis=0)
        corr_vs_default = 100 * (
            bootstrap_means[:, 2] / bootstrap_means[:, 1] - 1
        )
        paired[name] = {
            "steps": len(steps),
            "corr_vs_default_pct": float(100 * (means[2] / means[1] - 1)),
            "corr_vs_default_ci95": [
                float(value) for value in np.quantile(corr_vs_default, [0.025, 0.975])
            ],
        }
    (results / "paired_bootstrap.json").write_text(
        json.dumps(paired, indent=2) + "\n"
    )

    available = [name for name, _ in bins if summary[name]["steps"]]
    positions = np.arange(len(available))
    labels = {
        "baseline": "No EPLB",
        "default": "Default EPLB",
        "w03": "Corr aware λ=0.3",
    }
    colors = {"baseline": "#3973a8", "default": "#d97706", "w03": "#168a70"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, field, title in (
        (axes[0], "time_change_vs_no_eplb_pct", "DeepEP time vs no EPLB (%)"),
        (axes[1], "median_fanout", "Destination ranks per token"),
        (axes[2], "median_destination_skew", "Busiest / mean destination"),
    ):
        for variant in variants:
            ax.plot(
                positions,
                [summary[name]["variants"][variant][field] for name in available],
                "o-",
                color=colors[variant],
                label=labels[variant],
            )
        ax.set_xticks(
            positions,
            [
                f"{name.replace('_', '–')}\n(n={summary[name]['steps']})"
                for name in available
            ],
        )
        ax.set_xlabel("Actual total source tokens per forward")
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].axhline(0, color="#17212b", linewidth=0.8)
    fig.suptitle("H100 DEP8 captured routes: matched placement replay")
    fig.tight_layout()
    fig.savefig(results / "large_route_sweep.svg")
    fig.savefig(results / "large_route_sweep.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.prepared)


if __name__ == "__main__":
    main()
