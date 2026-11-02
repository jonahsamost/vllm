# SPDX-License-Identifier: Apache-2.0
"""Summarize matched H100 DeepEP timing across correlation weights."""

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


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(results: Path, prepared: Path) -> None:
    manifest = json.loads((prepared / "manifest.json").read_text())
    run = json.loads((results / "manifest.json").read_text())
    variants = tuple(manifest["variants"])
    if tuple(run["variants"]) != variants:
        raise ValueError("Measured variants differ from the prepared placements")
    expected = len(run["steps"]) * len(run["layers"]) * len(variants) * run["repeats"]

    combined = []
    with ExitStack() as stack:
        readers = [
            csv.DictReader(
                stack.enter_context((results / f"rank-{rank:02d}.csv").open(newline=""))
            )
            for rank in range(manifest["num_ranks"])
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
            sources = sum(int(row["source_tokens"]) for row in rank_rows)
            destinations = np.array(
                [[int(x) for x in row["dest_counts"].split(";")] for row in rank_rows]
            ).sum(axis=0)
            sends = int(destinations.sum())
            combined.append(
                {
                    "step": int(key[0]),
                    "layer": int(key[1]),
                    "variant": key[2],
                    "repetition": int(key[3]),
                    "source_tokens": sources,
                    "fanout": sends / sources,
                    "destination_skew": max(destinations) * len(rank_rows) / sends,
                    "total_ms": max(float(row["total_ms"]) for row in rank_rows),
                }
            )
    if len(combined) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(combined)}")
    _write_csv(results / "combined.csv", combined)

    grouped = defaultdict(list)
    for row in combined:
        grouped[row["step"], row["layer"], row["variant"]].append(row)
    cases = []
    for (step, layer, variant), rows in grouped.items():
        if len(rows) != run["repeats"]:
            raise ValueError(f"Missing repetitions at {step=}, {layer=}, {variant=}")
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
    cases.sort(
        key=lambda row: (row["step"], row["layer"], variants.index(row["variant"]))
    )
    _write_csv(results / "cases.csv", cases)

    primary_steps = set(manifest["strata"]["8k_16k"]) & set(run["steps"])
    primary = [row for row in cases if row["step"] in primary_steps]
    stats = {}
    for variant in variants:
        rows = [row for row in primary if row["variant"] == variant]
        stats[variant] = {
            "mean_ms": float(np.mean([row["total_ms"] for row in rows])),
            "median_fanout": float(np.median([row["fanout"] for row in rows])),
            "median_destination_skew": float(
                np.median([row["destination_skew"] for row in rows])
            ),
        }
    baseline_ms = stats["default"]["mean_ms"]
    for values in stats.values():
        values["time_change_vs_default_pct"] = 100 * (
            values["mean_ms"] / baseline_ms - 1
        )
    winner = min(
        manifest["correlation_weights"], key=lambda name: stats[name]["mean_ms"]
    )
    summary = {
        "primary_step_count": len(primary_steps),
        "primary_case_count": len(primary) // len(variants),
        "secondary_16k_32k_step_count": len(
            set(manifest["strata"]["16k_32k"]) & set(run["steps"])
        ),
        "winner": winner,
        "winner_correlation_weight": manifest["correlation_weights"][winner],
        "variants": stats,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    weights = manifest["correlation_weights"]
    names = sorted(weights, key=lambda name: weights[name])
    x = [weights[name] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, label in (
        (axes[0], "time_change_vs_default_pct", "Mean DeepEP time vs default (%)"),
        (axes[1], "median_fanout", "Destination ranks per token"),
        (axes[2], "median_destination_skew", "Busiest / mean destination"),
    ):
        ax.plot(x, [stats[name][field] for name in names], "o-", color="#168a70")
        ax.axhline(
            stats["default"][field], color="#d97706", linestyle="--", label="Default"
        )
        ax.axhline(
            stats["baseline"][field],
            color="#3973a8",
            linestyle="-.",
            label="No EPLB",
        )
        ax.axhline(
            stats["raw_corr"][field],
            color="#17212b",
            linestyle=":",
            label="Original score",
        )
        ax.set_xlabel("Correlation weight")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        f"H100 captured-route sweep: {len(primary_steps)} forwards at 8k–16k tokens"
    )
    fig.tight_layout()
    fig.savefig(results / "weight_sweep.svg")
    fig.savefig(results / "weight_sweep.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.prepared)


if __name__ == "__main__":
    main()
