# SPDX-License-Identifier: Apache-2.0
"""Aggregate matched DeepEP route replays and plot placement comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ("baseline", "default", "corr_aware")
COLORS = {"baseline": "#17212b", "default": "#d97706", "corr_aware": "#168a70"}
STRATA = ("decode", "medium", "high", "zero_1_to_3", "zero_4_to_7")
STRATUM_LABELS = ("Decode", "Medium", "High", "1–3 idle", "4–7 idle")
METRICS = ("layout_ms", "dispatch_ms", "combine_ms", "total_ms")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(input_dir: Path, prepared: Path) -> tuple[list[dict], list[dict]]:
    manifest = json.loads((prepared / "manifest.json").read_text())
    run = json.loads((input_dir / "manifest.json").read_text())
    step_stratum = {
        step: name for name, steps in manifest["strata"].items() for step in steps
    }
    files = [
        (input_dir / f"rank-{rank:02d}.csv").open(newline="")
        for rank in range(manifest["num_ranks"])
    ]
    combined = []
    try:
        readers = [csv.DictReader(file) for file in files]
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
                raise ValueError(f"Rank replay rows are not aligned at {key}")
            source = [int(row["source_tokens"]) for row in rank_rows]
            destinations = np.array(
                [
                    [int(count) for count in row["dest_counts"].split(";")]
                    for row in rank_rows
                ]
            ).sum(axis=0)
            sends = int(destinations.sum())
            total_source = sum(source)
            if sends <= 0 or total_source <= 0:
                raise ValueError(f"No routed tokens at {key}")
            row = {
                "step": int(key[0]),
                "layer": int(key[1]),
                "variant": key[2],
                "repetition": int(key[3]),
                "stratum": step_stratum[int(key[0])],
                "source_tokens": total_source,
                "zero_source_ranks": source.count(0),
                "max_source_tokens": max(source),
                "dest_counts": ";".join(map(str, destinations)),
                "total_token_sends": sends,
                "mean_fanout": sends / total_source,
                "dest_max_over_mean": max(destinations) * len(rank_rows) / sends,
            }
            row.update(
                {
                    field: max(float(rank_row[field]) for rank_row in rank_rows)
                    for field in METRICS[:-1]
                }
            )
            row["total_ms"] = max(float(rank_row["total_ms"]) for rank_row in rank_rows)
            combined.append(row)
    finally:
        for file in files:
            file.close()
    expected = len(run["steps"]) * len(run["layers"]) * len(VARIANTS) * run["repeats"]
    if len(combined) != expected:
        raise ValueError(f"Expected {expected} matched rows, found {len(combined)}")
    _write_csv(input_dir / "combined.csv", combined)

    groups = defaultdict(list)
    for row in combined:
        groups[(row["step"], row["layer"], row["variant"])].append(row)
    cases = []
    for key, records in groups.items():
        if len(records) != run["repeats"]:
            raise ValueError(f"Missing repetitions at {key}")
        row = {
            field: records[0][field]
            for field in (
                "step",
                "layer",
                "variant",
                "stratum",
                "source_tokens",
                "zero_source_ranks",
                "max_source_tokens",
                "total_token_sends",
                "mean_fanout",
                "dest_max_over_mean",
            )
        }
        row.update(
            {field: float(np.median([r[field] for r in records])) for field in METRICS}
        )
        cases.append(row)
    cases.sort(
        key=lambda row: (row["step"], row["layer"], VARIANTS.index(row["variant"]))
    )
    _write_csv(input_dir / "cases.csv", cases)

    by_case = defaultdict(dict)
    for row in cases:
        by_case[row["step"], row["layer"]][row["variant"]] = row
    paired = []
    for (step, layer), variants in by_case.items():
        if set(variants) != set(VARIANTS):
            raise ValueError(f"Missing placement at {step=}, {layer=}")
        baseline, default, corr = (variants[name] for name in VARIANTS)
        paired.append(
            {
                "step": step,
                "layer": layer,
                "stratum": baseline["stratum"],
                "source_tokens": baseline["source_tokens"],
                "zero_source_ranks": baseline["zero_source_ranks"],
                "baseline_ms": baseline["total_ms"],
                "default_ms": default["total_ms"],
                "corr_aware_ms": corr["total_ms"],
                "default_vs_baseline_pct": 100
                * (default["total_ms"] / baseline["total_ms"] - 1),
                "corr_vs_baseline_pct": 100
                * (corr["total_ms"] / baseline["total_ms"] - 1),
                "corr_vs_default_pct": 100
                * (corr["total_ms"] / default["total_ms"] - 1),
                "baseline_fanout": baseline["mean_fanout"],
                "default_fanout": default["mean_fanout"],
                "corr_fanout": corr["mean_fanout"],
                "baseline_skew": baseline["dest_max_over_mean"],
                "default_skew": default["dest_max_over_mean"],
                "corr_skew": corr["dest_max_over_mean"],
            }
        )
    _write_csv(input_dir / "paired.csv", paired)
    return cases, paired


def _source_bin(value: float) -> str:
    if value <= 64:
        return "≤64"
    if value <= 1024:
        return "65–1k"
    if value <= 4096:
        return "1k–4k"
    if value <= 8192:
        return "4k–8k"
    return ">8k"


SOURCE_BINS = ("≤64", "65–1k", "1k–4k", "4k–8k", ">8k")


def _quantile_edges(cases: list[dict], field: str) -> np.ndarray:
    values = np.array([row[field] for row in cases])
    return np.unique(np.quantile(values, np.linspace(0, 1, 6)))


def _bin_number(value: float, edges: np.ndarray) -> int:
    return min(int(np.searchsorted(edges[1:], value, side="right")), len(edges) - 2)


def plot_sweeps(paired: list[dict], output: Path) -> None:
    skew_edges = _quantile_edges(paired, "baseline_skew")
    fanout_edges = _quantile_edges(paired, "baseline_fanout")
    panels = (
        ("source_tokens", "Total source tokens", SOURCE_BINS),
        ("baseline_skew", "Baseline busiest / mean destination", skew_edges),
        ("baseline_fanout", "Baseline destination ranks per token", fanout_edges),
        ("zero_source_ranks", "Source ranks with zero tokens", range(8)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plotted_values = [0.0]
    for ax, (field, label, bins) in zip(axes.flat, panels):
        for variant, legend, metric, marker, style in (
            ("baseline", "No EPLB (0%)", None, "o", "-"),
            ("default", "Default EPLB", "default_vs_baseline_pct", "s", "--"),
            ("corr_aware", "Corr aware EPLB", "corr_vs_baseline_pct", "^", ":"),
        ):
            grouped = defaultdict(list)
            for row in paired:
                if field == "source_tokens":
                    bin_id = SOURCE_BINS.index(_source_bin(row[field]))
                elif field == "zero_source_ranks":
                    bin_id = row[field]
                else:
                    bin_id = _bin_number(row[field], bins)
                grouped[bin_id].append(0.0 if metric is None else row[metric])
            x = sorted(i for i, values in grouped.items() if len(values) >= 8)
            medians = [float(np.median(grouped[i])) for i in x]
            plotted_values.extend(medians)
            ax.plot(
                x,
                medians,
                marker=marker,
                linestyle=style,
                linewidth=2,
                color=COLORS[variant],
                label=legend,
            )
        ax.set_xticks(
            range(
                len(bins)
                if field in ("source_tokens", "zero_source_ranks")
                else len(bins) - 1
            )
        )
        if field == "source_tokens" or field == "zero_source_ranks":
            ax.set_xticklabels(bins)
        else:
            ax.set_xticklabels(
                [f"{bins[i]:.2f}–{bins[i + 1]:.2f}" for i in range(len(bins) - 1)],
                rotation=25,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Median time change vs no EPLB (%)")
        ax.grid(alpha=0.2)
    lower = np.floor(min(plotted_values) - 1)
    upper = np.ceil(max(plotted_values) + 1)
    for ax in axes.flat:
        ax.set_ylim(lower, upper)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3)
    fig.suptitle("Captured routes: DeepEP time change by expert placement", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def plot_heatmaps(paired: list[dict], output: Path) -> None:
    skew_edges = _quantile_edges(paired, "baseline_skew")
    comparisons = (
        ("default_vs_baseline_pct", "Default vs baseline"),
        ("corr_vs_baseline_pct", "Corr aware vs baseline"),
        ("corr_vs_default_pct", "Corr aware vs default"),
    )
    arrays = []
    for metric, _ in comparisons:
        cells = defaultdict(list)
        for row in paired:
            x = SOURCE_BINS.index(_source_bin(row["source_tokens"]))
            y = _bin_number(row["baseline_skew"], skew_edges)
            cells[y, x].append(row[metric])
        arrays.append(
            np.array(
                [
                    [
                        np.median(cells[y, x]) if len(cells[y, x]) >= 8 else np.nan
                        for x in range(len(SOURCE_BINS))
                    ]
                    for y in range(len(skew_edges) - 1)
                ]
            )
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    scale = float(
        np.nanpercentile(np.abs(np.concatenate([a.ravel() for a in arrays])), 95)
    )
    for ax, (_, title), values in zip(axes, comparisons, arrays):
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
        )
        ax.set_title(title)
        ax.set_xticks(range(len(SOURCE_BINS)), SOURCE_BINS, rotation=30)
        ax.set_yticks(
            range(len(skew_edges) - 1),
            [
                f"{skew_edges[i]:.2f}–{skew_edges[i + 1]:.2f}"
                for i in range(len(skew_edges) - 1)
            ],
        )
        ax.set_xlabel("Total source tokens")
    axes[0].set_ylabel("Baseline busiest / mean destination")
    fig.colorbar(image, ax=axes, label="Median matched change (%)")
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def plot_paired(paired: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    comparisons = (
        (
            ("default_vs_baseline_pct", "Default vs baseline", COLORS["default"]),
            ("corr_vs_baseline_pct", "Corr aware vs baseline", COLORS["corr_aware"]),
        ),
        (("corr_vs_default_pct", "Corr aware vs default", COLORS["corr_aware"]),),
    )
    for ax, specs in zip(axes, comparisons):
        for offset, (field, label, color) in enumerate(specs):
            x = np.arange(len(STRATA)) + (offset - (len(specs) - 1) / 2) * 0.12
            groups = [
                [row[field] for row in paired if row["stratum"] == s] for s in STRATA
            ]
            median = np.array([np.median(group) for group in groups])
            low = np.array([np.percentile(group, 25) for group in groups])
            high = np.array([np.percentile(group, 75) for group in groups])
            ax.errorbar(
                x,
                median,
                yerr=[median - low, high - median],
                marker="o",
                capsize=3,
                color=color,
                label=label,
            )
        ax.axhline(0, color="#666666", linewidth=1)
        ax.set_ylabel("Matched latency change (%)")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[-1].set_xticks(range(len(STRATA)), STRATUM_LABELS)
    fig.suptitle("Same captured step and layer, different expert placement")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def plot_fanout_change(paired: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax, stratum, label in zip(axes.flat, STRATA, STRATUM_LABELS):
        rows = [row for row in paired if row["stratum"] == stratum]
        fanout = np.array([row["corr_fanout"] - row["default_fanout"] for row in rows])
        time = np.array([row["corr_vs_default_pct"] for row in rows])
        ax.scatter(fanout, time, s=12, alpha=0.18, color=COLORS["corr_aware"])
        edges = np.quantile(fanout, np.linspace(0, 1, 6))
        centers = []
        medians = []
        for index in range(5):
            mask = (fanout >= edges[index]) & (
                (fanout <= edges[index + 1])
                if index == 4
                else (fanout < edges[index + 1])
            )
            if np.count_nonzero(mask) < 8:
                continue
            centers.append(float(np.median(fanout[mask])))
            medians.append(float(np.median(time[mask])))
        ax.plot(centers, medians, "o-", color="#17212b", linewidth=2)
        ax.axhline(0, color="#666666", linewidth=1)
        ax.axvline(0, color="#666666", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Change in destination ranks per token")
        ax.set_ylabel("DeepEP time change (%)")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        0.8,
        "Corr-aware minus default EPLB\n"
        "Left: fewer destinations\n"
        "Down: faster DeepEP\n\n"
        "Dots: matched forward/layer cases\n"
        "Line: median in each fan-out quintile\n\n"
        "Placement also changes destination balance.",
        va="top",
        fontsize=11,
    )
    fig.suptitle("Matched changes in fan-out and DeepEP time", fontsize=15)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def summarize(paired: list[dict], prepared: Path, output: Path) -> None:
    manifest = json.loads((prepared / "manifest.json").read_text())
    population = manifest["strata_population"]
    summary = {}
    for stratum in STRATA:
        rows = [row for row in paired if row["stratum"] == stratum]
        summary[stratum] = {"cases": len(rows)}
        summary[stratum]["routing"] = {
            variant: {
                "median_fanout": float(
                    np.median([row[f"{prefix}_fanout"] for row in rows])
                ),
                "median_destination_skew": float(
                    np.median([row[f"{prefix}_skew"] for row in rows])
                ),
            }
            for variant, prefix in (
                ("baseline", "baseline"),
                ("default", "default"),
                ("corr_aware", "corr"),
            )
        }
        for field in (
            "default_vs_baseline_pct",
            "corr_vs_baseline_pct",
            "corr_vs_default_pct",
        ):
            summary[stratum][field] = {
                "median_pct": float(np.median([row[field] for row in rows])),
                "fraction_faster": float(np.mean([row[field] < 0 for row in rows])),
            }
    total_steps = sum(population.values())
    weighted = {
        variant: sum(
            row[f"{variant}_ms"]
            * population[row["stratum"]]
            / (len(manifest["strata"][row["stratum"]]) * len(manifest["layers"]))
            for row in paired
        )
        / total_steps
        for variant in VARIANTS
    }
    summary["population_weighted_estimate"] = {
        "evaluated_step_population": total_steps,
        "mean_ms": weighted,
        "default_vs_baseline_pct": 100
        * (weighted["default"] / weighted["baseline"] - 1),
        "corr_vs_baseline_pct": 100
        * (weighted["corr_aware"] / weighted["baseline"] - 1),
        "corr_vs_default_pct": 100 * (weighted["corr_aware"] / weighted["default"] - 1),
    }
    rng = np.random.default_rng(17)
    draws = np.zeros((5000, len(VARIANTS)))
    for stratum, steps in manifest["strata"].items():
        by_step = defaultdict(list)
        for row in paired:
            if row["stratum"] == stratum:
                by_step[row["step"]].append(row)
        step_means = np.array(
            [
                [
                    np.mean([row[f"{variant}_ms"] for row in by_step[step]])
                    for variant in VARIANTS
                ]
                for step in steps
            ]
        )
        indices = rng.integers(0, len(steps), size=(len(draws), len(steps)))
        draws += step_means[indices].mean(axis=1) * (population[stratum] / total_steps)
    bootstrapped = {
        "default_vs_baseline_pct": 100 * (draws[:, 1] / draws[:, 0] - 1),
        "corr_vs_baseline_pct": 100 * (draws[:, 2] / draws[:, 0] - 1),
        "corr_vs_default_pct": 100 * (draws[:, 2] / draws[:, 1] - 1),
    }
    summary["population_weighted_estimate"]["step_bootstrap_95pct"] = {
        name: np.percentile(values, [2.5, 97.5]).tolist()
        for name, values in bootstrapped.items()
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    _, paired = aggregate(args.input, args.prepared)
    plot_sweeps(paired, args.input / "captured_sweeps.svg")
    plot_heatmaps(paired, args.input / "captured_heatmaps.svg")
    plot_paired(paired, args.input / "paired_effects.svg")
    plot_fanout_change(paired, args.input / "fanout_time_change.svg")
    summarize(paired, args.prepared, args.input / "summary.json")


if __name__ == "__main__":
    main()
