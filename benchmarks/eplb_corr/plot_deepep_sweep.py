# SPDX-License-Identifier: Apache-2.0
"""Plot latency from deepep_sweep.py's per-repetition CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEPS = (
    ("source_tokens", "Tokens originating per rank"),
    ("destination_weight", "Busiest destination / mean destination"),
    ("fanout", "Distinct destination ranks per token"),
    ("zero_source_ranks", "Ranks with zero source tokens"),
)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No measurements in {path}")
    for row in rows:
        for key in (
            "value",
            "source_tokens_per_rank",
            "hot_weight",
            "actual_fanout",
            "dest_max_over_mean",
            "layout_ms",
            "dispatch_ms",
            "combine_ms",
            "total_ms",
        ):
            row[key] = float(row[key])
    return rows


def plot_sweeps(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {
        "layout_ms": "#6a5acd",
        "dispatch_ms": "#d97706",
        "combine_ms": "#168a70",
    }
    for ax, (sweep, label) in zip(axes.flat, SWEEPS):
        groups = defaultdict(list)
        for row in rows:
            if row["sweep"] == sweep:
                groups[row["value"]].append(row)
        if not groups:
            ax.set_visible(False)
            continue
        points = []
        for value, records in groups.items():
            if sweep == "destination_weight":
                x = float(np.median([r["dest_max_over_mean"] for r in records]))
            elif sweep == "fanout":
                x = float(np.median([r["actual_fanout"] for r in records]))
            else:
                x = value
            points.append((x, records))
        points.sort(key=lambda point: point[0])
        x = [point[0] for point in points]
        for field, color in colors.items():
            ax.plot(
                x,
                [np.median([r[field] for r in records]) for _, records in points],
                marker="o",
                color=color,
                label=field.removesuffix("_ms").title(),
            )
        ax.plot(
            x,
            [np.median([r["total_ms"] for r in records]) for _, records in points],
            marker="o",
            color="#17212b",
            linewidth=2.2,
            label="Total p50",
        )
        ax.plot(
            x,
            [
                np.percentile([r["total_ms"] for r in records], 95)
                for _, records in points
            ],
            color="#17212b",
            linestyle="--",
            label="Total p95",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Slowest-rank wall time (ms)")
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5)
    fig.suptitle("DeepEP HT layout → dispatch → combine", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def plot_heatmap(rows: list[dict], output: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        if row["sweep"] == "volume_x_weight":
            groups[(row["source_tokens_per_rank"], row["hot_weight"])].append(
                row["total_ms"]
            )
    if not groups:
        return
    volumes = sorted({key[0] for key in groups})
    weights = sorted({key[1] for key in groups})
    values = np.array(
        [
            [np.median(groups[(volume, weight)]) for volume in volumes]
            for weight in weights
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(volumes)), [int(v) for v in volumes])
    ax.set_yticks(range(len(weights)), [int(w) for w in weights])
    ax.set_xlabel("Tokens originating per rank")
    ax.set_ylabel("Hot destination sampling weight")
    ax.set_title("DeepEP HT total latency, p50")
    fig.colorbar(image, ax=ax, label="Slowest-rank wall time (ms)")
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv)
    plot_sweeps(rows, args.output_dir / "deepep_sweeps.svg")
    plot_heatmap(rows, args.output_dir / "deepep_volume_skew_heatmap.svg")


if __name__ == "__main__":
    main()
