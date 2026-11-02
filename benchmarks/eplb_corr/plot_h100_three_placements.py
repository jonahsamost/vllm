# SPDX-License-Identifier: Apache-2.0
"""Compare three H100 captured-route placements using saved sweep results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(results: Path) -> None:
    summary = json.loads((results / "summary.json").read_text())
    stats = summary["variants"]
    names = ("baseline", "default", "w03")
    labels = ("No EPLB", "Default EPLB", "Corr aware λ=0.3")
    colors = ("#3973a8", "#d97706", "#168a70")
    baseline_ms = stats["baseline"]["mean_ms"]
    time_change = [
        100 * (stats[name]["mean_ms"] / baseline_ms - 1) for name in names
    ]
    fanout = [stats[name]["median_fanout"] for name in names]
    skew = [stats[name]["median_destination_skew"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = (
        (
            axes[0],
            time_change,
            "DeepEP time vs no EPLB",
            "% change; lower is faster",
            lambda value: f"{value:+.2f}%",
        ),
        (
            axes[1],
            fanout,
            "Destination ranks per token",
            "Median; lower is fewer destinations",
            lambda value: f"{value:.2f}",
        ),
        (
            axes[2],
            skew,
            "Busiest / mean destination",
            "Median; closer to 1 is more balanced",
            lambda value: f"{value:.3f}",
        ),
    )
    for ax, values, title, ylabel, formatter in panels:
        bars = ax.bar(range(3), values, color=colors, width=0.63)
        ax.set_xticks(range(3), labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.bar_label(bars, labels=[formatter(value) for value in values], padding=3)
    axes[0].axhline(0, color="#17212b", linewidth=0.8)
    axes[0].set_ylim(min(time_change) - 0.6, 0.6)
    axes[1].set_ylim(0, max(fanout) * 1.17)
    axes[2].set_ylim(0, max(skew) * 1.17)
    fig.suptitle(
        "H100 captured routes: 210 forwards at 8k–16k total tokens, eight layers"
    )
    fig.tight_layout()
    fig.savefig(results / "three_placements.svg")
    fig.savefig(results / "three_placements.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results)


if __name__ == "__main__":
    main()
