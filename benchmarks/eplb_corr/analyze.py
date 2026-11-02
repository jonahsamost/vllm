# SPDX-License-Identifier: Apache-2.0
"""Summarize paired replay results and draw throughput/interactivity curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

COLORS = {
    "baseline": "#515e70",
    "default": "#1388a8",
    "corr_aware": "#d65e35",
}
LABELS = {
    "baseline": "No EPLB",
    "default": "Default EPLB",
    "corr_aware": "Correlation-aware EPLB",
}


def read_rows(root: Path) -> list[dict]:
    rows = []
    for path in root.glob("*/c*/summary.json"):
        variant = path.parent.parent.name
        if variant not in COLORS:
            continue
        summary = json.loads(path.read_text())
        measured = summary["measured"]
        rows.append(
            {
                "variant": variant,
                "concurrency": summary["concurrency"],
                "input_sha256": summary["input_sha256"],
                "requests": measured["requests"],
                "failed": measured["failed"],
                "output_tokens_per_s": measured["output_tokens_per_s"],
                "requests_per_s": measured["requests_per_s"],
                "ttft_mean_ms": measured["ttft_ms"]["mean"],
                "ttft_p95_ms": measured["ttft_ms"]["p95"],
                "tpot_mean_ms": measured["tpot_ms"]["mean"],
                "tpot_p95_ms": measured["tpot_ms"]["p95"],
                "e2e_p50_ms": measured["e2e_ms"]["p50"],
            }
        )
    rows.sort(key=lambda row: (row["variant"], row["concurrency"]))
    if len({row["input_sha256"] for row in rows}) > 1:
        raise ValueError("Benchmark runs used different frozen inputs")
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(
    rows: list[dict], metric: str, title: str, y_label: str, output: Path
) -> None:
    complete = [
        row
        for row in rows
        if row["failed"] == 0
        and row["output_tokens_per_s"] is not None
        and row[metric] is not None
    ]
    if not complete:
        return
    width, height = 900, 610
    left, right, top, bottom = 95, 835, 75, 515
    x_max = max(row["output_tokens_per_s"] for row in complete) * 1.12
    y_max = max(row[metric] for row in complete) * 1.15

    def point(row: dict) -> tuple[float, float]:
        x = left + row["output_tokens_per_s"] / x_max * (right - left)
        y = bottom - row[metric] / y_max * (bottom - top)
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#263340">',
        f'<text x="{left}" y="35" font-size="23" font-weight="bold">'
        f'{escape(title)}</text>',
    ]
    for index in range(6):
        y = bottom - index / 5 * (bottom - top)
        value = y_max * index / 5
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                'stroke="#e7edf2"/>',
                f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
                f'font-size="13">{value:.0f}</text>',
            ]
        )
        x = left + index / 5 * (right - left)
        x_value = x_max * index / 5
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                'stroke="#edf1f4"/>',
                f'<text x="{x:.1f}" y="{bottom + 23}" text-anchor="middle" '
                f'font-size="13">{x_value:.0f}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
            'stroke="#344455" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
            'stroke="#344455" stroke-width="1.5"/>',
            f'<text x="{(left + right) / 2:.1f}" y="{height - 24}" '
            'text-anchor="middle" font-size="16">Output tokens / second</text>',
            f'<text transform="translate(24 {(top + bottom) / 2:.1f}) rotate(-90)" '
            f'text-anchor="middle" font-size="16">{escape(y_label)}</text>',
        ]
    )
    for legend_index, variant in enumerate(COLORS):
        variant_rows = sorted(
            (row for row in complete if row["variant"] == variant),
            key=lambda row: row["concurrency"],
        )
        if not variant_rows:
            continue
        color = COLORS[variant]
        coordinates = [point(row) for row in variant_rows]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="3"/>'
        )
        for row, (x, y) in zip(variant_rows, coordinates):
            parts.extend(
                [
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>',
                    f'<text x="{x + 8:.1f}" y="{y - 7:.1f}" '
                    f'font-size="11" fill="{color}">c{row["concurrency"]}</text>',
                ]
            )
        legend_x = left + legend_index * 240
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 65}" '
                f'x2="{legend_x + 25}" y2="{height - 65}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 34}" y="{height - 60}" '
                f'font-size="13">{escape(LABELS[variant])}</text>',
            ]
        )
    parts.extend(["</g>", "</svg>"])
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.results_root)
    if not rows:
        raise ValueError("No replay summaries found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "throughput_interactivity.csv")
    write_svg(
        rows,
        "tpot_mean_ms",
        "Throughput vs mean time per output token",
        "Mean time per output token (ms)",
        args.output_dir / "throughput_tpot.svg",
    )
    write_svg(
        rows,
        "ttft_p95_ms",
        "Throughput vs p95 time to first token",
        "p95 time to first token (ms)",
        args.output_dir / "throughput_ttft.svg",
    )
    print(json.dumps({"points": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
