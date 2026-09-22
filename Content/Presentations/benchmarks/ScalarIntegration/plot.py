#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot scalar-integration speedup from benchmark CSV files."
    )
    parser.add_argument("csv", nargs="+", type=Path, help="benchmark CSV file(s)")
    parser.add_argument("--output", required=True, type=Path, help="output PDF")
    return parser.parse_args()


def read_series(path):
    series = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            label = row["label"]
            series.setdefault(label, []).append(
                (int(row["intervals"]), float(row["speedup"]))
            )
    return series


def main():
    arguments = parse_arguments()
    combined = {}
    for path in arguments.csv:
        for label, points in read_series(path).items():
            if label in combined:
                raise ValueError(f"duplicate series label: {label}")
            combined[label] = sorted(points)

    figure, axes = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for label, points in combined.items():
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
        )

    axes.axhline(1.0, color="black", linewidth=1, label="Serial")
    axes.set_xscale("log", base=10)
    axes.set_yscale("log")
    axes.set_xlabel("Number of intervals")
    axes.set_ylabel("Speedup over serial")
    axes.set_title("Kokkos speedup over serial: scalar integration")
    axes.grid(which="both", linestyle=":", linewidth=0.6)
    axes.legend()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output)


if __name__ == "__main__":
    main()
