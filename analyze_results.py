"""
Plot benchmark results and generate a results-based analysis.

Input:  performance_results.json
Output: benchmark_plots/     (PNG charts)
        analysis.txt         (discussion with measured results)

Run:
    python analyze_results.py
"""

import json
import os
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt


INPUT_FILE = "performance_results.json"
PLOT_DIR = "benchmark_plots"

METRICS = {
    "creation_time_seconds": ("Creation time", "Seconds"),
    "random_search_avg_seconds": ("Average random search time", "Seconds per lookup"),
    "range_search_avg_seconds": ("Average range search time", "Seconds per range"),
}

DISPLAY_NAMES = {
    "btree": "SQLite B-tree",
    "linear_regression": "Linear regression",
    "neural_network": "Neural network",
}

COLORS = {
    "btree": "#2878B5",
    "linear_regression": "#F28E2B",
    "neural_network": "#59A14F",
}


def load_results(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    experiments = data.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in the JSON file.")

    expected_count = 4 * 3 * 3
    if len(experiments) != expected_count:
        print(
            f"Warning: found {len(experiments)} experiments; "
            f"the assignment expects {expected_count}."
        )

    return data, experiments


def format_seconds(value):
    """Format short timings so they're easier to read."""
    if value == 0:
        return "0 seconds"
    if value < 0.001:
        return f"{value * 1_000_000:.2f} µs"
    if value < 1:
        return f"{value * 1_000:.2f} ms"
    return f"{value:.3f} s"


def grouped_average(experiments, metric, data_type=None, index_type=None,
                    data_size=None):
    values = [
        row[metric]
        for row in experiments
        if (data_type is None or row["data_type"] == data_type)
        and (index_type is None or row["index_type"] == index_type)
        and (data_size is None or row["data_size"] == data_size)
    ]
    return mean(values) if values else None


def winners_by_operation(experiments):
    """Find the lowest average time per operation across all experiments."""
    winners = {}
    for metric, (label, _) in METRICS.items():
        averages = {}
        for index_type in DISPLAY_NAMES:
            value = grouped_average(experiments, metric, index_type=index_type)
            if value is not None:
                averages[index_type] = value
        winners[metric] = min(averages, key=averages.get)
    return winners


def make_plots(experiments):
    os.makedirs(PLOT_DIR, exist_ok=True)

    for metric, (title, ylabel) in METRICS.items():
        fig, axes = plt.subplots(
            1, 3, figsize=(16, 5), sharey=True, constrained_layout=True
        )

        for axis, data_type in zip(axes, ("linear", "skewed", "temporal")):
            for index_type in DISPLAY_NAMES:
                rows = sorted(
                    [
                        row for row in experiments
                        if row["data_type"] == data_type
                        and row["index_type"] == index_type
                    ],
                    key=lambda row: row["data_size"],
                )
                if not rows:
                    continue

                x_values = [row["data_size"] for row in rows]
                y_values = [row[metric] for row in rows]

                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2,
                    label=DISPLAY_NAMES[index_type],
                    color=COLORS[index_type],
                )

            axis.set_title(data_type.capitalize())
            axis.set_xlabel("Number of records")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", linestyle="--", alpha=0.4)

        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"{title} by data type and index")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.subplots_adjust(bottom=0.22)

        output_path = os.path.join(PLOT_DIR, f"{metric}.png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot: {output_path}")


def make_analysis(experiments):
    winners = winners_by_operation(experiments)
    metric_names = {
        "creation_time_seconds": "creation",
        "random_search_avg_seconds": "random search",
        "range_search_avg_seconds": "range search",
    }

    lines = []
    lines.append("Benchmark Analysis\n")

    # Paragraph 1: operation winners and representative measurements.
    winner_sentences = []
    for metric, operation in metric_names.items():
        index_type = winners[metric]
        avg = grouped_average(experiments, metric, index_type=index_type)
        winner_sentences.append(
            f"{DISPLAY_NAMES[index_type]} had the lowest overall mean "
            f"{operation} time ({format_seconds(avg)})"
        )

    lines.append(
        "Across all tested dataset sizes and distributions, "
        + "; ".join(winner_sentences)
        + ". These are averages across the experiments, so results can vary "
          "by data type and size. "
    )

    # Add best method for each operation at the largest size.
    largest_size = max(row["data_size"] for row in experiments)
    largest_details = []
    for metric, operation in metric_names.items():
        averages = {
            index_type: grouped_average(
                experiments, metric, index_type=index_type,
                data_size=largest_size
            )
            for index_type in DISPLAY_NAMES
        }
        available = {
            key: value for key, value in averages.items() if value is not None
        }
        if available:
            best = min(available, key=available.get)
            largest_details.append(
                f"at {largest_size:,} records, {DISPLAY_NAMES[best]} was "
                f"fastest for {operation} ({format_seconds(available[best])})"
            )

    lines.append(
        "At the largest tested size, "
        + "; ".join(largest_details)
        + ". Creation time includes the work performed by each implementation: "
          "SQLite row insertion plus index creation for the B-tree, and model "
          "training plus calibration for learned indexes. These timings are "
          "therefore implementation-level comparisons, not isolated algorithm "
          "costs.\n"
    )

    # Paragraph 2: distribution effects and scaling.
    distribution_sentences = []
    for data_type in ("linear", "skewed", "temporal"):
        best_by_metric = []
        for metric, operation in metric_names.items():
            averages = {
                index_type: grouped_average(
                    experiments, metric, data_type=data_type,
                    index_type=index_type
                )
                for index_type in DISPLAY_NAMES
            }
            available = {
                key: value for key, value in averages.items()
                if value is not None
            }
            if available:
                best = min(available, key=available.get)
                best_by_metric.append(
                    f"{operation}: {DISPLAY_NAMES[best]}"
                )
        distribution_sentences.append(
            f"For {data_type} data, the lowest average times were "
            + ", ".join(best_by_metric)
        )

    scaling_sentences = []
    min_size = min(row["data_size"] for row in experiments)
    for metric, operation in metric_names.items():
        for index_type in DISPLAY_NAMES:
            small = grouped_average(
                experiments, metric, index_type=index_type, data_size=min_size
            )
            large = grouped_average(
                experiments, metric, index_type=index_type,
                data_size=largest_size
            )
            if small is not None and large is not None:
                scaling_sentences.append(
                    f"{DISPLAY_NAMES[index_type]} {operation} changed from "
                    f"{format_seconds(small)} at {min_size:,} records to "
                    f"{format_seconds(large)} at {largest_size:,}"
                )

    lines.append(
        "Distribution-specific results were: "
        + "; ".join(distribution_sentences)
        + ". The charts should be used to inspect whether a model’s performance "
          "changes on skewed or temporal keys; learned-index prediction quality "
          "depends on how well the key-to-position relationship is modeled. "
          "Across sizes, "
        + "; ".join(scaling_sentences)
        + ".\n"
    )

    # Paragraph 3: trade-offs, recommendations, surprises, caveats.
    lines.append(
        "The main trade-off is that a B-tree provides general-purpose ordered "
        "lookup and range-query support without needing the key distribution "
        "to follow a learned pattern. A learned index can narrow a search when "
        "key positions are predictable, but it adds model-training and "
        "calibration costs, consumes model resources, and may need rebuilding "
        "when data changes. In this benchmark, learned-index lookups also use "
        "a binary-search fallback for correctness. I would recommend a B-tree "
        "for frequently updated data, unpredictable distributions, and "
        "dependable range queries. I would consider a learned index for mostly "
        "static, large datasets with a stable, predictable key distribution, "
        "after measuring its build and maintenance costs. "
    )

    lines.append(
        "A result worth discussing as surprising is any case where the "
        "apparently more sophisticated learned model was slower than the "
        "B-tree; model prediction overhead and training cost can outweigh "
        "search savings. Use the actual plots to identify and state your own "
        "specific surprise rather than assuming one. The B-tree baseline here "
        "is SQLite’s B-tree index, not a separately implemented B+ tree, and "
        "benchmark times depend on hardware, software versions, and system "
        "load. These limitations should be acknowledged when interpreting "
        "the results."
    )

    return "\n".join(lines)


def main():
    data, experiments = load_results(INPUT_FILE)
    make_plots(experiments)

    analysis = make_analysis(experiments)
    with open("analysis.txt", "w", encoding="utf-8") as file:
        file.write(analysis)

    print("\nSaved analysis.txt")
    print("\n--- Analysis preview ---\n")
    print(analysis)


if __name__ == "__main__":
    main()