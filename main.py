"""
B-tree and learned-index benchmark for Python 3.12.

Install:
    python -m pip install numpy==1.26.4 scikit-learn==1.5.2

Run:
    python main.py

Optional:
    python main.py --student-name "Your Name" --student-id "Your ID"

Output:
    performance_results.json
"""

import argparse
import json
import sqlite3
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SIZES = [1_000, 10_000, 100_000, 1_000_000]
DATA_TYPES = ["linear", "skewed", "temporal"]
RANDOM_LOOKUPS = 1_000
RANGE_QUERIES = 10
RANGE_FRACTION = 0.10
SEED = 42


def generate_data(size, data_type, rng):
    """Return integer keys for one of the required data distributions."""
    if data_type == "linear":
        # Uniformly distributed unique keys.
        return rng.permutation(np.arange(size, dtype=np.int64))

    if data_type == "skewed":
        # Zipf-distributed values. Repeated keys are expected.
        values = rng.zipf(a=1.5, size=size)
        return (values % max(1, size * 10)).astype(np.int64)

    if data_type == "temporal":
        # Timestamp-like values with a trend, seasonal pattern, and noise.
        t = np.arange(size, dtype=np.float64)
        period = max(1, size // 12)
        seasonal = 0.12 * size * np.sin(2 * np.pi * t / period)
        noise = rng.normal(0, max(1.0, size * 0.005), size)
        timestamps = t + seasonal + noise
        timestamps -= timestamps.min()
        return np.rint(timestamps).astype(np.int64)

    raise ValueError(f"Unsupported data type: {data_type}")


def create_ranges(sorted_keys, rng):
    """Create 10 ranges spanning about 10% of the records by sorted position."""
    n = len(sorted_keys)
    width = max(1, int(n * RANGE_FRACTION))
    ranges = []

    for _ in range(RANGE_QUERIES):
        max_start = max(0, n - width)
        start_pos = int(rng.integers(0, max_start + 1))
        end_pos = min(n - 1, start_pos + width - 1)

        ranges.append(
            (int(sorted_keys[start_pos]), int(sorted_keys[end_pos]))
        )

    return ranges


def measure_random_search(search_function, keys, rng):
    """Return average seconds per successful lookup over 1,000 lookups."""
    queries = rng.choice(keys, size=RANDOM_LOOKUPS, replace=True)

    start = time.perf_counter()
    for key in queries:
        search_function(int(key))

    return (time.perf_counter() - start) / RANDOM_LOOKUPS


def benchmark_btree(keys, ranges, rng):
    """Build and query SQLite's B-tree index."""
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE records (key INTEGER NOT NULL)")

    start = time.perf_counter()
    connection.executemany(
        "INSERT INTO records (key) VALUES (?)",
        ((int(key),) for key in keys),
    )
    connection.execute("CREATE INDEX records_key_idx ON records(key)")
    connection.commit()
    creation_time = time.perf_counter() - start

    def search(key):
        connection.execute(
            "SELECT key FROM records WHERE key = ? LIMIT 1",
            (key,),
        ).fetchone()

    random_search_time = measure_random_search(search, keys, rng)

    start = time.perf_counter()
    for low, high in ranges:
        connection.execute(
            "SELECT key FROM records "
            "WHERE key BETWEEN ? AND ? ORDER BY key",
            (low, high),
        ).fetchall()

    range_search_time = (time.perf_counter() - start) / len(ranges)
    connection.close()

    return creation_time, random_search_time, range_search_time


def build_model(index_type):
    """Return the requested learned-index model."""
    if index_type == "linear_regression":
        return make_pipeline(
            StandardScaler(),
            LinearRegression(),
        )

    if index_type == "neural_network":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                solver="adam",
                max_iter=100,
                early_stopping=True,
                random_state=SEED,
            ),
        )

    raise ValueError(f"Unsupported learned index: {index_type}")


def benchmark_learned(index_type, sorted_keys, ranges, rng):
    """
    Train a model to predict a key's approximate position in sorted_keys.

    The model narrows the search to a calibrated window. A full binary-search
    fallback ensures that lookups remain correct if the window is insufficient.
    """
    unique_keys = np.unique(sorted_keys)
    ranks = np.searchsorted(sorted_keys, unique_keys, side="left").astype(
        np.float64
    )

    # Limit neural-network training data for practical runtime at large sizes.
    if index_type == "neural_network" and len(unique_keys) > 50_000:
        sample_positions = np.linspace(
            0, len(unique_keys) - 1, 50_000
        ).astype(np.int64)
        train_keys = unique_keys[sample_positions]
        train_ranks = ranks[sample_positions]
    else:
        train_keys = unique_keys
        train_ranks = ranks

    model = build_model(index_type)

    start = time.perf_counter()
    model.fit(train_keys.reshape(-1, 1), train_ranks)

    # Calibrate the search-window size using prediction errors.
    predictions = model.predict(unique_keys.reshape(-1, 1))
    max_error = int(np.ceil(np.max(np.abs(predictions - ranks)))) + 2
    creation_time = time.perf_counter() - start

    def lower_bound(key):
        predicted_rank = float(
            model.predict(np.array([[key]], dtype=np.float64))[0]
        )
        predicted_rank = int(round(predicted_rank))

        left = max(0, predicted_rank - max_error)
        right = min(len(sorted_keys), predicted_rank + max_error + 1)

        position = left + int(
            np.searchsorted(sorted_keys[left:right], key, side="left")
        )

        # Fall back to a full binary search if the window missed the key.
        if position >= len(sorted_keys) or sorted_keys[position] != key:
            position = int(np.searchsorted(sorted_keys, key, side="left"))

        return position

    def search(key):
        # Locate the first occurrence; all benchmark query keys exist.
        return lower_bound(key)

    random_search_time = measure_random_search(search, sorted_keys, rng)

    start = time.perf_counter()
    for low, high in ranges:
        left = lower_bound(low)
        right = int(np.searchsorted(sorted_keys, high, side="right"))
        # Materialize the matching records, as a range query returns records.
        _ = sorted_keys[left:right]

    range_search_time = (time.perf_counter() - start) / len(ranges)

    return creation_time, random_search_time, range_search_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-name", default="Your Full Name")
    parser.add_argument("--student-id", default="Your Student ID")
    parser.add_argument(
        "--output",
        default="performance_results.json",
    )
    args = parser.parse_args()

    results = []
    rng = np.random.default_rng(SEED)

    for size in SIZES:
        for data_type in DATA_TYPES:
            print(f"Generating {data_type} data: N={size:,}")
            keys = generate_data(size, data_type, rng)
            sorted_keys = np.sort(keys)
            ranges = create_ranges(sorted_keys, rng)

            print("  Benchmarking SQLite B-tree...")
            creation, random_time, range_time = benchmark_btree(
                keys, ranges, rng
            )
            results.append({
                "data_size": size,
                "data_type": data_type,
                "index_type": "btree",
                "creation_time_seconds": round(creation, 9),
                "random_search_avg_seconds": round(random_time, 9),
                "range_search_avg_seconds": round(range_time, 9),
            })

            for index_type in ("linear_regression", "neural_network"):
                print(f"  Benchmarking {index_type}...")
                creation, random_time, range_time = benchmark_learned(
                    index_type, sorted_keys, ranges, rng
                )
                results.append({
                    "data_size": size,
                    "data_type": data_type,
                    "index_type": index_type,
                    "creation_time_seconds": round(creation, 9),
                    "random_search_avg_seconds": round(random_time, 9),
                    "range_search_avg_seconds": round(range_time, 9),
                })

    output = {
        "student_name": args.student_name,
        "student_id": args.student_id,
        "llm_used": "ChatGPT",
        "experiments": results,
    }

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)

    print(f"\nSaved {len(results)} experiments to {args.output}")


if __name__ == "__main__":
    main()