import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from python_on_whales import docker

from verticox.common import get_test_dataset, unpack_events

_BENCHMARK_DIR = Path(__file__).absolute().parent
_DATA_DIR = _BENCHMARK_DIR / "data"

_RUNTIME_PATTERN = re.compile(r"Runtime: ([\d\.]+)")
NUM_RECORDS = [20, 40, 60, 100, 200, 500]
NUM_FEATURES = [2, 3, 4, 5, 6]
NUM_PARTIES = 2


def benchmark(num_records, num_features):
    """
    TODO: Make it possible to specify number of nodes.
    Benchmark verticox+ with specific parameters.
    Args:
        num_records: Total number of records in dataset
        num_features: Total number of features

    Returns:

    """
    print(f'Benchmarking with {num_records} records and {num_features} features')
    # Prepare dataset
    features, outcome, column_names = get_test_dataset(num_records, feature_limit=num_features)

    print(f"Column names: {column_names}")

    split = len(column_names) // NUM_PARTIES

    features = pd.DataFrame(features, columns=column_names)

    feature_sets = [features[column_names[:split]],
                    features[column_names[split:]]]

    prepare_dataset(feature_sets, outcome)

    # Check data dir
    print(f"Data dir content: {list(_DATA_DIR.iterdir())}")

    # Run test

    docker.compose.up(force_recreate=True, abort_on_container_exit=True)
    log = docker.compose.logs(services=["aggregator"], tail=10)

    print(f"Tail of aggregator log: \n{log}")
    runtime = re.search(_RUNTIME_PATTERN, log)
    seconds = runtime.groups()[0]
    seconds = float(seconds)

    print(f"Run took {seconds} seconds")
    return seconds


def prepare_dataset(feature_sets: List[pd.DataFrame], outcome: np.array):
    # Make sure to clear old data
    if _DATA_DIR.exists():
        shutil.rmtree(_DATA_DIR.absolute())
    _DATA_DIR.absolute().mkdir()

    for idx, feature_set in enumerate(feature_sets):
        filename = f"features_{idx}.parquet"
        feature_set.to_parquet(_DATA_DIR / filename)

    event_time, event_happened = unpack_events(outcome)

    outcome_df = pd.DataFrame({"event_happened": event_happened, "event_time": event_time})
    outcome_df.to_parquet(_DATA_DIR / "outcome.parquet")


def main():
    columns = ["num_records", "num_features", "runtime"]
    report_filename = f"report-{datetime.now().isoformat()}.csv"

    report_path = _BENCHMARK_DIR / report_filename

    with report_path.open('w', buffering=1) as f:
        writer = csv.writer(f)

        # Write header first
        writer.writerow(columns)

        for records in NUM_RECORDS:
            for features in NUM_FEATURES:
                runtime = benchmark(records, features)
                writer.writerow((records, features, runtime))


if __name__ == "__main__":
    main()
