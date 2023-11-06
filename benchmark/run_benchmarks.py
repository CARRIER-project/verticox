import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from python_on_whales import docker
import re
from verticox.common import get_test_dataset, unpack_events

_BENCHMARK_DIR = Path(__file__).parent
_DATA_DIR = _BENCHMARK_DIR / "data"

_RUNTIME_PATTERN = re.compile(r"Runtime: ([\d\.]*)\n")


def main():
    # Prepare dataset
    features, outcome, column_names = get_test_dataset(20, feature_limit=2)
    features = pd.DataFrame(features, columns=column_names)

    feature_sets = [features[[c]] for c in column_names]

    prepare_dataset(feature_sets, outcome)

    # Check data dir
    print(f"Data dir content: {list(_DATA_DIR.iterdir())}")

    # Run test
    docker.compose.up(force_recreate=True, abort_on_container_exit=True)

    log = docker.compose.logs(services=["aggregator"], tail=10)
    print(f"Tail of aggregator log: \n{log}")
    runtime = re.match(_RUNTIME_PATTERN, log)
    seconds = runtime.groups()[0]
    seconds = float(seconds)

    print(f"Run took {seconds} seconds")


def prepare_dataset(feature_sets: List[pd.DataFrame], outcome: np.array):
    _DATA_DIR.mkdir(exist_ok=True)

    for idx, feature_set in enumerate(feature_sets):
        filename = f"features_{idx}.parquet"
        feature_set.to_parquet(_DATA_DIR / filename)

    event_time, event_happened = unpack_events(outcome)

    outcome_df = pd.DataFrame({"event_happened": event_happened, "event_time": event_time})
    outcome_df.to_parquet(_DATA_DIR / "outcome.parquet")


if __name__ == "__main__":
    main()
