import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from python_on_whales import docker
from verticox.common import get_test_dataset, unpack_events

_COMPOSE_DIR = Path(__file__).parent
_DATA_DIR = _COMPOSE_DIR / "data"


def main():
    os.chdir(_COMPOSE_DIR)

    # Prepare dataset
    features, outcome, column_names = get_test_dataset(100, feature_limit=2)
    features = pd.DataFrame(features, columns=column_names)

    feature_sets = [features[[c]] for c in column_names]

    prepare_dataset(feature_sets, outcome)

    # Check data dir

    # Run test
    docker.compose.up(force_recreate=True)


def prepare_dataset(feature_sets: List[pd.DataFrame], outcome: np.array):
    for idx, feature_set in enumerate(feature_sets):
        filename = f"features_{idx}.parquet"
        feature_set.to_parquet(_DATA_DIR / filename)

    event_happened, event_time = unpack_events(outcome)

    outcome_df = pd.DataFrame({"event_happened": event_happened, "event_time": event_time})
    outcome_df.to_parquet(_DATA_DIR / "outcome.parquet")


if __name__ == "__main__":
    main()
