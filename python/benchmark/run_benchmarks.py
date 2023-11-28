import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List
from jinja2 import Environment, FileSystemLoader
import numpy as np
import pandas as pd
from python_on_whales import docker

from verticox.common import get_test_dataset, unpack_events

_BENCHMARK_DIR = Path(__file__).absolute().parent
_TEMPLATES_DIR = _BENCHMARK_DIR / "templates"
_DATA_DIR = _BENCHMARK_DIR / "data"

_RUNTIME_PATTERN = re.compile(r"Runtime: ([\d\.]+)")
NUM_RECORDS = [20, 40, 60, 100, 200, 500]
NUM_FEATURES = [2, 3, 4, 5, 6]
NUM_DATANODES = [1, 2, 3, 4, 5]


class NotEnoughFeaturesException(Exception):
    pass


def benchmark(num_records, num_features, num_datanodes):
    """
    TODO: Make it possible to specify number of nodes.
    Benchmark verticox+ with specific parameters.
    Args:
        num_datanodes:
        num_records: Total number of records in dataset
        num_features: Total number of features

    Returns:

    """
    print(f'Benchmarking with {num_records} records, '
          f'{num_features} features, '
          f'{num_datanodes} datanodes')

    if num_features < num_datanodes:
        raise NotEnoughFeaturesException(f"Less features than datanodes"
                                         f"\nNumber of features: {num_features}, "
                                         f"number of datanodes: {num_datanodes}")

    # Prepare dataset
    features, outcome, column_names = get_test_dataset(num_records, feature_limit=num_features)

    print(f"Column names: {column_names}")

    split = len(column_names) // num_datanodes

    features = pd.DataFrame(features, columns=column_names)

    feature_sets = [features[column_names[:split]],
                    features[column_names[split:]]]

    prepare_dataset(feature_sets, outcome)

    # Check data dir
    print(f"Data dir content: {list(_DATA_DIR.iterdir())}")

    prepare_java_properties(num_datanodes)

    prepare_compose_file(num_datanodes)

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


def get_template_env() -> Environment:
    return Environment(loader=FileSystemLoader(_TEMPLATES_DIR))


def get_properties_template():
    return get_template_env().get_template("application-datanode.properties.jinja")


def get_compose_template():
    return get_template_env().get_template("docker-compose.yml.jinja")


def prepare_java_properties(num_datanodes: int):
    all_java_servers = {"http://commodity", "http://javanode-outcome"}
    all_java_servers = all_java_servers | {f"http://javanode{n}" for n in range(num_datanodes)}
    # Regular datanodes
    for i in range(num_datanodes):
        server_name = f"http://javanode{i}"
        other_servers = all_java_servers - {server_name}

        create_properties_file(f"application-datanode{i}.properties",
                               server_name,
                               f"features_{i}.parquet",
                               other_servers)

    # Outcome node
    server_name = "http://javanode-outcome"
    create_properties_file("application-outcomenode.properties", server_name, "outcome.parquet",
                           all_java_servers - {server_name})

    # Commodity node
    server_name = "http://commodity"
    create_properties_file("application-commodity.properties", server_name, None,
                           all_java_servers - {server_name})


def create_properties_file(filename, server_name, data_filename, other_servers):
    properties = get_properties_template().render(servers=other_servers,
                                                  datafile=data_filename,
                                                  name=server_name)
    with open(_BENCHMARK_DIR / filename, "w") as f:
        f.write(properties)


def prepare_compose_file(num_datanodes: int):
    compose = get_compose_template().render(num_datanodes=num_datanodes)

    with open(_BENCHMARK_DIR / "docker-compose.yml", "w") as f:
        f.write(compose)


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
                for datanodes in NUM_DATANODES:
                    try:
                        runtime = benchmark(records, features, datanodes)
                        writer.writerow((records, features, runtime))
                    except NotEnoughFeaturesException:
                        print("Skipping")


if __name__ == "__main__":
    main()
