#!/usr/bin/env python3

import csv
import json
import re
import shutil
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import List

import clize
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from python_on_whales import docker, DockerException

from verticox.datasets import get_test_dataset, unpack_events, NotEnoughFeaturesException

_BENCHMARK_DIR = Path(__file__).absolute().parent
_TEMPLATES_DIR = _BENCHMARK_DIR / "templates"
_DATA_DIR = _BENCHMARK_DIR / "data"

_CONVERGENCE_RUNTIME_PATTERN = re.compile(r"Fitting runtime: ([\d\.]+)")
_PREPARATION_RUNTIME_PATTERN = re.compile(r"Preparation runtime: ([\d\.]+)")
_COMPARISON_PATTERN = re.compile(r"Benchmark output: (.+)")

BenchmarkResult = namedtuple("BenchmarkResult", ["records", "features",
                                                 "iterations", "parties",
                                                 "preparation_runtime", "convergence_runtime",
                                                 "mse", "sad", "mad", "comment"])


def benchmark(num_records: int, num_features: int, num_datanodes: int, dataset: str,
              total_num_iterations: (None, int) = None, rebuild: bool = False) -> BenchmarkResult:
    """
    TODO: Make it possible to specify number of nodes.
    Benchmark verticox+ with specific parameters.
    Args:

        total_num_iterations:
        num_datanodes:
        num_records: Total number of records in dataset
        num_features: Total number of features
        dataset: Which dataset to use, options are "whas500" and "aids".
        rebuild:

    Returns:

    """
    print(f'Benchmarking with {num_records} records, '
          f'{num_features} features, '
          f'{num_datanodes} datanodes, '
          f'{total_num_iterations} iterations'
          )

    if num_features < num_datanodes:
        raise NotEnoughFeaturesException(f"Less features than datanodes"
                                         f"\nNumber of features: {num_features}, "
                                         f"number of datanodes: {num_datanodes}")

    orchestrate_nodes(num_datanodes, num_features, num_records, dataset,
                      total_num_iterations=total_num_iterations)

    # Run test
    docker.compose.up(force_recreate=True, abort_on_container_exit=True, remove_orphans=True,
                      build=rebuild)
    log = docker.compose.logs(services=["aggregator"], tail=10)

    print(f"Tail of aggregator log: \n{log}")
    preparation_seconds = get_runtime(_PREPARATION_RUNTIME_PATTERN, log)
    convergence_seconds = get_runtime(_CONVERGENCE_RUNTIME_PATTERN, log)

    results = {"preparation_runtime": preparation_seconds,
               "convergence_runtime": convergence_seconds,
               "records": num_records,
               "features": num_features,
               "parties": num_datanodes,
               "iterations": total_num_iterations
               }

    comparison = re.search(_COMPARISON_PATTERN, log)
    comparison = comparison.groups()[0]
    metrics = json.loads(comparison)

    results.update(metrics)

    print(f"Preparation took {preparation_seconds} seconds\nConvergence took {convergence_seconds}")
    return BenchmarkResult(**results)


def get_runtime(pattern, log):
    runtime = re.search(pattern, log)
    seconds = runtime.groups()[0]
    seconds = float(seconds)
    return seconds


def orchestrate_nodes(num_datanodes: int, num_features: int, num_records: int, dataset: str,
                      total_num_iterations):
    # Prepare dataset
    features, outcome, column_names = get_test_dataset(num_records, feature_limit=num_features,
                                                       dataset=dataset)
    features = pd.DataFrame(features, columns=column_names)
    feature_sets = split_features(features, num_datanodes)
    write_datasets(feature_sets, outcome)
    prepare_java_properties(num_datanodes)
    prepare_compose_file(num_datanodes, total_num_iterations)


def split_features(features: pd.DataFrame, num_datanodes: int) -> list[pd.DataFrame]:
    column_names = features.columns
    split = len(column_names) / num_datanodes
    feature_sets = []

    split_indices = np.arange(num_datanodes)
    split_indices = split_indices * split
    split_indices = np.floor(split_indices).astype(int)

    for i in range(len(split_indices) - 1):
        columns = column_names[split_indices[i]:split_indices[i + 1]]
        feature_sets.append(features[columns])

    # Add the last split as well
    columns = column_names[split_indices[-1]:]
    feature_sets.append(features[columns])

    return feature_sets


def write_datasets(feature_sets: List[pd.DataFrame], outcome: np.array):
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
    # Remove old properties:
    for properties_file in _BENCHMARK_DIR.glob("*.properties"):
        properties_file.unlink()

    all_java_servers = {"http://commodity:80", "http://javanode-outcome:80"}
    all_java_servers = all_java_servers | {f"http://javanode{n}:80" for n in range(num_datanodes)}
    # Regular datanodes
    for i in range(num_datanodes):
        server_name = f"http://javanode{i}"
        other_servers = all_java_servers - {server_name}

        create_properties_file(f"application-datanode{i}.properties",
                               server_name,
                               f"features_{i}.parquet",
                               other_servers)

    # Outcome node
    server_name = "http://javanode-outcome:80"
    create_properties_file("application-outcomenode.properties", server_name, "outcome.parquet",
                           all_java_servers - {server_name})

    # Commodity node
    server_name = "http://commodity:80"
    create_properties_file("application-commodity.properties", server_name, None,
                           all_java_servers - {server_name})


def create_properties_file(filename, server_name, data_filename, other_servers):
    properties = get_properties_template().render(servers=other_servers,
                                                  datafile=data_filename,
                                                  name=server_name)
    with open(_BENCHMARK_DIR / filename, "w") as f:
        f.write(properties)


def prepare_compose_file(num_datanodes: int, total_num_iterations: int):
    compose = get_compose_template().render(num_datanodes=num_datanodes,
                                            total_num_iterations=total_num_iterations)

    with open(_BENCHMARK_DIR / "docker-compose.yml", "w") as f:
        f.write(compose)


def main(parameter_table: str, dataset="seer"):
    """
    Benchmark verticox+ while varying number of datanodes, number of records and number of features.
    Args:
        parameter_table:
        dataset:

    Returns:

    """
    report_filename = f"report-{dataset}_{datetime.now().isoformat()}.csv"

    report_path = _BENCHMARK_DIR / report_filename
    parameter_path = _BENCHMARK_DIR / parameter_table

    with (parameter_path.open("r") as parameter_file,
          report_path.open("w", buffering=1) as result_file):
        reader = csv.DictReader(parameter_file)

        writer = csv.DictWriter(result_file, fieldnames=BenchmarkResult._fields)
        writer.writeheader()

        rebuild = True

        for parameters in reader:
            try:
                records = int(parameters["records"])
                features = int(parameters["features"])
                parties = int(parameters["parties"])
                iterations = int(parameters["iterations"])

                results = benchmark(records,
                                    features,
                                    parties,
                                    dataset,
                                    total_num_iterations=iterations,
                                    rebuild=rebuild)

                writer.writerow(results._asdict())
                rebuild = False
            except NotEnoughFeaturesException:
                writer.writerow({"comment": "Not enough features"})
                print("Skipping")
            except DockerException:
                print(f"Current run threw error, skipping")
                writer.writerow({"num_records": records, "num_features": features,
                                 "datanodes": parties, "comment": "error"})


if __name__ == "__main__":
    clize.run(main)
