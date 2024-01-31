import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd
from sksurv.functions import StepFunction
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.util import info, get_env_var

from verticox import datanode, node_manager
from verticox.cross_validation import kfold_cross_validate
from verticox.defaults import DEFAULT_KFOLD_SPLITS

DATABASE_URI = "DATABASE_URI"
DATANODE_TIMEOUT = None
DATA_LIMIT = 10
DEFAULT_PRECISION = 1e-6
DEFAULT_RHO = 0.5
COMMODITY_PROPERTIES = [f"--server.port={node_manager.JAVA_PORT}"]
NO_OP_TIME = 360
_SOME_ID = 1
_WORKAROUND_DATABASE_URI = "default.parquet"

# Methods
NO_OP = "no_op"


@data(1)
@algorithm_client
def fit(
        client: AlgorithmClient,
        data: pd.DataFrame,
        feature_columns: List[str],
        event_times_column: str,
        event_happened_column: str,
        include_value=True,
        datanode_ids: List[int] = None,
        central_node_id: int = None,
        precision: float = DEFAULT_PRECISION,
        rho=DEFAULT_RHO,
        database=None,
        *_args,
        **_kwargs,
):
    """

    Args:
        client:
        data:
        feature_columns:
        event_times_column:
        event_happened_column:
        include_value:
        datanode_ids:
        central_node_id:
        precision:
        rho:
        *_args:
        **_kwargs:

    Returns:

    """
    manager = node_manager.V6NodeManager(
        client,
        data,
        datanode_ids,
        central_node_id,
        event_happened_column,
        event_times_column,
        feature_columns,
        include_value,
        convergence_precision=precision,
        rho=rho,
        database=database,
    )
    try:
        info(f"Start running verticox on features: {feature_columns}")

        manager.start_nodes()

        start_time = time.time()
        manager.fit()
        end_time = time.time()
        duration = end_time - start_time
        info(f"Verticox algorithm complete after {duration} seconds")

        info("Killing datanodes")
        return {"coefs": manager.coefs,
                "baseline_hazard_x": list(manager.baseline_hazard.x),
                "baseline_hazard_y": list(manager.baseline_hazard.y)
                }
    except Exception as e:
        info(f"Algorithm ended with exception {e}")
        info(traceback.format_exc())
    finally:
        manager.kill_all_algorithms()


def cross_validate(client: AlgorithmClient,
                   data: pd.DataFrame,
                   feature_columns: List[str],
                   event_times_column: str,
                   event_happened_column: str,
                   include_value=True,
                   datanode_ids: List[int] = None,
                   central_node_id: int = None,
                   precision: float = DEFAULT_PRECISION,
                   rho=DEFAULT_RHO,
                   n_splits=DEFAULT_KFOLD_SPLITS,
                   *_args,
                   **_kwargs):
    manager = node_manager.V6NodeManager(
        client,
        data,
        datanode_ids,
        central_node_id,
        event_happened_column,
        event_times_column,
        feature_columns,
        include_value,
        convergence_precision=precision,
        rho=rho,
    )
    try:
        info(f"Start running verticox on features: {feature_columns}")

        manager.start_nodes()

        start_time = time.time()
        c_indices, coefs, baseline_hazards = kfold_cross_validate(manager, n_splits=n_splits)
        end_time = time.time()
        duration = end_time - start_time
        info(f"Verticox algorithm complete after {duration} seconds")

        info("Killing datanodes")
        # Make baseline hazard functions serializable
        baseline_hazards = [stepfunction_to_tuple(f) for f in baseline_hazards]

        return c_indices, coefs, baseline_hazards
    except Exception as e:
        info(f"Algorithm ended with exception {e}")
        info(traceback.format_exc())
    finally:
        manager.kill_all_algorithms()


def stepfunction_to_tuple(f: StepFunction) -> Tuple[
    List[Union[int, float]], List[Union[int, float]]]:
    """
    Converts stepfunction to a tuple of lists. This makes the object serializable.
    Args:
        f:

    Returns:

    """
    x = f.x.tolist()
    y = f.y.tolist()

    return x, y


# TODO: Remove this ugly workaround!
def _move_parquet_file(database:str):
    env_name = f"{database.upper()}_{DATABASE_URI}"
    info(f"Env name {env_name}")
    current_location = get_env_var(env_name)
    current_location = Path(current_location)

    info(f"Moving parquet file from {current_location} to {_WORKAROUND_DATABASE_URI}")
    target = current_location.parent / _WORKAROUND_DATABASE_URI

    if target != current_location:
        shutil.copy(current_location, target)

    return str(target.absolute())


@data(1)
def no_op(*args, **kwargs):
    info(f"Sleeping for {NO_OP_TIME}")
    time.sleep(NO_OP_TIME)
    info("Shutting down.")


def _filter_algorithm_addresses(addresses, label):
    for a in addresses:
        if a["label"] == label:
            yield a


@data(1)
def run_datanode(
        data: pd.DataFrame,
        *args,
        feature_columns: List[str] = (),
        event_time_column: str = None,
        include_column: str = None,
        include_value: bool = None,
        external_commodity_address=None,
        address=None,
        **kwargs,
):
    """
    Starts the datanode as gRPC server
    Args:
        data: the entire dataset
        external_commodity_address:
        include_value: This value in the data means the record is NOT right-censored
        feature_columns: the names of the columns that will be treated as features (covariants) in
        the analysis
        event_time_column: the name of the column that indicates event time
        include_column: the name of the column that indicates whether an event has taken
                                place or whether the sample is right censored. If the value is
                                False, the sample is right censored.
        address:

    Returns: None


    """
    info(f"Feature columns: {feature_columns}")
    info(f"All columns: {data.columns}")
    info(f"Event time column: {event_time_column}")
    info(f"Censor column: {include_column}")
    # The current datanode might not have all the features
    feature_columns = [f for f in feature_columns if f in data.columns]

    info(f"Feature columns after filtering: {feature_columns}")
    features = data[feature_columns].values

    datanode.serve(
        data=features,
        feature_names=feature_columns,
        port=node_manager.PYTHON_PORT,
        include_column=include_column,
        include_value=include_value,
        timeout=DATANODE_TIMEOUT,
        commodity_address=external_commodity_address,
        address=address,
    )


# Note this function also exists in other algorithm packages but since it is so easy to implement I
# decided to do that rather than rely on other algorithm packages.
@data(1)
def column_names(data: pd.DataFrame, *args, **kwargs):
    """


    Args:
        client:
        data:

    Returns:

    """
    return data.columns.tolist()


@data(1)
def run_java_server(_data, *_args, database=None, **kwargs):
    info("Starting java server")
    command = _get_java_command()
    info(f"Running command: {command}")
    target_uri = _move_parquet_file(database)
    subprocess.run(command, env=_get_workaround_sysenv(target_uri))


@data(1)
def test_sum_local_features(
        data: pd.DataFrame, features: List[str], mask, *args, **kwargs
):
    # Only check requested features
    data = data[features]

    # Exclude censored data
    data = data[mask]

    return data.sum(axis=0).values


def _get_java_command():
    return ["java", "-jar", _get_jar_path()] + COMMODITY_PROPERTIES


def _get_jar_path():
    return os.environ.get("JAR_PATH")


def _get_workaround_sysenv(target_uri):
    env = os.environ
    env[DATABASE_URI] = target_uri
    return env
