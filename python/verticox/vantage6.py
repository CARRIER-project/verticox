import os
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
from verticox.preprocess import preprocess_data, Columns

DATABASE_URI = "DATABASE_URI"
DATA_LIMIT = 10
DEFAULT_PRECISION = 1e-6
DEFAULT_RHO = 0.5
COMMODITY_PROPERTIES = [f"--server.port={node_manager.JAVA_PORT}"]
NO_OP_TIME = 360
_SOME_ID = 1
_WORKAROUND_DATABASE_URI = "default.parquet"
DATABASE_DIR = "/mnt/data" # For preprocessing

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
        include_value: any = True,
        datanode_ids: List[int] = None,
        central_node_id: int = None,
        precision: float = DEFAULT_PRECISION,
        rho: float = DEFAULT_RHO,
        database: str|None = None,
        *_args,
        **_kwargs,
):
    """
    Fit a cox proportional hazards model using the Verticox+ algorithm

    Args:
        client: v6 client provided by the algorithm wrapper
        data: dataframe containing the data, provided by algorithm wrapper
        feature_columns: The columns to be used as features
        event_times_column: The name of the column that contains the event times
        event_happened_column: The name of the column that contains whether an event has happened,
        or whether the sample is right censored.
        include_value: The value in the event_happened_column that means the record is NOT right-censored
        datanode_ids: List of organization ids of the nodes that will be used as feature nodes
        central_node_id:  Organization id of the node that will be used as the central node. This
        node should contain the outcome data.
        precision: Precision for the Cox model. The algorithm will stop when the difference
        between iterations falls below this number
        rho: Penalty parameter
        database: Name of the database to be used (default is "default")
        *_args:
        **_kwargs:

    Returns: A dictionary containing the coefficients of the model ("coefs") and the baseline
    hazard function of the model ("baseline_hazard_x" and "baseline_hazard_y").
    """

    # Preprocessing data
    # TODO: This can removed once we move to v6 version 5.x
    columns = Columns(feature_columns, event_times_column, event_happened_column)
    data, columns, data_location = preprocess_data(data, output_dir=DATABASE_DIR,columns=columns )

    info(f"Columns: {columns}")

    manager = node_manager.V6NodeManager(
        client,
        data,
        datanode_ids,
        central_node_id,
        columns.event_happened_column,
        columns.event_times_column,
        columns.feature_columns,
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

@data(1)
@algorithm_client
def cross_validate(client: AlgorithmClient,
                   data: pd.DataFrame,
                   feature_columns: List[str],
                   event_times_column: str,
                   event_happened_column: str,
                   include_value=True,
                   datanode_ids: List[int] = None,
                   central_node_id: int = None,
                   convergence_precision: float = DEFAULT_PRECISION,
                   rho: float = DEFAULT_RHO,
                   n_splits: int = DEFAULT_KFOLD_SPLITS,
                   *_args,
                   **_kwargs):
    """
    Fit a cox proportional hazards model using the Verticox+ algorithm using crossvalidation.
    Works similarly to the `fit` method, but trains multiple times on smaller subsets of the data
    using k-fold crossvalidation.

    Args:
        client: v6 client provided by the algorithm wrapper
        data: dataframe containing the data, provided by algorithm wrapper
        feature_columns: The columns to be used as features
        event_times_column: The name of the column that contains the event times
        event_happened_column: The name of the column that contains whether an event has happened,
        or whether the sample is right censored.
        include_value: The value in the event_happened_column that means the record is NOT right-censored
        datanode_ids: List of organization ids of the nodes that will be used as feature nodes
        central_node_id:  Organization id of the node that will be used as the central node. This
        node should contain the outcome data.
        between iterations falls below this number
        convergence_precision: Precision for the Cox model. The algorithm will stop when the difference
        rho: Penalty parameter
        n_splits: Number of splits for crossvalidation
        *_args:
        **_kwargs:

    Returns:  A tuple containing 3 lists: `c_indices`, `coefs`, `baseline_hazards`

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
        convergence_precision=convergence_precision,
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
        baseline_hazards = [_stepfunction_to_tuple(f) for f in baseline_hazards]

        print(f'Returning c_indices: {c_indices}\ncoefs: {coefs}\nbaseline_hazards: {baseline_hazards}')
        return c_indices, coefs, baseline_hazards
    except Exception as e:
        info(f"Algorithm ended with exception {e}")
        info(traceback.format_exc())
    finally:
        manager.kill_all_algorithms()


def _stepfunction_to_tuple(f: StepFunction) -> Tuple[
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
def _get_data_dir(database:str= "default"):
    env_name = f"{database.upper()}_{DATABASE_URI}"
    info(f"Env name {env_name}")
    current_location = get_env_var(env_name)
    current_location = Path(current_location)

    data_dir = current_location.parent

    return str(data_dir.absolute())


@data(1)
def no_op(*args, **kwargs):
    """
    A function that does nothing for a while. It is used as a partial algorithm within the verticox+
    algorithm and and should not be called by itself.

    Args:
        *args:
        **kwargs:

    Returns:

    """
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
        selected_columns: List[str] = (),
        event_time_column: str|None = None,
        include_column: str|None = None,
        include_value: bool|None = None,
        external_commodity_address: str|None = None,
        address=None,
        **kwargs,
):
    """
    Starts the datanode (feature node) as gRPC server. This function is a partial function called by
    the main verticox algorithm. It is not meant to be called by itself.

    Args:
        data: the entire dataset, provided by the algorithm wrapper
        include_value: This value in the data means the record is NOT right-censored
        selected_columns: the names of the columns that will be treated as features (covariants) in
        the analysis
        event_time_column: the name of the column that indicates event time
        include_column: the name of the column that indicates whether an event has taken
                                place or whether the sample is right censored. If the value is
                                False, the sample is right censored.
        external_commodity_address: Address of the n-party product protocol commodity server
        address: The address where this server will be running.

    Returns: None


    """
    info(f"Selected columns: {selected_columns}")
    info(f"Columns present in dataset: {data.columns}")
    info(f"Event time column: {event_time_column}")
    info(f"Censor column: {include_column}")


    columns = Columns(selected_columns, None, None)

    features, new_columns = preprocess_data(data, columns)

    # The current datanode might not have all the features
    selected_columns = [f for f in new_columns.feature_columns if f in data.columns]
    info(f"Feature columns after filtering: {selected_columns}")
    features = data[selected_columns]

    datanode.serve(
        data=features.values,
        feature_names=selected_columns,
        port=node_manager.PYTHON_PORT,
        include_column=include_column,
        include_value=include_value,
        commodity_address=external_commodity_address,
        address=address,
    )


# Note this function also exists in other algorithm packages but since it is so easy to implement I
# decided to do that rather than rely on other algorithm packages.
@data(1)
def column_names(data: pd.DataFrame, *args, **kwargs):
    """
    Returns the names of the columns in the data. Useful to investigate the dataset before
    running the actual algorithm.


    Args:
        client: v6 client provided by the algorithm wrapper
        data: dataframe containing the data, provided by algorithm wrapper

    Returns: a list of column names

    """
    return data.columns.tolist()


@data(1)
def run_java_server(_data: pd.DataFrame, *_args, features=None,
                    event_times_column=None,
                    event_happened_column=None, **kwargs):
    """
    Partial function that starts the java server. This function is called by the main verticox+
    algorithm (`fit` or `cross_validate`) and should not be called by itself.
    Args:
        _data: data provided by the vantage6 algorithm wrapper
        *_args:
        features: list of column names that will be used as features
        event_times_column: Name of the column that contains the event times
        event_happened_column: Name of the column that contains whether an event has happened,
        or whether the sample is right-censored
        **kwargs:

    """
    info("Starting java server")
    command = _get_java_command()
    info(f"Running command: {command}")
    #target_uri = _move_parquet_file(database)

    columns = Columns(features, event_times_column, event_happened_column)
    data, column_names, data_path = preprocess_data(_data, columns, _get_data_dir())

    subprocess.run(command, env=_get_workaround_sysenv(data_path))


@data(1)
def test_sum_local_features(
        data: pd.DataFrame, features: List[str], mask, *args, **kwargs
):
    """
    Obsolete

    Args:
        data:
        features:
        mask:
        *args:
        **kwargs:

    Returns:

    """
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
