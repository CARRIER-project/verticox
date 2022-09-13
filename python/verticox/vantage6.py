import time
from typing import List

import grpc
import pandas as pd
from vantage6.client import ContainerClient
from vantage6.tools.util import info

from verticox import datanode
from verticox.aggregator import Aggregator
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

PORT = 8888
MAX_RETRIES = 20
GRPC_OPTIONS = [('wait_for_ready', True)]
SLEEP = 5
DATANODE_TIMEOUT = None
DATA_LIMIT = 10
DEFAULT_PRECISION = 1e-3
DEFAULT_RHO = 0.5


def _limit_data(data):
    return data.iloc[:5]


def verticox(client: ContainerClient, data: pd.DataFrame, feature_columns: List[str],
             event_times_column: str, event_happened_column: str, datanode_ids: List[int] = None,
             precision: float = DEFAULT_PRECISION, rho=DEFAULT_RHO,
             *_args, **_kwargs):
    """
    TODO: Describe precision parameter
    Args:
        client:
        data:
        feature_columns:
        event_times_column:
        event_happened_column:
        datanode_ids:
        precision: determines precision in multiple places in the optimization process
        rho:
        *_args:
        **_kwargs:

    Returns:

    """
    start_time = time.time()
    event_times = data[event_times_column].values
    event_happened = data[event_happened_column]

    if datanode_ids is None:
        organizations = client.get_organizations_in_my_collaboration()
        datanode_ids = [organization.get("id") for organization in organizations]

    datanode_input = {
        "method": "run_datanode",
        "kwargs": {
            'feature_columns': feature_columns,
            'event_time_column': event_times_column,
            'event_happened_column': event_happened_column
        }
    }

    # create a new task for all organizations in the collaboration.
    info("Dispatching node-tasks")
    task = client.create_new_task(
        input_=datanode_input,
        organization_ids=datanode_ids
    )

    addresses = _get_algorithm_addresses(client, datanode_ids, task['id'])

    stubs = []
    # Create gRPC stubs
    for a in addresses:
        stubs.append(_get_stub(a['ip'], a['port']))

    aggregator = Aggregator(stubs, event_times, event_happened, convergence_precision=precision,
                            rho=rho)
    aggregator.fit()
    end_time = time.time()
    duration = end_time - start_time
    info(f'Verticox algorithm complete after {duration} seconds')
    info('Retrieving betas')
    betas = aggregator.get_betas()

    info('Killing datanodes')
    aggregator.kill_all_datanodes()
    return betas


def _get_stub(ip: str, port: int):
    """
    Get gRPC client for the datanode.

    Args:
        ip:
        port:

    Returns:

    """
    info(f'Connecting to datanode at {ip}:{port}')
    channel = grpc.insecure_channel(f'{ip}:{port}', options=GRPC_OPTIONS)
    return DataNodeStub(channel)


def _get_algorithm_addresses(client: ContainerClient, datanode_ids, task_id):
    addresses = client.get_algorithm_addresses(task_id=task_id)

    retries = 0
    # Wait for nodes to get ready
    while len(addresses) < len(datanode_ids):
        addresses = client.get_algorithm_addresses(task_id=task_id)

        if retries >= MAX_RETRIES:
            raise Exception(f'Could not connect to all {len(datanode_ids)} datanodes. There are '
                            f'only {len(addresses)} nodes available')
        time.sleep(SLEEP)
        retries += 1
    return addresses


def RPC_run_datanode(data: pd.DataFrame, feature_columns: List[str], event_time_column: str,
                     event_happened_column: str, *_args, **_kwargs):
    """
    Starts the datanode as gRPC server
    Args:
        data: the entire dataset
        feature_columns: the names of the columns that will be treated as features (covariants) in
        the analysis
        event_time_column: the name of the column that indicates event time
        event_happened_column: the name of the column that indicates whether an event has taken
                                place or whether the sample is right censored. If the value is
                                False, the sample is right censored.
        *args:
        **kwargs:

    Returns: None

    """
    info(f'Feature columns: {feature_columns}')
    info(f'Event time column: {event_time_column}')
    info(f'Event happened column: {event_happened_column}')

    # The current datanode might not have all the features
    feature_columns = [f for f in feature_columns if f in data.columns]

    features = data[feature_columns].values
    event_times = data[event_time_column].values
    event_happened = data[event_happened_column].values

    datanode.serve(features, feature_columns, event_times, event_happened, PORT,
                   timeout=DATANODE_TIMEOUT)

    return None


# Note this function also exists in other algorithm packages but since it is so easy to implement I
# decided to do that rather than rely on other algorithm packages.
def RPC_column_names(data: pd.DataFrame, *args, **kwargs):
    """


    Args:
        client:
        data:

    Returns:

    """
    return data.columns.tolist()
