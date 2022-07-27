import time
from typing import List

import grpc
import pandas as pd
from vantage6.client import ContainerClient
from vantage6.tools.util import info

from verticox import datanode
from verticox.aggregator import Aggregator
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

PORT = 7777
MAX_RETRIES = 10
GRPC_OPTIONS = [('wait_for_ready', True)]


def master(client: ContainerClient, data, event_times_column, event_happened_column,
           datanode_ids=None, *_args, **_kwargs):
    event_times = data[event_times_column].values
    event_happened = data[event_happened_column]

    if datanode_ids is None:
        organizations = client.get_organizations_in_my_collaboration()
        datanode_ids = [organization.get("id") for organization in organizations]

    datanode_input = {
        "method": "run_datanode",
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
        stubs.append(_get_stub(a))

    info('Starting aggregator')
    aggregator = Aggregator(stubs, event_times, event_happened)
    aggregator.fit()

    info('Verticox algorithm complete')
    info('Retrieving betas')
    betas = aggregator.get_betas()

    info('Killing datanodes')
    aggregator.kill_all_datanodes()
    return betas


def _get_stub(a):
    port = a['port']
    info(f'Connecting to datanode at port {port}')
    channel = grpc.insecure_channel(f'localhost:{port}', options=GRPC_OPTIONS)
    # ready = grpc.channel_ready_future(channel)
    # ready.result(timeout=300)
    stub = DataNodeStub(channel)


def _get_algorithm_addresses(client: ContainerClient, datanode_ids, task):
    addresses = client.get_algorithm_addresses(task_id=task['id'])

    retries = 0
    # Wait for nodes to get ready
    while (len(addresses) < len(datanode_ids)) and retries < MAX_RETRIES:
        time.sleep(1)
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
    # The current datanode might not have all the features
    feature_columns = [f for f in feature_columns if f in data.columns]

    features = data[feature_columns].values
    event_times = data[event_time_column].values
    event_happened = data[event_happened_column].values

    datanode.serve(features, event_times, event_happened, PORT)

    return None
