import os
import subprocess
import time
from typing import List, Tuple, Any, Dict

from vantage6.common import debug

from verticox.scalarproduct import NPartyScalarProductClient
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
COMMODITY_PROPERTIES = [f'--server.port={PORT}']
_PROTOCOL = 'http://'


def _limit_data(data):
    return data.iloc[:5]


def verticox(client: ContainerClient, data: pd.DataFrame, feature_columns: List[str],
             event_times_column: str, event_happened_column: str, include_value=True,
             central_node_id: int = None,
             datanode_ids: List[int] = None,
             precision: float = DEFAULT_PRECISION, rho=DEFAULT_RHO,
             *_args, **_kwargs):
    '''
    TODO: Describe precision parameter
    Args:
        central_node_id:
        include_value:
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

    '''
    start_time = time.time()
    event_times = data[event_times_column].values
    event_happened = data[event_happened_column]

    commodity_address = _run_java_nodes(client, central_node_id, datanode_ids)

    datanode_input = {
        'method': 'run_datanode',
        'kwargs': {
            'feature_columns': feature_columns,
            'event_time_column': event_times_column,
            'include_column': event_happened_column,
            'include_value': include_value,
            'commodity_address': commodity_address
        }
    }

    # create a new task for all organizations in the collaboration.
    info('Dispatching node-tasks')
    task = client.create_new_task(
        input_=datanode_input,
        organization_ids=datanode_ids
    )

    addresses = _get_algorithm_addresses(client, len(datanode_ids), task['id'])

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


def _run_java_nodes(client: ContainerClient, current_organization, datanode_ids: List[int]) -> str:
    # Kick off java nodes
    java_node_input = {'method': 'run_java_server'}

    # First dispatch just the commodity node, otherwise we don't know which address belongs to
    # this node
    _start_local_java()
    commodity_uri = f'{_PROTOCOL}localhost:{PORT}'

    datanode_addresses = _start_containers(client, java_node_input, datanode_ids)
    datanode_uris = [_construct_uri(a) for a in datanode_addresses]

    # Do initial setup for nodes
    scalar_product_client = NPartyScalarProductClient(commodity_address=commodity_uri,
                                                      other_addresses=datanode_uris)

    scalar_product_client.initialize_servers()

    return commodity_uri


def _start_local_java():
    command = _get_java_command()
    process = subprocess.Popen(command)

    return process


def _get_java_command():
    return ['java', '-jar', _get_jar_path()] + COMMODITY_PROPERTIES


def _construct_uri(result_address: Dict[str, Any]):
    return f'{_PROTOCOL}{result_address["ip"]}:{result_address["port"]}'


def _start_containers(client, input, org_ids) -> List[Dict[str, Any]]:
    """
    Trigger a task at the nodes at org_ids and retrieve the addresses for those algorithm containers
    Args:
        client:
        input:
        org_ids:

    Returns:

    """
    task = client.create_new_task(input, organization_ids=org_ids)
    addresses = _get_algorithm_addresses(client, len(org_ids), task['id'])
    return addresses


def _get_stub(ip: str, port: int):
    '''
    Get gRPC client for the datanode.

    Args:
        ip:
        port:

    Returns:

    '''
    info(f'Connecting to datanode at {ip}:{port}')
    channel = grpc.insecure_channel(f'{ip}:{port}', options=GRPC_OPTIONS)
    return DataNodeStub(channel)


def _get_algorithm_addresses(client: ContainerClient,
                             expected_amount: int, task_id) -> List[Dict[str, Any]]:
    addresses = client.get_algorithm_addresses(task_id=task_id)

    retries = 0
    # Wait for nodes to get ready
    while len(addresses) < expected_amount:
        addresses = client.get_algorithm_addresses(task_id=task_id)

        if retries >= MAX_RETRIES:
            raise Exception(f'Could not connect to all {expected_amount} datanodes. There are '
                            f'only {len(addresses)} nodes available')
        time.sleep(SLEEP)
        retries += 1
    return addresses


def RPC_run_datanode(data: pd.DataFrame, feature_columns: List[str], event_time_column: str,
                     include_column: str, include_value: bool, commodity_address=None, *_args,
                     **_kwargs):
    """
    Starts the datanode as gRPC server
    Args:
        include_value: This value in the data means the record is NOT right-censored
        data: the entire dataset
        feature_columns: the names of the columns that will be treated as features (covariants) in
        the analysis
        event_time_column: the name of the column that indicates event time
        include_column: the name of the column that indicates whether an event has taken
                                place or whether the sample is right censored. If the value is
                                False, the sample is right censored.
        *args:
        **kwargs:

    Returns: None

    """
    info(f'Feature columns: {feature_columns}')
    info(f'Event time column: {event_time_column}')
    info(f'Censor column: {include_column}')

    # The current datanode might not have all the features
    feature_columns = [f for f in feature_columns if f in data.columns]

    features = data[feature_columns].values
    datanode.serve(features, feature_columns, port=PORT, include_column=include_column,
                   include_value=include_value, timeout=DATANODE_TIMEOUT,
                   commodity_address=commodity_address)

    return None


# Note this function also exists in other algorithm packages but since it is so easy to implement I
# decided to do that rather than rely on other algorithm packages.
def RPC_column_names(data: pd.DataFrame, *args, **kwargs):
    '''


    Args:
        client:
        data:

    Returns:

    '''
    return data.columns.tolist()


def RPC_run_java_server(_data, *_args, **_kwargs):
    info('Starting java server')

    command = _get_java_command()
    debug(f'Running command: {command}')
    subprocess.run(command)


def _get_jar_path():
    return os.environ.get('JAR_PATH')
