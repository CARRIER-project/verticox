import os
import shutil
import subprocess
import sys
import time
import traceback as tb
from dataclasses import dataclass
from pathlib import Path
from typing import List
import traceback
import pandas as pd
from vantage6.client import ContainerClient
from vantage6.tools.util import info

from verticox import datanode
from verticox.aggregator import Aggregator
from verticox.scalarproduct import NPartyScalarProductClient
from verticox.ssl import get_secure_stub

DATABASE_URI = 'DATABASE_URI'

PYTHON_PORT = 8888
JAVA_PORT = 9999
PORTS_PER_CONTAINER = 2
MAX_RETRIES = 20

SLEEP = 5
DATANODE_TIMEOUT = None
DATA_LIMIT = 10
DEFAULT_PRECISION = 1e-3
DEFAULT_RHO = 0.5
COMMODITY_PROPERTIES = [f'--server.port={JAVA_PORT}']
WAIT_CONTAINER_STARTUP = 10
_PYTHON = 'python'
_JAVA = 'java'
_SOME_ID = 1
_WORKAROUND_DATABASE_URI = 'default.parquet'

# Methods
NO_OP = 'no_op'


@dataclass
class ContainerAddresses:
    """
    Class to keep track of the various types of algorithm addresses
    """
    # Maps organization ids to uris
    python: List[str]
    java: List[str]

    @staticmethod
    def parse_addresses(v6_container_addresses):
        info(f'Parsing addresses: {v6_container_addresses}')
        python_addresses = []
        java_addresses = []

        for addr in v6_container_addresses:
            label = addr['label']
            uri = f'{addr["ip"]}:{addr["port"]}'

            if label == _PYTHON:
                python_addresses.append(uri)
            elif label == _JAVA:
                java_addresses.append(uri)

        return ContainerAddresses(python_addresses, java_addresses)


def _get_node_ip(client: ContainerClient, organization_id: int):
    params = {'method': 'no_op'}

    task = client.create_new_task(params, [organization_id])
    addresses = _get_algorithm_addresses(client, 1, task['id'])

    address = addresses.python[0]
    return address.split(':')[0]


def _limit_data(data):
    return data.iloc[:5]


def verticox(client: ContainerClient, data: pd.DataFrame, feature_columns: List[str],
             event_times_column: str, event_happened_column: str, include_value=True,
             datanode_ids: List[int] = None, central_node_id: int = None,
             precision: float = DEFAULT_PRECISION, rho=DEFAULT_RHO,
             *_args, **_kwargs):
    """
    TODO: Describe precision parameter
    Args:
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
    :param datanode_ids:
    :param central_node_id:

    """
    try:
        info(f'Start running verticox on features: {feature_columns}')

        info(f'My database: {client.database}')

        event_times = data[event_times_column].values
        event_happened = data[event_happened_column]

        info('Starting java containers')
        _run_java_nodes(client, datanode_ids, central_node_id)

        info('Starting python containers')
        addresses = _start_python_containers(client, datanode_ids, event_happened_column,
                                             event_times_column,
                                             feature_columns, include_value)

        info(f'Python datanode addresses: {addresses}')

        stubs = []
        # Create gRPC stubs
        for a in addresses:
            # TODO: This part is stupid, it should be separate host and port in the first place.
            host, port = tuple(a.split(':'))
            stubs.append(get_secure_stub(host, port))

        info(f'Created {len(stubs)} RPC stubs')

        aggregator, betas = compute_betas(event_happened, event_times, precision, rho, stubs)

        baseline_hazard = aggregator.compute_baseline_hazard_function()

        info('Killing datanodes')
        aggregator.kill_all_datanodes()
        return betas, baseline_hazard
    except Exception as e:
        info(f'Algorithm ended with exception {e}')
        exc_type, exc_value, exc_traceback = sys.exc_info()
        info(str(traceback.format_exception(exc_value)))


def compute_betas(event_happened, event_times, precision, rho, stubs):
    start_time = time.time()
    aggregator = Aggregator(stubs, event_times, event_happened, convergence_precision=precision,
                            rho=rho)
    aggregator.fit()
    end_time = time.time()
    duration = end_time - start_time
    info(f'Verticox algorithm complete after {duration} seconds')
    info('Retrieving betas')
    betas = aggregator.get_betas()
    return aggregator, betas


def _start_python_containers(client, datanode_ids, event_happened_column, event_times_column,
                             feature_columns, include_value):
    addresses = []

    for id in datanode_ids:
        # First run a no-op task to retrieve the address
        ip = _get_node_ip(client, id)

        info(f'Address: {ip}')

        datanode_input = {
            'method': 'run_datanode',
            'kwargs': {
                'feature_columns': feature_columns,
                'event_time_column': event_times_column,
                'include_column': event_happened_column,
                'include_value': include_value,
                'address': ip
            }
        }
        # create a new task for all organizations in the collaboration.
        info('Dispatching python datanode task')
        task = client.create_new_task(datanode_input, organization_ids=[id])
        address = _get_algorithm_addresses(client, 1, task['id'])

        addresses += address.python
    return addresses


def _run_java_nodes(client: ContainerClient,
                    datanode_ids: List[int],
                    central_node_id: int) -> str:
    # Kick off java nodes
    java_node_input = {'method': 'run_java_server'}

    commodity_address = _start_containers(client, java_node_input, [central_node_id])
    info(f'Commodity address: {commodity_address}')
    commodity_address = commodity_address.java[0]

    info(f'Running java nodes on organizations {datanode_ids}')
    datanode_addresses = _start_containers(client, java_node_input, datanode_ids)

    info(f'Addresses: {datanode_addresses}')

    # Wait for a bit for the containers to start up
    time.sleep(WAIT_CONTAINER_STARTUP)

    # Do initial setup for nodes
    scalar_product_client = \
        NPartyScalarProductClient(commodity_address=commodity_address,
                                  other_addresses=datanode_addresses.java)

    scalar_product_client.initialize_servers()

    return commodity_address


def _get_internal_java_address():
    commodity_uri = f'localhost:{JAVA_PORT}'
    return commodity_uri


def _get_java_command():
    return ['java', '-jar', _get_jar_path()] + COMMODITY_PROPERTIES


# TODO: Remove this ugly workaround!
def _move_parquet_file():
    current_location = os.environ[DATABASE_URI]
    current_location = Path(current_location)

    target = current_location.parent / _WORKAROUND_DATABASE_URI
    shutil.copy(current_location, target)

    return str(target.absolute())


def _get_current_java_address(client: ContainerClient, some_id):
    """

    Args:
        client:
        some_id:

    Returns:

    """
    input_ = {'method': 'no_op'}
    task = client.create_new_task(input_, organization_ids=[some_id])

    info(f'No-op task {task}')

    my_task_id = task['id'] - 1

    info(f'Get task: {client.get_task(my_task_id)}')
    info(f'Get previous task: {client.get_task(my_task_id - 1)}')

    address = client.get_algorithm_addresses(task_id=my_task_id)
    info(f' Current address {address}')
    parsed = ContainerAddresses.parse_addresses(address)

    return parsed.java[0]


def RPC_no_op(*args, **kwargs):
    pass


def _start_containers(client, input, org_ids) -> ContainerAddresses:
    """
    Trigger a task at the nodes at org_ids and retrieve the addresses for those algorithm containers
    Args:
        client:
        input:
        org_ids:

    Returns:

    """

    # Every container will have two addresses because it has both a java and a python endpoint
    expected_num_addresses = len(org_ids) * PORTS_PER_CONTAINER

    task = client.create_new_task(input, organization_ids=org_ids)
    addresses = _get_algorithm_addresses(client, expected_num_addresses, task['id'])
    return addresses


def _get_algorithm_addresses(client: ContainerClient,
                             expected_amount: int, task_id) \
        -> ContainerAddresses:
    addresses = client.get_algorithm_addresses(task_id=task_id)

    retries = 0
    # Wait for nodes to get ready
    while True:
        addresses = client.get_algorithm_addresses(task_id=task_id)

        if len(addresses) >= expected_amount:
            break

        if retries >= MAX_RETRIES:
            raise Exception(f'Could not connect to all {expected_amount} datanodes. There are '
                            f'only {len(addresses)} nodes available')
        time.sleep(SLEEP)
        retries += 1

    return ContainerAddresses.parse_addresses(addresses)


def _filter_algorithm_addresses(addresses, label):
    for a in addresses:
        if a['label'] == label:
            yield a


def RPC_run_datanode(data: pd.DataFrame,
                     feature_columns: List[str] = (),
                     event_time_column: str = None,
                     include_column: str = None, include_value: bool = None,
                     external_commodity_address=None,
                     address=None,
                     *_args,
                     **_kwargs):
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
        *args:
        **kwargs:

    Returns: None

    """
    info(f'Feature columns: {feature_columns}')
    info(f'All columns: {data.columns}')
    info(f'Event time column: {event_time_column}')
    info(f'Censor column: {include_column}')
    try:
        # The current datanode might not have all the features
        feature_columns = [f for f in feature_columns if f in data.columns]

        info(f'Feature columns after filtering: {feature_columns}')
        features = data[feature_columns].values

        datanode.serve(features=features, feature_names=feature_columns, port=PYTHON_PORT,
                       include_column=include_column,
                       include_value=include_value, timeout=DATANODE_TIMEOUT,
                       commodity_address=external_commodity_address,
                       address=address)
    except Exception as e:
        ex_type, ex_value, ex_tb = sys.exc_info()
        info('Some exception happened')
        info(str(tb.format_tb(ex_tb)))
        raise e
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
    info(f'Running command: {command}')
    target_uri = _move_parquet_file()
    subprocess.run(command, env=_get_workaround_sysenv(target_uri))


def RPC_test_sum_local_features(data: pd.DataFrame, features: List[str], mask, *args, **kwargs):
    # Only check requested features
    data = data[features]

    # Exclude censored data
    data = data[mask]

    return data.sum(axis=0).values


def _get_workaround_sysenv(target_uri):
    env = os.environ
    env[DATABASE_URI] = target_uri
    return env


def _get_jar_path():
    return os.environ.get('JAR_PATH')
