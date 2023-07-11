import time
from dataclasses import dataclass
from typing import List, Any, Union, Iterable

import pandas as pd
from vantage6.client import ContainerClient
from vantage6.common import info

from verticox.scalarproduct import NPartyScalarProductClient
from verticox.aggregator import Aggregator
from verticox.grpc.datanode_pb2 import Empty
from verticox.ssl import get_secure_stub

_PYTHON = 'python'
_JAVA = 'java'
PORTS_PER_CONTAINER = 2
PYTHON_PORT = 8888
JAVA_PORT = 9999

WAIT_CONTAINER_STARTUP = 10

SLEEP = 10
NODE_TIMEOUT = 360

MAX_RETRIES = NODE_TIMEOUT // SLEEP


class NodeManagerException(Exception):
    pass


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


class V6NodeManager:

    def __init__(self, v6_client: ContainerClient,
                 data: pd.DataFrame,
                 datanode_organizations: List[int],
                 central_organization: int,
                 event_happened_column: str,
                 event_times_column: str,
                 features: List[str],
                 include_value: Any,
                 rows: Union[Iterable, None] = None,
                 **aggregator_kwargs,
                 ):
        """
        Node manager for vantage6 infrastructure. Will take care of starting and shutting the
        python and java containers

        :param v6_client: The vantage6 container client
        :param data: The data available on this node
        :param datanode_organizations: The organizations that contain the covariates/features
        :param central_organization: The organization that will run the central computation (at
            the moment it also needs to have access to the outcome data
        :param event_happened_column: The name of the column that will indicate whether an event
            happened (whether the record is right-censored or not)
        :param event_times_column: The name of the column that contains the outcome time
        :param features: The list of names of the covariates that have to be considered in this
            analysis
        :param include_value: The value in the column indicating right censoring that will
            indicate whether a record should be fully included in the analysis (so whether it is NOT
            right censored)
        :param rows: The indices of the rows to be included in training. The default is to
            include all rows.
        """
        self.v6_client = v6_client
        self.datanode_organizations = datanode_organizations
        self.central_organization = central_organization
        self.event_happened_column = event_happened_column
        self.event_times_column = event_times_column
        self.features = features
        self.include_value = include_value
        self.aggregator_kwargs = aggregator_kwargs

        self.commodity_address = None
        self.stubs = None
        self._betas = None
        self._baseline_hazard = None
        self._aggregator = None
        self._scalar_product_client = None

        if rows is not None:
            data = data[rows]

        self._data = data
        self._event_times = data[event_times_column].values
        self._event_happened = data[event_happened_column].values

    @property
    def betas(self):
        if self._betas is None:
            raise NodeManagerException('Trying to access betas before model has been fit.')
        return self._betas

    @property
    def baseline_hazard(self):
        if self._baseline_hazard is None:
            raise NodeManagerException('Trying to access baseline hazard before model has been fit')
        return self._baseline_hazard

    @property
    def scalar_product_client(self) -> NPartyScalarProductClient:
        if self._scalar_product_client is None:
            raise NodeManagerException('Trying to use scalar product client before it has been '
                                       'initialized')
        return self._scalar_product_client

    def fit(self):
        self._aggregator = Aggregator(self.stubs, self._event_times,
                                      self._event_happened,
                                      **self.aggregator_kwargs)
        self._aggregator.fit()

        self._betas = self._aggregator.get_betas()

        self._baseline_hazard = self._aggregator.compute_baseline_hazard_function()

    def _start_containers(self, input, org_ids) -> ContainerAddresses:
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

        task = self.v6_client.create_new_task(input, organization_ids=org_ids)
        addresses = self._get_algorithm_addresses(expected_num_addresses, task['id'])
        return addresses

    def _get_node_ip(self, organization_id: int):
        params = {'method': 'no_op'}

        info(f'Getting ip for organization {organization_id}')
        task = self.v6_client.create_new_task(params, [organization_id])
        addresses = self._get_algorithm_addresses(1, task['id'])

        address = addresses.python[0]
        info(f'Address: {address}')
        return address.split(':')[0]

        pass

    def start_nodes(self):
        info('Starting java containers')
        self._start_java_algorithms()

        info('Starting python containers')
        self._start_python_algorithms()
        self._create_stubs()

    def _create_stubs(self):
        stubs = []
        # Create gRPC stubs
        for a in self.python_addresses:
            # TODO: This part is stupid, it should be separate host and port in the first place.
            host, port = tuple(a.split(':'))
            stubs.append(get_secure_stub(host, port))

        info(f'Created {len(stubs)} RPC stubs')
        self.stubs = stubs

    def kill_all_algorithms(self):
        try:
            self._kill_all_python_nodes()
        except Exception as e:
            info(f'Couldn\'t kill all python nodes: {e}')
        try:
            self._kill_all_java_nodes()
        except Exception as e:
            info(f'Couldn\'t kill all java nodes: {e}')

    def _kill_all_python_nodes(self):
        for stub in self.stubs:
            stub.kill(Empty())

    def _kill_all_java_nodes(self):
        self.scalar_product_client.kill_nodes()

    def _get_algorithm_addresses(self, expected_amount: int, task_id) -> ContainerAddresses:
        addresses = self.v6_client.get_algorithm_addresses(task_id=task_id)

        retries = 0
        # Wait for nodes to get ready
        while True:
            addresses = self.v6_client.get_algorithm_addresses(task_id=task_id)

            if len(addresses) >= expected_amount:
                break

            if retries >= MAX_RETRIES:
                raise Exception(f'Could not connect to all {expected_amount} datanodes. There are '
                                f'only {len(addresses)} nodes available')
            time.sleep(SLEEP)
            retries += 1

        return ContainerAddresses.parse_addresses(addresses)

    def _start_python_algorithms(self):
        addresses = []
        info(f'Datanode ids: {self.datanode_organizations}')
        info(f'Commodity address: {self.commodity_address}')

        for id in self.datanode_organizations:
            # First run a no-op task to retrieve the address
            ip = self._get_node_ip(id)

            info(f'Address: {ip}')

            datanode_input = {
                'method': 'run_datanode',
                'kwargs': {
                    'feature_columns': self.features,
                    'event_time_column': self.event_times_column,
                    'include_column': self.event_happened_column,
                    'include_value': self.include_value,
                    'address': ip,
                    'external_commodity_address': self.commodity_address
                }
            }
            # create a new task for all organizations in the collaboration.
            info('Dispatching python datanode task')
            task = self.v6_client.create_new_task(datanode_input, organization_ids=[id])
            address = self._get_algorithm_addresses(1, task['id'])

            addresses += address.python
        self.python_addresses = addresses

    def _start_java_algorithms(self):
        # Kick off java nodes
        java_node_input = {'method': 'run_java_server'}

        commodity_address = self._start_containers(java_node_input, [self.central_organization])
        info(f'Commodity address: {commodity_address}')
        commodity_address = commodity_address.java[0]

        info(f'Running java nodes on organizations {self.datanode_organizations}')
        datanode_addresses = self._start_containers(java_node_input, self.datanode_organizations)

        info(f'Addresses: {datanode_addresses}')

        # Wait for a bit for the containers to start up
        time.sleep(WAIT_CONTAINER_STARTUP)

        # Do initial setup for nodes
        self._scalar_product_client = \
            NPartyScalarProductClient(commodity_address=commodity_address,
                                      other_addresses=datanode_addresses.java)

        self._scalar_product_client.initialize_servers()

        self.commodity_address = commodity_address
