"""
`node_manager` is concerned with setting up, resetting and killing the various nodes that are
part of the verticox+ algorithm.
"""
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import List, Any, Union, Iterable

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from vantage6.algorithm.client import AlgorithmClient
from vantage6.common import info

from verticox.aggregator import Aggregator
from verticox.common import Split
from verticox.grpc.datanode_pb2 import Empty, Rows, Subset
from verticox.grpc.datanode_pb2_grpc import DataNodeStub
from verticox.scalarproduct import NPartyScalarProductClient
from verticox.ssl import get_secure_stub

_PYTHON = "python"
_JAVA = "java"
PORTS_PER_CONTAINER = 2
PYTHON_PORT = 8888
JAVA_PORT = 9999

WAIT_CONTAINER_STARTUP = 10

SLEEP = 10
NODE_TIMEOUT = 360
MAX_WORKERS = 10

MAX_RETRIES = NODE_TIMEOUT // SLEEP

DOCKER_COMPOSE_PYTHON_NODES = ("pythonnode1:7777", "pythonnode2:7777")
DOCKER_COMPOSE_JAVA_NODES = ("javanode1:80", "javanode2:80", "javanode-outcome:80")
DOCKER_COMPOSE_COMMODITY_NODE = "commodity:80"

Outcome = namedtuple("Outcome", "time event_happened")

Result = namedtuple("Result", "coefs baseline_hazard")


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
        info(f"Parsing addresses: {v6_container_addresses}")
        python_addresses = []
        java_addresses = []

        for addr in v6_container_addresses:
            label = addr["label"]
            uri = f'{addr["ip"]}:{addr["port"]}'

            if label == _PYTHON:
                python_addresses.append(uri)
            elif label == _JAVA:
                java_addresses.append(uri)

        return ContainerAddresses(python_addresses, java_addresses)


class BaseNodeManager(ABC):
    @abstractmethod
    def __init__(
            self,
            data: pd.DataFrame,
            event_times_column,
            event_happened_column,
            aggregator_kwargs,
            features=None,
            include_value=True,
    ):
        # Putting the results in one tuple makes it easier to reset when a new fold needs to be
        # activated. The results are the only part of the state that need to be reset.
        self._result: Result = None

        self._scalar_product_client = None
        self._stubs = None
        self._event_happened_column = event_happened_column
        self._event_times_column = event_times_column

        self._features = features
        self._include_value = include_value
        self._aggregator_kwargs = aggregator_kwargs
        self._data = data
        self._train_selection = None

        self._aggregator = None
        self._scalar_product_client = None
        self.split: Split = None
        self._python_addresses = None
        self._aggregator: Union[None, Aggregator] = None

    @property
    def data(self):
        return self._data

    @property
    def num_total_records(self):
        return self.data.shape[0]

    @property
    def num_current_selection(self):
        return len(self.split.train)

    @property
    def result(self):
        if self._result is None:
            raise NodeManagerException(
                "Trying to access results before model has been fit."
            )
        return self._result

    @property
    def coefs(self):
        return self.result.coefs

    @property
    def baseline_hazard(self):
        return self.result.baseline_hazard

    @property
    def scalar_product_client(self) -> NPartyScalarProductClient:
        if self._scalar_product_client is None:
            raise NodeManagerException(
                "Trying to use scalar product client before it has been " "initialized"
            )
        return self._scalar_product_client

    @property
    def stubs(self):
        if self._stubs is None:
            raise NodeManagerException("Stubs haven't been initialized yet.")
        return self._stubs

    @stubs.setter
    def stubs(self, stubs: List[DataNodeStub]):
        self._stubs = stubs

    def reset(self, train_selection: Union[None, Iterable[int]] = None):
        if train_selection is not None:
            info("Computing on subset of data")

        self._train_selection = train_selection
        self._reset_central_node(train_selection)
        self._reset_java_nodes(train_selection)
        self._reset_python_nodes(train_selection)

        self._aggregator = None

    def _reset_java_nodes(self, training_selection: Union[None, Iterable[int]] = None):
        if training_selection is None:
            selection = np.ones(self.num_total_records, dtype=bool)
        else:
            selection = np.zeros(self.num_total_records, dtype=bool)
            selection[training_selection] = True

        selection = selection.tolist()
        self.scalar_product_client.activate_fold(selection)

    def _reset_central_node(self, train_selection: Union[Iterable[int], None]):
        """

        Args:
            train_selection: an iterable containing the indices of the records that need to be
            included in the training set

        Returns:

        """
        if train_selection is not None:
            train_mask = np.zeros(self.num_total_records, dtype=bool)
            train_mask[train_selection] = True

            train_data = self.data.iloc[train_mask]
            test_data = self.data.iloc[~train_mask]
            self.split = Split(
                self._get_outcome(train_data),
                self._get_outcome(test_data),
                self.data
            )
        else:
            # If there is no selection train_outcome is all data and test set is None
            self.split = Split(self._get_outcome(self.data), None, self.data)

    def _get_outcome(self, data):
        return Outcome(
            data[self._event_times_column].values,
            data[self._event_happened_column].values,
        )

    def _reset_python_nodes(self, rows):
        message = Rows(rows=rows)

        for stub in self.stubs:
            stub.reset(message)

    def fit(self):
        preparation_start = datetime.now()
        self._aggregator = Aggregator(
            self.stubs,
            self.split.train.time,
            self.split.train.event_happened,
            **self._aggregator_kwargs,
        )
        preparation_end = datetime.now()

        preparation_runtime = preparation_end - preparation_start

        runtime_start = datetime.now()
        self._aggregator.fit()
        runtime_end = datetime.now()

        runtime = runtime_end - runtime_start
        print(f"Preparation runtime: {preparation_runtime.total_seconds()}")
        print(f"Fitting runtime: {runtime.total_seconds()}")

        info(f"Finished fitting model")

        betas = self._aggregator.betas_
        baseline_hazard = self._aggregator.baseline_hazard_function_

        self._result = Result(coefs=betas, baseline_hazard=baseline_hazard)

    def test(self):
        print(f'Num training samples: {len(self.split.train.time)}')
        print(f'Num test samples: {len(self.split.test.time)}')
        c_index = self.compute_c_index(subset=Subset.TEST)

        return c_index

    def compute_c_index(self, subset=Subset.TEST):
        """
        Concordance index based on the concordance_index_censored
        implementation from sksurv.

        Args:
            subset:

        Returns:

        .._Concordance_index_censored implementation:
            https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html
        """
        estimates = self._aggregator.compute_record_level_sigmas(subset)

        c_index, _, _, _, _ = concordance_index_censored(self.split.test.event_happened,
                                                         self.split.test.time, estimates)
        return c_index

    def start_nodes(self):
        info("Starting java containers")
        self.start_java_algorithms()

        info("Starting python containers")
        self.start_python_algorithms()
        self.create_stubs()

        self.reset()

    @abstractmethod
    def start_java_algorithms(self):
        pass

    @abstractmethod
    def start_python_algorithms(self):
        pass

    def create_stubs(self):
        stubs = []
        # Create gRPC stubs
        for a in self._python_addresses:
            # TODO: This part is stupid, it should be separate host and port in the first place.
            host, port = tuple(a.split(":"))
            stubs.append(get_secure_stub(host, port))

        info(f"Created {len(stubs)} RPC stubs")
        self.stubs = stubs

    @abstractmethod
    def kill_all_algorithms(self):
        pass


class LocalNodeManager(BaseNodeManager):
    def __init__(
            self,
            data: pd.DataFrame,
            event_times_column,
            event_happened_column,
            aggregator_kwargs,
            commodity_address=DOCKER_COMPOSE_COMMODITY_NODE,
            python_datanode_addresses=DOCKER_COMPOSE_PYTHON_NODES,
            other_java_addresses=DOCKER_COMPOSE_JAVA_NODES,
            **kwargs,
    ):
        super().__init__(
            data=data,
            event_times_column=event_times_column,
            event_happened_column=event_happened_column,
            aggregator_kwargs=aggregator_kwargs,
            **kwargs,
        )
        self._commodity_address = commodity_address
        self._other_java_addresses = other_java_addresses
        self._python_addresses = python_datanode_addresses

    def start_python_algorithms(self):
        """When running locally, the python nodes are not controlled by the node manager,
        so this function does nothing."""
        pass

    def kill_all_algorithms(self):
        """
        We don't need to kill the algorithms locally
        Returns:

        """
        pass

    def start_java_algorithms(self):
        self._scalar_product_client = NPartyScalarProductClient(
            commodity_address=self._commodity_address,
            other_addresses=self._other_java_addresses,
        )
        self._scalar_product_client.initialize_servers()


class V6NodeManager(BaseNodeManager):
    def __init__(
            self,
            v6_client: AlgorithmClient,
            data: pd.DataFrame,
            datanode_organizations: List[int],
            central_organization: int,
            event_happened_column: str,
            event_times_column: str,
            features: List[str],
            include_value: Any,
            rows: Union[Iterable, None] = None,
            database=None,
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
        super().__init__(
            data,
            event_times_column,
            event_happened_column,
            aggregator_kwargs,
            features=features,
            include_value=include_value,
        )

        self._v6_client: AlgorithmClient = v6_client
        self._datanode_organizations = datanode_organizations
        self._central_organization = central_organization
        self._commodity_address = None
        self._database = database

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

        task = self._v6_client.task.create(input, organizations=org_ids, name="Start containers")
        addresses = self._get_algorithm_addresses(expected_num_addresses, task["id"])
        return addresses

    def _get_node_ip(self, organization_id: int):
        params = {"method": "no_op"}

        info(f"Getting ip for organization {organization_id}")
        task = self._v6_client.task.create(params, [organization_id], "no_op", "no_op task")
        addresses = self._get_algorithm_addresses(1, task["id"])

        address = addresses.python[0]
        info(f"Address: {address}")
        return address.split(":")[0]

    def kill_all_algorithms(self):
        try:
            self._kill_all_python_nodes()
        except Exception as e:
            info(f"Couldn't kill all python nodes: {e}")
        try:
            self._kill_all_java_nodes()
        except Exception as e:
            info(f"Couldn't kill all java nodes: {e}")

    def _kill_all_python_nodes(self):
        for stub in self.stubs:
            stub.kill(Empty())

    def _kill_all_java_nodes(self):
        self.scalar_product_client.kill_nodes()

    def _get_algorithm_addresses(
            self, expected_amount: int, task_id
    ) -> ContainerAddresses:
        retries = 0

        # Wait for nodes to get ready
        while True:
            addresses = self._v6_client.vpn.get_child_addresses()

            info(f"Addresses: {addresses}")
            # Filter on current task
            addresses = [a for a in addresses if a["task_id"] == task_id]

            if len(addresses) >= expected_amount:
                break

            if retries >= MAX_RETRIES:
                raise Exception(
                    f"Could not connect to all {expected_amount} datanodes. There are "
                    f"only {len(addresses)} nodes available"
                )
            time.sleep(SLEEP)
            retries += 1

        return ContainerAddresses.parse_addresses(addresses)

    def start_python_algorithms(self):
        info(f"Datanode ids: {self._datanode_organizations}")
        info(f"Commodity address: {self._commodity_address}")

        with ThreadPoolExecutor(MAX_WORKERS) as executor:
            addresses = executor.map(
                self._start_python_algorithm_at_organization,
                self._datanode_organizations,
            )

        self._python_addresses = list(addresses)

    def _start_python_algorithm_at_organization(self, id):
        # First run a no-op task to retrieve the address
        ip = self._get_node_ip(id)
        info(f"Address: {ip}")
        datanode_input = {
            "method": "run_datanode",
            "kwargs": {
                "feature_columns": self._features,
                "event_time_column": self._event_times_column,
                "include_column": self._event_happened_column,
                "include_value": self._include_value,
                "address": ip,
                "external_commodity_address": self._commodity_address,
            },
        }
        # create a new task for all organizations in the collaboration.
        info("Dispatching python datanode task")
        task = self._v6_client.task.create(datanode_input, [id], "python_datanode",
                                           "start python datanode")
        addresses = self._get_algorithm_addresses(1, task["id"])
        return addresses.python[0]

    def start_java_algorithms(self):
        # Kick off java nodes
        java_node_input = {"method": "run_java_server", "kwargs": {"database": self._database}}

        commodity_address = self._start_containers(
            java_node_input, [self._central_organization]
        )
        info(f"Commodity address: {commodity_address}")
        commodity_address = commodity_address.java[0]

        # For just the java part we need to run a datanode as well as a commodity node at the
        # central server because the commodity node doesn't access the data
        datanode_organizations = self._datanode_organizations + [
            self._central_organization
        ]

        info(f"Running java nodes on organizations {datanode_organizations}")

        datanode_addresses = self._start_containers(
            java_node_input, datanode_organizations
        )

        info(f"Addresses: {datanode_addresses}")

        # Wait for a bit for the containers to start up
        time.sleep(WAIT_CONTAINER_STARTUP)

        # Do initial setup for nodes
        self._scalar_product_client = NPartyScalarProductClient(
            commodity_address=commodity_address, other_addresses=datanode_addresses.java
        )

        self._scalar_product_client.initialize_servers()

        self._commodity_address = commodity_address
