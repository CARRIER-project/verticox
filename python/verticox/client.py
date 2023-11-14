import logging
import time
from collections import namedtuple
from typing import List

from vantage6.client import Client

_VERTICOX_IMAGE = "harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox"
_DATA_FORMAT = "json"
_SLEEP = 5
_TIMEOUT = 5 * 60  # 5 minutes
_DEFAULT_PRECISION = 1e-5


class VerticoxClient:
    def __init__(
            self,
            v6client: Client,
            collaboration=None,
            log_level=logging.INFO,
            image=_VERTICOX_IMAGE,
    ):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._v6client = v6client
        self._image = image
        collaborations = self._v6client.collaboration.list()
        if len(collaborations) > 1:
            raise VerticoxClientException(
                f"You are in multiple collaborations, please specify "
                f"one of:\n {collaborations}"
            )

        self.collaboration_id = collaborations[0]["id"]

    def get_active_node_organizations(self):
        nodes = self._v6client.node.list(is_online=True)

        # TODO: Add pagination support
        nodes = nodes["data"]
        return [n["organization"]["id"] for n in nodes]

    def get_column_names(self, **kwargs):
        active_nodes = self.get_active_node_organizations()
        self._logger.debug(f"There are currently {len(active_nodes)} active nodes")

        task = self._run_task(
            "column_names", organizations=active_nodes, master=False, **kwargs
        )
        return task

    def fit(
            self,
            feature_columns,
            outcome_time_column,
            right_censor_column,
            feature_nodes,
            outcome_node,
            precision=_DEFAULT_PRECISION,
            database="default",
    ):
        """
        Run cox proportional hazard analysis on the entire dataset.

        Args:
            feature_columns: a list of column names that you want to use as features
            outcome_time_column: the column name of the outcome time
            right_censor_column: the column name of the binary value that indicates if an event
            happened.
            feature_nodes: A list of node ids from the datasources that contain the feature columns
            outcome_node: The node id of the datasource that contains the outcome
            precision: precision of the verticox algorithm. The smaller the number, the more
            precise the result. Smaller precision will take longer to compute though. The default is
            1e-5
            database: If the nodes have multiple datasources, indicate the label of the datasource
            you would like to use. Otherwise the default will be used.

        Returns:

        """
        input_params = {
            "feature_columns": feature_columns,
            "event_times_column": outcome_time_column,
            "event_happened_column": right_censor_column,
            "datanode_ids": feature_nodes,
            "central_node_id": outcome_node,
            "precision": precision,
        }

        return self._run_task(
            "fit", True, [outcome_node], kwargs=input_params, database=database
        )

    def cross_validate(self,
                       feature_columns,
                       outcome_time_column,
                       right_censor_column,
                       feature_nodes,
                       outcome_node,
                       precision=_DEFAULT_PRECISION,
                       database="default"):
        input_params = {
            "feature_columns": feature_columns,
            "event_times_column": outcome_time_column,
            "event_happened_column": right_censor_column,
            "datanode_ids": feature_nodes,
            "central_node_id": outcome_node,
            "precision": precision,
        }

        return self._run_task(
            "cross_validate", True, [outcome_node], kwargs=input_params, database=database
        )

    def _run_task(
            self, method, master, organizations: List[int], kwargs=None, database="default"
    ):
        if kwargs is None:
            kwargs = {}
        # TODO: Construct description out of parameters
        description = ""
        name = ""
        task_input = {"method": method, "master": master, "kwargs": kwargs}

        print(
            f"""
                    task = self.v6client.task.create(collaboration={self.collaboration_id},
                                             organizations={organizations},
                                             name={name},
                                             image={self._image},
                                             description={description},
                                             input={task_input},
                                             data_format={_DATA_FORMAT},
                                             database={database}
                                             )
            """
        )
        task = self._v6client.task.create(
            collaboration=self.collaboration_id,
            organizations=organizations,
            name=name,
            image=self._image,
            description=description,
            input=task_input,
            data_format=_DATA_FORMAT,
            database=database,
        )

        return Task(self._v6client, task)


Result = namedtuple("Result", ["organization", "content", "log"])


class Task:
    def __init__(self, client: Client, task_data):
        self._raw_data = task_data
        self.client = client

        self.result_ids = [r["id"] for r in task_data["results"]]

    def get_results(self, timeout=_TIMEOUT):
        results = []
        max_retries = timeout // _SLEEP
        retries = 0
        result_ids = set(self.result_ids)
        results_complete = set()

        while retries < max_retries:
            results_missing = result_ids - results_complete
            for missing in results_missing:
                result = self.client.result.get(missing)
                if result["finished_at"] is not None:
                    print("Received a result")
                    organization_id = result["organization"]["id"]
                    result_content = result["result"]
                    result_log = result["log"]
                    results.append(Result(organization_id, result_content, result_log))
                    results_complete.add(missing)

            if len(results) >= len(self.result_ids):
                return results
            retries += 1
            time.sleep(_SLEEP)

        raise VerticoxClientException(f"Timeout after {timeout} seconds")


class VerticoxClientException(Exception):
    pass
