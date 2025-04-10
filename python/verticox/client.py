import json
import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List

from verticox.config import DOCKER_IMAGE
from matplotlib import pyplot as plt
from vantage6.client import Client

_DATA_FORMAT = "json"
_SLEEP = 5
_TIMEOUT = 5 * 60  # 5 minutes
_DEFAULT_PRECISION = 1e-5

PartialResult = namedtuple("Result", ["organization", "content", "log"])
HazardFunction = namedtuple("HazardFunction", ["x", "y"])


class VerticoxClient:
    """
    Client for running verticox. This client is a wrapper around the vantage6 client to simplify
    use.
    """
    def __init__(
            self,
            v6client: Client,
            collaboration=None,
            log_level=logging.INFO,
            image=DOCKER_IMAGE,
    ):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._v6client = v6client
        self._image = image
        collaborations = self._v6client.collaboration.list()["data"]
        if len(collaborations) > 1:
            raise VerticoxClientException(
                f"You are in multiple collaborations, please specify "
                f"one of:\n {collaborations}"
            )

        self.collaboration_id = collaborations[0]["id"]

    def get_active_node_organizations(self) -> List[int]:
        """
        Get the organization ids of the active nodes in the collaboration.

        Returns: a list of organization ids

        """
        nodes = self._v6client.node.list(is_online=True)

        # TODO: Add pagination support
        nodes = nodes["data"]
        return [n["organization"]["id"] for n in nodes]

    def get_column_names(self, **kwargs):
        """
        Get the column names of the dataset at all active nodes.

        Args:
            **kwargs:

        Returns:

        """
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

        Returns: a `Task` object containing info about the task.

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
                       n_splits = 10,
                       database="default"):
        """
        Run cox proportional hazard analysis on the entire dataset using cross-validation. Uses 10
        fold by default.

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
            n_splits: The number of folds to use for cross-validation. Default is 10.
            database: If the nodes have multiple datasources, indicate the label of the datasource
            you would like to use. Otherwise the default will be used.

        Returns: a `Task` object containing info about the task.
        """
        input_params = {
            "feature_columns": feature_columns,
            "event_times_column": outcome_time_column,
            "event_happened_column": right_censor_column,
            "datanode_ids": feature_nodes,
            "central_node_id": outcome_node,
            "convergence_precision": precision,
            "n_splits": n_splits,
        }

        return self._run_task(
            "cross_validate", True, [outcome_node], kwargs=input_params, database=database
        )

    def _run_task(
            self, method, master, organizations: List[int], kwargs=None, database="default"
    ):
        if kwargs is None:
            kwargs = {}
        kwargs["database"] = database
        # TODO: Construct description out of parameters
        description = ""
        name = "method"
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
            input_=task_input,
            databases=database,
        )

        match method:
            case "fit":
                return FitTask(self._v6client, task)
            case "cross_validate":
                return CrossValTask(self._v6client, task)
            case _:
                return Task(self._v6client, task)


@dataclass
class FitResult:
    """
    FitResult contains the result of a fit task. It contains the coefficients and the baseline
    hazard function.
    """
    coefs: Dict[str, float]
    baseline_hazard: HazardFunction

    @staticmethod
    def parse(results: List[Dict[str, any]]):
        # Assume that there is only one "partial" result
        content = json.loads(results[0]["result"])

        coefs = content["coefs"]
        baseline_hazard = HazardFunction(content["baseline_hazard_x"], content["baseline_hazard_y"])

        return FitResult(coefs, baseline_hazard)

    def plot(self):
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        ax[0].plot(self.baseline_hazard.x, self.baseline_hazard.y)
        ax[0].set_title("Baseline hazard")
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("hazard score")
        ax[1].bar(self.coefs.keys(), self.coefs.values(), label="coefficients")
        ax[1].set_title("Coefficients")


@dataclass
class CrossValResult:
    """
    CrossValResult contains the result of a cross-validation task. It contains the c-indices,
    coefficients and baseline hazard functions for each fold.
    """
    c_indices: List[float]
    coefs: List[Dict[str, float]]
    baseline_hazards: List[HazardFunction]

    @staticmethod
    def parse(partialResults: list[dict]):
        # Cross validation should only have one partial result
        result = partialResults[0]["result"]
        result = json.loads(result)
        c_indices, coefs, baseline_hazards = result
        baseline_hazards = [HazardFunction(*h) for h in baseline_hazards]

        return CrossValResult(c_indices, coefs, baseline_hazards)

    def plot(self):
        num_folds = len(self.c_indices)
        fig, ax = plt.subplots(num_folds, 2, constrained_layout=True)

        for fold in range(num_folds):
            ax[fold][0].plot(self.baseline_hazards[fold].x, self.baseline_hazards[fold].y)
            ax[fold][0].set_title(f"Baseline hazard fold {fold}")
            ax[fold][1].bar(self.coefs[fold].keys(), self.coefs[fold].values())
            ax[fold][1].set_title(f"Coefficients fold {fold}")


class Task:
    """
    Task is a wrapper around the vantage6 task object.
    """
    def __init__(self, client: Client, task_data):
        self._raw_data = task_data
        self.client = client
        self.task_id = task_data["id"]

    def get_results(self) -> PartialResult:
        """
        Get the results of the task. This will block until the task is finished.

        Returns:

        """
        results = self.client.wait_for_results(self.task_id)
        return self._parse_results(results["data"])


    @staticmethod
    def _parse_results(results) -> FitResult| CrossValResult:
        return results


class FitTask(Task):
    @staticmethod
    def _parse_results(results):
        return FitResult.parse(results)


class CrossValTask(Task):
    @staticmethod
    def _parse_results(results):
        return CrossValResult.parse(results)


class VerticoxClientException(Exception):
    pass
