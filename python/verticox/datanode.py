import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List, Union

import clize
import grpc
import numpy as np
import pandas as pd
from vantage6.tools.util import info

import verticox.ssl
from verticox.common import Split
from verticox.grpc.datanode_pb2 import (
    LocalParameters,
    NumFeatures,
    NumSamples,
    Empty,
    Beta,
    FeatureNames,
    RecordLevelSigma,
    AverageSigma,
    Subset,
    InitialValues,
    Rows,
    RecordLevelSigmaRequest,
    AverageSigmaRequest,
)
from verticox.grpc.datanode_pb2_grpc import (
    DataNodeServicer,
    add_DataNodeServicer_to_server,
)
from verticox.scalarproduct import NPartyScalarProductClient

logger = logging.getLogger(__name__)
DEFAULT_PORT = 7777
GRACE = 30
TIMEOUT = 3600 * 24
DEFAULT_DATA = "whas500"


class DataNodeException(Exception):
    pass


class DataNode(DataNodeServicer):
    @dataclass
    class State:
        # The data belongs to the state because it can change based on which selection is activated
        # Parts that stay constant over iterations
        features_multiplied: np.array
        beta: np.array
        rho: float
        z: np.array
        gamma: np.array
        sum_Dt: np.array
        aggregated_gamma: np.array = None
        sigma: float = None

    def __init__(
            self,
            all_features: np.array,
            feature_names: List[str],
            name=None,
            server=None,
            include_column: Optional[str] = None,
            include_value: bool = True,
            commodity_address: str = None,
    ):
        """

        Args:
            all_features:
            event_times:
            rho:
        """

        self.name = name
        self._logger = logging.getLogger(f"{__name__} :: {self.name}")
        self._logger.debug(f"Initializing datanode {self.name}")
        self._all_data = all_features
        self._censor_name = include_column
        self._censor_value = include_value

        self.feature_names = feature_names
        self.server = server
        # Parts that stay constant over iterations

        self.n_party_address = commodity_address

        self.state: Union[DataNode.State, None] = None
        self.split: Union[Split, None] = None

    @property
    def num_features(self):
        return self._all_data.shape[1]

    @property
    def prepared(self):
        return self.state is not None

    @staticmethod
    @np.vectorize
    def include(event):
        return event[0]

    def _select_train_test(self, selected_rows) -> Split[np.array, np.array, np.array]:
        """

        Args:
            selected_rows:

        Returns: Tuple containing (TRAIN_DATA, TEST_DATA)

        """
        if not selected_rows:
            return Split(self._all_data, self._all_data, self._all_data)

        mask = np.zeros(self._all_data.shape[0], dtype=bool)
        mask[selected_rows] = True
        return Split(self._all_data[mask], self._all_data[~mask], self._all_data)

    def prepare(self, request: InitialValues, context=None):
        """
        Set initial values for running verticox
        Args:
            request:
            context:

        Returns:

        """
        self._prepare(request.gamma, request.z, request.beta, request.rho)

        return Empty()

    def _prepare(self, gamma, z, beta, rho):
        num_samples = self.split.train.shape[0]
        covariates_multiplied = DataNode._multiply_features(self.split.train)
        sum_Dt = self._compute_sum_Dt_n_party_scalar_product(
            self.feature_names,
            self._censor_name,
            self._censor_value,
            self.n_party_address,
        )
        self.state = DataNode.State(
            features_multiplied=covariates_multiplied,
            gamma=np.array(gamma),
            z=np.array(z),
            beta=np.array(beta),
            rho=rho,
            sum_Dt=sum_Dt,
        )

    def reset(self, request: Rows, context=None):
        """
        Reset state and reselect active records
        Args:
            request:
            context:

        Returns:

        """
        self.state = None

        rows = request.rows

        self.split = self._select_train_test(rows)

        return Empty()

    @staticmethod
    def _compute_sum_Dt(Dt, features):
        logger.debug("Computing sum dt locally")
        # TODO: I think this can be simpler but for debugging's sake I will follow the paper
        result = np.zeros((features.shape[1]))

        for t, indices in Dt.items():
            for i in indices:
                result = result + features[i]

        return result

    @staticmethod
    def _compute_sum_Dt_n_party_scalar_product(
            local_feature_names, censor_feature, censor_value, commodity_address
    ):
        logger.info("Computing sum Dt with n party scalar product")
        client = NPartyScalarProductClient(commodity_address=commodity_address)

        result = client.sum_relevant_values(
            local_feature_names, censor_feature, censor_value
        )

        return np.array(result, dtype=float)

    def fit(self, request, context=None):
        if not self.prepared:
            raise DataNodeException("Datanode has not been prepared!")

        self._logger.debug("Performing local update...")

        sigma, beta = DataNode._local_update(
            self.split.train,
            self.state.z,
            self.state.gamma,
            self.state.rho,
            self.state.features_multiplied,
            self.state.sum_Dt,
        )

        self.state.sigma = sigma
        self.state.beta = beta

        response = LocalParameters(
            gamma=self.state.gamma.tolist(), sigma=sigma.tolist()
        )
        self._logger.debug("Finished local update, returning results.")

        return response

    def updateParameters(self, request, context=None):
        self.state.z = np.array(request.z)
        self.state.sigma = np.array(request.sigma)
        self.state.aggregated_gamma = np.array(request.gamma)

        return Empty()

    def computeGamma(self, request, context=None):
        """
        Equation 18
        Args:
            request:
            context:

        Returns:

        """
        self.state.gamma = self.state.aggregated_gamma + self.state.rho * (
                self.state.sigma - self.state.z
        )

        return Empty()

    def getNumFeatures(self, request, context=None):
        num_features = self.num_features
        return NumFeatures(numFeatures=num_features)

    def getNumSamples(self, request, context=None):
        if self.prepared:
            num_samples = self.split.train.shape[0]
            return NumSamples(numSamples=num_samples)

        raise DataNodeException("Datanode has not been prepared yet!")

    def getBeta(self, request, context=None):
        result = self.state.beta.tolist()
        return Beta(beta=result)

    def kill(self, request, context=None):
        self.server.stop(GRACE)

        return Empty()

    def getFeatureNames(self, request, context: grpc.ServicerContext):
        if self.feature_names is not None:
            unicode_names = [n.encode("utf-8") for n in self.feature_names]
            return FeatureNames(names=unicode_names)
        else:
            context.abort(
                grpc.StatusCode.NOT_FOUND, "This datanode does not have feature names"
            )

    def getRecordLevelSigma(
            self, request: RecordLevelSigmaRequest, context: grpc.ServicerContext = None
    ) -> RecordLevelSigma:
        """
        Get the sigma value for every record. Sigma is defined as :math: `\beta_k \cdot x`

        :param request:
        :param context:
        :return:
        """
        data = self.retrieve_subset(request.subset)

        sigmas = DataNode.compute_record_level_sigma(data, self.state.beta)

        return RecordLevelSigma(sigma=sigmas)

    def retrieve_subset(self, subset: Subset):
        match subset:
            case Subset.TEST:
                data = self.split.test
            case Subset.TRAIN:
                data = self.split.train
            case Subset.ALL:
                data = self.split.all
        return data

    @staticmethod
    def compute_record_level_sigma(covariates, beta):
        return np.tensordot(covariates, beta, (1, 0))

    def getAverageSigma(
            self, request: AverageSigmaRequest, context=None
    ) -> AverageSigma:
        """
        Get sigma value averaged over all records.
        :param request:
        :param context:
        :return:
        """
        if request.subset == Subset.TRAIN:
            average = DataNode.compute_average_sigma(self.split.train, self.state.beta)
        elif request.subset == Subset.TEST:
            average = DataNode.compute_average_sigma(self.split.test, self.state.beta)
        else:
            average = DataNode.compute_average_sigma(self._all_data, self.state.beta)

        return AverageSigma(sigma=average)

    @staticmethod
    def compute_average_sigma(features: np.array, beta: np.array):
        average_covariates = features.mean(axis=0)
        sigmas = np.dot(average_covariates, beta)
        return sigmas

    @staticmethod
    def _sum_covariates(covariates: np.array):
        return np.sum(covariates, axis=0)

    @staticmethod
    def _multiply_features(features: np.array):
        return np.matmul(features.T, features)
        # return np.square(features).sum()

    @staticmethod
    def _compute_beta(
            features: np.array,
            z: np.array,
            gamma: np.array,
            rho,
            features_multiplied,
            sum_Dt,
    ):
        first_component = np.linalg.inv(rho * features_multiplied)

        second_component = np.zeros((features.shape[1],))

        for sample_idx in range(features.shape[0]):
            second_component = (
                    second_component
                    + (rho * z[sample_idx] - gamma[sample_idx]) * features[sample_idx]
            )

        second_component = second_component + sum_Dt

        return np.matmul(first_component, second_component)

    @staticmethod
    def _compute_sigma(beta, features):
        sigma = np.zeros((features.shape[0]))
        for n in range(features.shape[0]):
            sigma[n] = np.dot(beta, features[n])

        return sigma

    @staticmethod
    def _local_update(
            features: np.array,
            z: np.array,
            gamma: np.array,
            rho,
            features_multiplied,
            sum_Dt,
    ):
        beta = DataNode._compute_beta(
            features, z, gamma, rho, features_multiplied, sum_Dt
        )

        return DataNode._compute_sigma(beta, features), beta


def serve(
        *,
        data: np.array,
        feature_names=None,
        include_column=None,
        include_value=True,
        commodity_address=None,
        port=DEFAULT_PORT,
        timeout=TIMEOUT,
        secure=True,
        address=None,
        verbose=False
):
    if verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(level=loglevel)
    info(f"Data shape {data.shape}")
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    add_DataNodeServicer_to_server(
        DataNode(
            all_features=data,
            feature_names=feature_names,
            include_column=include_column,
            include_value=include_value,
            commodity_address=commodity_address,
            server=server,
        ),
        server,
    )

    server_endpoint = f"[::]:{port}"

    if secure:
        private_key, cert_chain = verticox.ssl.generate_self_signed_certificate(address)
        server_credentials = grpc.ssl_server_credentials(((private_key, cert_chain),))

        server.add_secure_port(server_endpoint, server_credentials)

    else:
        server.add_insecure_port(server_endpoint)
    info(f"Starting datanode on port {port} with timeout {timeout}")
    before = time.time()
    server.start()
    server.wait_for_termination(timeout=timeout)
    total_time = time.time() - before
    info(f"Stopped datanode after {total_time} seconds")


def serve_standalone(*, data_path, address, commodity_address, include_column):
    data = pd.read_parquet(data_path)
    serve(
        data=data.values,
        feature_names=data.columns,
        address=address,
        commodity_address=commodity_address,
        include_column=include_column,
    )


if __name__ == "__main__":
    clize.run(serve_standalone)
