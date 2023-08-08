import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List, Any

import clize
import grpc
import numpy as np
import pandas as pd
from vantage6.tools.util import info

import verticox.ssl
from verticox.grpc.datanode_pb2 import LocalParameters, NumFeatures, \
    NumSamples, Empty, Beta, FeatureNames, RecordLevelSigma, AverageSigma, Subset, \
    PartialHazardRatio
from verticox.grpc.datanode_pb2_grpc import DataNodeServicer, add_DataNodeServicer_to_server
from verticox.scalarproduct import NPartyScalarProductClient

logger = logging.getLogger(__name__)
DEFAULT_PORT = 7777
GRACE = 30
TIMEOUT = 3600
DEFAULT_DATA = 'whas500'


class DataNode(DataNodeServicer):
    @dataclass
    class State:
        # The data belongs to the state because it can change based on which selection is activated
        # Parts that stay constant over iterations
        features: np.array
        feature_names: List[str]
        features_multiplied: np.array
        censor_name: str
        censor_value: Any
        beta: np.array = None
        rho: float = None
        sigma: float = None
        z: np.array = None
        gamma: np.array = None
        aggregated_gamma: np.array = None
        sum_Dt: np.array = None
        _prepared = False

        @property
        def prepared(self):
            return self._prepared

        @property
        def num_features(self):
            return self.features.shape[1]

        @property
        def num_samples(self):
            return self.features.shape[0]

        def prepare(self, gamma, z, beta, rho, sum_Dt):
            self.gamma = gamma
            self.z = z
            self.beta = beta
            self.rho = rho
            self.sum_Dt = sum_Dt

            self._prepared = True

    def __init__(self, features: np.array = None, feature_names: Optional[List[str]] = None,
                 name=None, server=None, include_column: Optional[str] = None,
                 include_value: bool = True,
                 commodity_address: str = None,
                 beta: np.array = None):
        """

        Args:
            features:
            event_times:
            rho:
        """

        self.name = name
        self._logger = logging.getLogger(f'{__name__} :: {self.name}')
        self._logger.setLevel(logging.DEBUG)
        self._logger.debug(f'Initializing datanode {self.name}')

        # Square all covariates and sum them together
        # The formula says for every patient, x needs to be multiplied by itself.
        # Squaring all covariates with themselves comes down to the same thing since x_nk is
        # supposed to be one-dimensional
        self.state = DataNode.State(features=features,
                                    feature_names=feature_names,
                                    features_multiplied=DataNode._multiply_features(features),
                                    censor_name=include_column,
                                    censor_value=include_value,
                                    beta=beta)

        self.server = server
        # Parts that stay constant over iterations

        self.n_party_address = commodity_address

    @staticmethod
    @np.vectorize
    def include(event):
        return event[0]

    def prepare(self, request, context=None):
        sum_Dt = self.compute_sum_Dt_n_party_scalar_product(self.state.feature_names,
                                                            self.state.censor_name,
                                                            self.state.censor_value,
                                                            self.n_party_address)

        self.state.prepare(gamma=np.full((self.state.num_samples,), request.gamma),
                           z=np.full((self.state.num_samples,), request.z),
                           beta=np.full((self.state.num_features,), request.beta),
                           rho=request.rho,
                           sum_Dt=sum_Dt)

        return Empty()

    @staticmethod
    def compute_sum_Dt(Dt, features):
        logger.debug('Computing sum dt locally')
        # TODO: I think this can be simpler but for debugging's sake I will follow the paper
        result = np.zeros((features.shape[1]))

        for t, indices in Dt.items():
            for i in indices:
                result = result + features[i]

        return result

    @staticmethod
    def compute_sum_Dt_n_party_scalar_product(local_feature_names, censor_feature,
                                              censor_value, commodity_address):
        logger.info('Computing sum Dt with n party scalar product')
        client = NPartyScalarProductClient(commodity_address=commodity_address)

        result = client.sum_relevant_values(local_feature_names, censor_feature,
                                            censor_value)

        assert len(result) == len(local_feature_names)

        return np.array(result, dtype=float)

    def fit(self, request, context=None):
        if not self.state.prepared:
            raise Exception('Datanode has not been prepared!')

        self._logger.debug('Performing local update...')

        sigma, beta = DataNode._local_update(self.state.features, self.state.z, self.state.gamma,
                                             self.state.rho, self.state.features_multiplied,
                                             self.state.sum_Dt)

        self.state.sigma = sigma
        self.state.beta = beta

        response = LocalParameters(gamma=self.state.gamma.tolist(), sigma=sigma.tolist())
        self._logger.debug('Finished local update, returning results.')

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
        self.state.gamma = self.state.aggregated_gamma + self.state.rho * \
                           (self.state.sigma - self.state.z)

        return Empty()

    def getNumFeatures(self, request, context=None):
        num_features = self.state.num_features
        return NumFeatures(numFeatures=num_features)

    def getNumSamples(self, request, context=None):
        num_samples = self.state.num_samples

        return NumSamples(numSamples=num_samples)

    def getBeta(self, request, context=None):
        self._logger.debug('Returning beta')
        result = self.state.beta.tolist()
        self._logger.debug('Converted beta to list')
        return Beta(beta=result)

    def kill(self, request, context=None):
        self.server.stop(GRACE)

        return Empty()

    def getFeatureNames(self, request, context: grpc.ServicerContext):
        if self.state.feature_names is not None:
            unicode_names = [n.encode('utf-8') for n in self.state.feature_names]
            return FeatureNames(names=unicode_names)
        else:
            context.abort(grpc.StatusCode.NOT_FOUND, 'This datanode does not have feature names')

    def getRecordLevelSigma(self, request,
                            context: grpc.ServicerContext = None) -> RecordLevelSigma:
        """
        Get the sigma value for every record. Sigma is defined as :math: `\beta_k \cdot x`

        :param request:
        :param context:
        :return:
        """

        sigmas = np.tensordot(self.state.features, self.state.beta, (1, 0))

        return RecordLevelSigma(sigma=sigmas)

    def getAverageSigma(self, request: Empty, context=None) -> AverageSigma:
        """
        Get sigma value averaged over all records.
        :param request:
        :param context:
        :return:
        """
        # TODO: We need to select a subpopulation

        sigmas = np.tensordot(self.state.features, self.state.beta, (1, 0))
        average = np.average(sigmas, axis=0)

        return AverageSigma(sigma=average)

    def computePartialHazardRatio(self, request: Subset, context=None) -> PartialHazardRatio:
        """
        Compute the hazard ratio of a subset of records
        :param request: The indices of the subset of records
        :param context:
        :return:
        """
        indices = list(request.indices)
        subset = self.state.features[indices, :]
        sigmas = np.tensordot(subset, self.state.beta, (1, 0))

        return PartialHazardRatio(partialHazardRatios=sigmas.tolist())

    @staticmethod
    def _sum_covariates(covariates: np.array):
        return np.sum(covariates, axis=0)

    @staticmethod
    def _multiply_features(features: np.array):
        return np.matmul(features.T, features)
        # return np.square(features).sum()

    @staticmethod
    def _compute_beta(features: np.array, z: np.array, gamma: np.array, rho,
                      features_multiplied, sum_Dt):
        first_component = np.linalg.inv(rho * features_multiplied)

        second_component = np.zeros((features.shape[1],))

        for sample_idx in range(features.shape[0]):
            second_component = second_component + \
                               (rho * z[sample_idx] - gamma[sample_idx]) * features[sample_idx]

        second_component = second_component + sum_Dt

        return np.matmul(first_component, second_component)

    @staticmethod
    def _compute_sigma(beta, features):
        sigma = np.zeros((features.shape[0]))
        for n in range(features.shape[0]):
            sigma[n] = np.dot(beta, features[n])

        return sigma

    @staticmethod
    def _local_update(features: np.array, z: np.array, gamma: np.array, rho,
                      features_multiplied, sum_Dt):
        beta = DataNode._compute_beta(features, z, gamma, rho, features_multiplied, sum_Dt)

        return DataNode._compute_sigma(beta, features), beta


def serve(*, data: np.array, feature_names=None,
          include_column=None, include_value=True,
          commodity_address=None, port=DEFAULT_PORT,
          timeout=TIMEOUT, secure=True, address=None):
    logging.basicConfig(level=logging.DEBUG)
    info(f'Data shape {data.shape}')
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    add_DataNodeServicer_to_server(
        DataNode(features=data, feature_names=feature_names, include_column=include_column,
                 include_value=include_value, commodity_address=commodity_address, server=server),
        server)

    server_endpoint = f'[::]:{port}'

    if secure:
        private_key, cert_chain = verticox.ssl.generate_self_signed_certificate(address)
        server_credentials = grpc.ssl_server_credentials(((private_key, cert_chain),))

        server.add_secure_port(server_endpoint, server_credentials)

    else:
        server.add_insecure_port(server_endpoint)
    info(f'Starting datanode on port {port} with timeout {timeout}')
    before = time.time()
    server.start()
    server.wait_for_termination(timeout=timeout)
    total_time = time.time() - before
    info(f'Stopped datanode after {total_time} seconds')


def serve_standalone(*, data_path, address, commodity_address, include_column):
    data = pd.read_parquet(data_path)
    serve(data=data.values, feature_names=data.columns, address=address,
          commodity_address=commodity_address, include_column=include_column)


if __name__ == '__main__':
    clize.run(serve_standalone)
