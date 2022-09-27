import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

import grpc
import numpy as np
from vantage6.tools.util import info

from verticox.common import group_samples_on_event_time
from verticox.grpc.datanode_pb2 import LocalParameters, NumFeatures, \
    NumSamples, Empty, Beta, FeatureNames
from verticox.grpc.datanode_pb2_grpc import DataNodeServicer, add_DataNodeServicer_to_server
from verticox.scalarproduct import NPartyScalarProductClient, NPartyParameters

logger = logging.getLogger(__name__)
DEFAULT_PORT = 7777
GRACE = 30
TIMEOUT = 3600


class DataNode(DataNodeServicer):
    def __init__(self, features: np.array = None, feature_names: Optional[List[str]] = None,
                 event_times: Optional[np.array] = None,
                 event_happened: Optional[np.array] = None, name=None, server=None,
                 censor_name: Optional[str] = None, censor_value: bool = True,
                 n_party_address: str = None):
        """

        Args:
            features:
            event_times:
            rho:
        """

        self.name = name
        self._logger = logging.getLogger(f'{__name__} :: {self.name}')
        self._logger.debug(f'Initializing datanode {self.name}')
        self.features = features
        self.feature_names = feature_names
        self.num_features = self.features.shape[1]
        self.event_times = event_times
        self.event_happened = event_happened
        self.server = server
        # Parts that stay constant over iterations
        # Square all covariates and sum them together
        # The formula says for every patient, x needs to be multiplied by itself.
        # Squaring all covariates with themselves comes down to the same thing since x_nk is
        # supposed to be one-dimensional
        self.features_multiplied = DataNode._multiply_features(features)

        self.num_samples = self.features.shape[0]
        self.censor_name = censor_name
        self.censor_value = censor_value
        self.n_party_address = n_party_address

        self.rho = None
        self.beta = None
        self.sigma = None
        self.z = None
        self.gamma = None
        self.aggregated_gamma = None
        self.Dt = None
        self.sum_Dt = None
        self.prepared = False


    @staticmethod
    @np.vectorize
    def include(event):
        return event[0]

    def prepare(self, request, context=None):
        self.gamma = np.full((self.num_samples,), request.gamma)
        self.z = np.full((self.num_samples,), request.z)
        self.beta = np.full((self.num_features,), request.beta)
        self.rho = request.rho
        self.Dt = group_samples_on_event_time(self.event_times, self.event_happened)
        #self.sum_Dt = self.compute_sum_Dt(self.Dt, self.features)
        self.sum_Dt = self.compute_sum_Dt_n_party_scalar_product(self.feature_names,
                                                                 self.censor_name,
                                                                 self.censor_value,
                                                                 self.n_party_parameters)

        self.prepared = True

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
        logger.debug('Computing sum Dt with n party scalar product')
        client = NPartyScalarProductClient(commodity_address=commodity_address)

        result = np.zeros(len(local_feature_names))

        for feature_idx, feature_name in enumerate(local_feature_names):
            result[feature_idx] = client.sum_relevant_values(local_feature_names, censor_feature,
                                                   censor_value)

        return result

    def fit(self, request, context=None):
        if not self.prepared:
            raise Exception('Datanode has not been prepared!')

        self._logger.info('Performing local update...')
        sigma, beta = DataNode._local_update(self.features, self.z, self.gamma, self.rho,
                                             self.features_multiplied, self.sum_Dt)

        self.sigma = sigma
        self.beta = beta

        self._logger.debug(f'Beta: {json.dumps(self.beta.tolist())}')

        response = LocalParameters(gamma=self.gamma.tolist(), sigma=sigma.tolist())
        self._logger.info('Finished local update, returning results.')

        return response

    def updateParameters(self, request, context=None):
        self.z = np.array(request.z)
        self.sigma = np.array(request.sigma)
        self.aggregated_gamma = np.array(request.gamma)

        return Empty()

    def computeGamma(self, request, context=None):
        """
        Equation 18
        Args:
            request:
            context:

        Returns:

        """
        self.gamma = self.aggregated_gamma + self.rho * (self.sigma - self.z)

        return Empty()

    def getNumFeatures(self, request, context=None):
        num_features = self.num_features
        return NumFeatures(numFeatures=num_features)

    def getNumSamples(self, request, context=None):
        num_samples = self.num_samples

        return NumSamples(numSamples=num_samples)

    def getBeta(self, request, context=None):
        self._logger.debug(f'Returning beta')
        result = self.beta.tolist()
        self._logger.debug('Converted beta to list')
        return Beta(beta=result)

    def kill(self, request, context=None):
        self.server.stop(GRACE)

        return Empty()

    def getFeatureNames(self, request, context: grpc.ServicerContext):
        if self.feature_names is not None:
            unicode_names = [n.encode('utf-8') for n in self.feature_names]
            return FeatureNames(names=unicode_names)
        else:
            context.abort(grpc.StatusCode.NOT_FOUND, 'This datanode does not have feature names')

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


def serve(features=None, feature_names=None, event_times=None, event_happened=None,
          port=DEFAULT_PORT, timeout=TIMEOUT):
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    add_DataNodeServicer_to_server(
        DataNode(features=features, feature_names=feature_names, event_times=event_times,
                 event_happened=event_happened, server=server), server)
    server.add_insecure_port(f'[::]:{port}')
    info(f'Starting datanode on port {port} with timeout {timeout}')
    before = time.time()
    server.start()
    server.wait_for_termination(timeout=timeout)
    total_time = time.time() - before
    info(f'Stopped datanode after {total_time} seconds')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    serve()
