import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import grpc
import numpy as np
from sksurv.datasets import load_whas500

from verticox.grpc.datanode_pb2 import LocalAuxiliaries, NumFeatures
from verticox.grpc.datanode_pb2_grpc import DataNodeServicer, add_DataNodeServicer_to_server

logger = logging.getLogger(__name__)

RHO = 0.25
PORT = 8888
MAX_WORKERS = 1


class DataNode(DataNodeServicer):
    def __init__(self, features: np.array = None, events: Optional[np.array] = None,
                 rho: float = RHO):
        """

        Args:
            features:
            events:
            rho:
        """
        self.features = features
        self.events = events

        self.rho = rho
        # Parts that stay constant over iterations
        # Square all covariates and sum them together
        # The formula says for every patient, x needs to be multiplied by itself.
        # Squaring all covariates with themselves comes down to the same thing since x_nk is supposed to
        # be one-dimensional
        self.features_multiplied = DataNode.multiply_covariates(features)
        self.features_sum = features.sum(axis=0)

    def update(self, request, context=None):
        logger.info('Performing local update...')
        sigma = DataNode.local_update(self.features,
                                      np.array(request.z),
                                      np.array(request.gamma),
                                      self.rho, self.features_multiplied, self.features_sum)

        # TODO: Kind of pointless to send back the same gamma, will probably remove this later
        response = LocalAuxiliaries(gamma=request.gamma, sigma=sigma.tolist())
        logger.info('Finished local update, returning results.')

        return response

    def getNumFeatures(self, request, context=None):
        num_features = self.features.shape[1]
        return NumFeatures(numFeatures=num_features)



    @staticmethod
    def sum_covariates(covariates: np.array):
        return np.sum(covariates, axis=0)

    @staticmethod
    def multiply_covariates(features: np.array):
        return np.square(features).sum()

    @staticmethod
    def elementwise_multiply_sum(one_dim: np.array, two_dim: np.array):
        """
        Every element in one_dim does elementwise multiplication with its corresponding row in two_dim.

        All rows of the result will be summed together vertically.
        """
        multiplied = np.zeros(two_dim.shape)
        for i in range(one_dim.shape[0]):
            multiplied[i] = one_dim[i] * two_dim[i]

        return multiplied.sum(axis=0)

    @staticmethod
    def compute_beta(features: np.array, z: np.array, gamma: np.array, rho,
                     multiplied_covariates, covariates_sum):
        first_component = 1 / (rho * multiplied_covariates)

        pz = rho * z

        second_component = \
            DataNode.elementwise_multiply_sum(pz - gamma, features) + covariates_sum

        return second_component / first_component

    @staticmethod
    def compute_sigma(beta, covariates):
        return np.matmul(covariates, beta)

    @staticmethod
    def local_update(covariates: np.array, z: np.array, gamma: np.array, rho,
                     covariates_multiplied, covariates_sum):
        beta = DataNode.compute_beta(covariates, z, gamma, rho, covariates_multiplied,
                                     covariates_sum)

        return DataNode.compute_sigma(beta, covariates)


def serve():
    features, events = load_whas500()
    features = features.values.astype(float)
    server = grpc.server(ThreadPoolExecutor(max_workers=MAX_WORKERS))
    add_DataNodeServicer_to_server(DataNode(features=features, events=events), server)
    server.add_insecure_port(f'[::]:{PORT}')
    logger.info(f'Starting datanode on port {PORT}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    serve()
