import logging
from typing import List

import numpy as np
from numpy.typing import ArrayLike

from verticox.grpc.datanode_pb2 import UpdateRequest, Empty
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

logger = logging.getLogger(__name__)


class Aggregator:

    def __init__(self, institutions: List[DataNodeStub]):
        self.institutions = tuple(institutions)
        self.num_institutions = len(institutions)
        self.features_per_institution = self.get_features_per_institution()
        self.num_samples = self.get_num_samples()

        # Initializing parameters
        self.z = np.zeros(self.num_samples)
        self.gamma = np.zeros(self.num_samples)

    def fit(self):
        # TODO: I think sigma is supposed to be one element per sample because how else can we
        #  sum up all the numbers?
        sigma_per_institution = np.zeros((self.num_institutions, self.num_samples,))
        gamma_per_institution = np.zeros((self.num_institutions, self.num_samples,))

        # TODO: Parallelize
        for idx, institution in enumerate(self.institutions):
            request = UpdateRequest(z=self.z, gamma=self.gamma)
            updated = institution.update(request)
            sigma_per_institution[idx] = np.array(updated.sigma)
            gamma_per_institution[idx] = np.array(updated.gamma)

        sigma = self.aggregate_sigmas(sigma_per_institution)
        gamma = self.aggregate_gammas(gamma_per_institution)

    def aggregate_sigmas(self, sigmas: ArrayLike):
        return sigmas.sum() / self.num_institutions

    def aggregate_gammas(self, gammas: ArrayLike):
        return gammas.sum() / self.num_institutions

    def get_features_per_institution(self):
        num_features = []
        for institution in self.institutions:
            response = institution.getNumFeatures(Empty())
            num_features.append(response.numFeatures)

        return num_features

    def get_num_samples(self):
        """
        Get the number of samples in the dataset.
        This function assumes that the data in the datanodes has already been aligned and that they
        all have exactly the same amount of samples.

        The number of samples is determined by querying one of the institutions.
        """
        response = self.institutions[0].getNumSamples(Empty())
        return response.numSamples
