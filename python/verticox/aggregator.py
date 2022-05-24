import logging
from typing import List

import grpc
import numpy as np

from verticox.grpc.datanode_pb2 import UpdateRequest, Empty
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

logger = logging.getLogger(__name__)

NUM_PATIENTS = 5
NUM_COVARATES = 10


class Aggregator:

    def __init__(self, institutions):
        self.institutions = institutions

        self.features_per_institution = Aggregator.get_features_per_institution(institutions)

        # Initializing parameters
        self.z = np.zeros(NUM_PATIENTS)
        self.gamma = np.zeros(NUM_PATIENTS)

    def fit(self):
        gamma_per_institution = np.array
        for idx, institution in enumerate(self.institutions):
            request = UpdateRequest(z=self.z, gamma=self.gamma)
            updated = institution.update(request)

    def get_features_per_institution(self):
        num_features = []
        for institution in self.institutions:
            num_features.append(institution.getNumFeatures(Empty()))

        return num_features

    def get_num_samples(self):
        """
        Get the number of samples in the dataset.
        This function assumes that the data in the datanodes has already been aligned and that they
        all have exactly the same amount of samples.

        The number of samples is determined by querying one of the institutions.
        """

