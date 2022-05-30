import logging
from typing import List, Dict
from scipy.optimize import minimize
import numpy as np
from numpy.typing import ArrayLike

from verticox.grpc.datanode_pb2 import UpdateRequest, Empty
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

logger = logging.getLogger(__name__)

RHO = 0.25


def group_samples_at_risk(event_times: ArrayLike, right_censored: ArrayLike):
    """
    Groups the indices of samples on whether they are at risk at a certain time.

    A sample is at risk at a certain time when its event time is greater or equal that time.

    TODO: Figure out what to do with right-censored samples
    Args:
        event_times:
        right_censored:

    Returns:

    """
    unique_times = np.unique(event_times)

    grouped = {}

    for t in unique_times:
        grouped[t] = np.argwhere(event_times >= t).flatten()

    return grouped


def L_z_parametrized(z: np.array, K: int, gamma: np.array, sigma, rho, Rt):
    dt = len(Rt)

    component1 = L_z_component1(z, K, Rt, dt)
    component2 = L_z_component2(z, K, sigma, gamma, rho)

    return component1 + component2


def L_z_component1(z, K, Rt, dt):
    result = 0
    for t, group in Rt.items():
        z_at_risk = z[group]
        result += dt * (K * np.exp(z_at_risk)).sum()

    return result


def L_z_component2(z, K, sigma, gamma, rho):
    element_wise = np.square(z) / 2 - sigma + (gamma / rho) * z
    return K * rho * element_wise.sum()


def find_z(num_parties, gamma: ArrayLike, sigma: ArrayLike, rho: float,
           samples_at_risk: Dict[int, List],
           z_start: ArrayLike):
    logger.debug('Creating L(z) with parameters:')
    logger.debug(f'num_parties: {num_parties}  gamma: {gamma}, sigma: {sigma} rho: {rho} '
                 f'samples_at_risk: {samples_at_risk.popitem()}...')
    L_z = lambda z: L_z_parametrized(z, num_parties, gamma, sigma, rho, samples_at_risk)

    logger.debug(f'Finding minimum z starting at {z_start[:5]}...')

    minimum = minimize(L_z, z_start)

    logger.debug(f'Found minimum z at {minimum}')
    return minimum.x


class Aggregator:

    def __init__(self, institutions: List[DataNodeStub], event_times: ArrayLike,
                 right_censored: ArrayLike):
        """
        Initialize regular verticox aggregator. Note that this type of aggregator needs access to the
        event times of the samples.

        Args:
            institutions:
            event_times:
        """

        self.institutions = tuple(institutions)
        self.num_institutions = len(institutions)
        self.features_per_institution = self.get_features_per_institution()
        self.num_samples = self.get_num_samples()
        self.rho = RHO  # TODO: Make dynamic
        self.event_times = event_times
        self.right_censored = right_censored
        self.Rt = group_samples_at_risk(event_times, right_censored)
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

        z = find_z(self.num_institutions, gamma, sigma, self.rho, self.Rt, self.z)

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
