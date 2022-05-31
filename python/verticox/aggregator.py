import logging
from typing import List, Dict

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from verticox.grpc.datanode_pb2 import Empty, AggregatedParameters
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

logger = logging.getLogger(__name__)

RHO = 0.25
E = 0.1


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


class Aggregator:
    """
    Central node that aggregates the result of the datanodes and coordinates calculations

    TODO: This implementation is matching the paper quite closely. Unfortunately it relies a lot on
            state. This will have to be fixed later.
    """

    def __init__(self, institutions: List[DataNodeStub], event_times: ArrayLike,
                 right_censored: ArrayLike, e: float = E):
        """
        Initialize regular verticox aggregator. Note that this type of aggregator needs access to the
        event times of the samples.

        Args:
            institutions:
            event_times:
            right_censored:
            e: threshold value
        """
        self.e = e
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
        self.sigma = np.zeros(self.num_samples)
        self.sigma_per_institution = np.zeros((self.num_institutions, self.num_samples,))
        self.gamma_per_institution = np.zeros((self.num_institutions, self.num_samples,))
        self.num_iterations = 0

    def fit(self):
        z_old = self.z

        while True:
            self.fit_one()

            # Turning the while in the paper into a do-until type thing. This means I have to
            # flip the > sign
            if np.linalg.norm(self.z - z_old) <= e and np.linalg.norm(self.z - self.sigma) <= \
                    self.e:
                break

        logger.info(f'Finished training after {self.num_iterations} iterations')

    def fit_one(self):
        # TODO: Parallelize
        for idx, institution in enumerate(self.institutions):
            updated = institution.fit(Empty())
            self.sigma_per_institution[idx] = np.array(updated.sigma)
            self.gamma_per_institution[idx] = np.array(updated.gamma)

        self.sigma = self.aggregate_sigmas(self.sigma_per_institution)
        self.gamma = self.aggregate_gammas(self.gamma_per_institution)

        z = Lz.find_z(self.num_institutions, self.gamma, self.sigma, self.rho, self.Rt, self.z)

        z_per_institution = self.compute_z_per_institution(self.gamma_per_institution,
                                                           self.sigma_per_institution, z)

        # Update parameters at datanodes
        for idx, node in enumerate(self.institutions):
            params = AggregatedParameters(gamma=self.gamma.tolist(), sigma=self.sigma.tolist(),
                                          z=z_per_institution[idx].tolist())
            node.updateParameters(params)
            node.computeGamma(Empty())

        self.num_iterations += 1
        logger.debug(f'Num iterations: {self.num_iterations}')

    def compute_z_per_institution(self, gamma_per_institution, sigma_per_institution, z):
        """
        Equation 11
        Args:
            gamma_per_institution:
            sigma_per_institution:
            z:

        Returns:

        """
        z_per_institution = np.zeros((self.num_institutions, self.num_samples))
        sigma_gamma_all_institutions = sigma_per_institution + gamma_per_institution / self.rho
        sigma_gamma_all_institutions.sum(axis=0)
        # TODO: vectorize
        for i in range(self.num_institutions):
            z_per_institution[i] = z + sigma_per_institution[i] + gamma_per_institution / self.rho \
                                   - 1 / self.num_institutions * sigma_gamma_all_institutions

        return z_per_institution

    def aggregate_sigmas(self, sigmas: ArrayLike):
        return sigmas.sum(axis=0) / self.num_institutions

    def aggregate_gammas(self, gammas: ArrayLike):
        return gammas.sum(axis=0) / self.num_institutions

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


class Lz:

    @staticmethod
    def parametrized(z: np.array, K: int, gamma: np.array, sigma: float, rho: float, Rt: ArrayLike):
        """
        Equation 12
        Args:
            z:
            K:
            gamma:
            sigma:
            rho:
            Rt:

        Returns:

        """
        dt = len(Rt)

        component1 = Lz.component1(z, K, Rt, dt)
        component2 = Lz.component2(z, K, sigma, gamma, rho)

        return component1 + component2

    @staticmethod
    def component1(z, K, Rt, dt):
        result = 0
        for t, group in Rt.items():
            z_at_risk = z[group]
            result += dt * (K * np.exp(z_at_risk)).sum()

        return result

    @staticmethod
    def component2(z, K, sigma, gamma, rho):
        element_wise = np.square(z) / 2 - sigma + (gamma / rho) * z
        return K * rho * element_wise.sum()

    #
    # @staticmethod
    # def compute_first_order_derivative(z: ArrayLike, ):
    #

    @staticmethod
    def find_z(num_parties, gamma: ArrayLike, sigma: ArrayLike,
               rho: float, Rt: Dict[int, List[int]], z_start: ArrayLike):
        logger.debug('Creating L(z) with parameters:')
        logger.debug(f'num_parties: {num_parties}  gamma: {gamma}, sigma: {sigma} rho: {rho} '
                     f'Rt: {list(Rt.keys())[:5]}...')
        L_z = lambda z: Lz.parametrized(z=z, K=num_parties, gamma=gamma, sigma=sigma, rho=rho,
                                        Rt=Rt)

        logger.debug(f'Finding minimum z starting at {z_start[:5]}...')

        minimum = minimize(L_z, z_start)

        logger.debug(f'Found minimum z at {minimum}')
        return minimum.x
