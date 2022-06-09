import logging
from typing import List, Dict, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
import asyncio
from verticox.grpc.datanode_pb2 import Empty, AggregatedParameters
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

logger = logging.getLogger(__name__)

RHO = 0.25
E = 0.001
OPTIMIZATION_METHOD = 'TNC'
ARRAY_LOG_LIMIT = 5


def group_samples_at_risk(event_times: ArrayLike,
                          right_censored: ArrayLike) -> Dict[Union[float, int], List[int]]:
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
        self.z_old = self.z
        self.gamma = np.zeros(self.num_samples)
        self.sigma = np.zeros(self.num_samples)
        self.gamma_per_institution = np.zeros((self.num_institutions, self.num_samples,))
        self.num_iterations = 0

    def fit(self):

        while True:
            self.fit_one()

            # Turning the while in the paper into a do-until type thing. This means I have to
            # flip the > sign
            if np.linalg.norm(self.z - self.z_old) <= self.e and \
                    np.linalg.norm(self.z - self.sigma) <= \
                    self.e:
                break

        logger.info(f'Finished training after {self.num_iterations} iterations')

    def fit_one(self):
        # TODO: Parallelize
        sigma_per_institution = np.zeros((self.num_institutions, self.num_samples,))

        for idx, institution in enumerate(self.institutions):
            updated = institution.fit(Empty())
            sigma_per_institution[idx] = np.array(updated.sigma)
            self.gamma_per_institution[idx] = np.array(updated.gamma)

        self.sigma = self.aggregate_sigmas(sigma_per_institution)
        self.gamma = self.aggregate_gammas(self.gamma_per_institution)

        logger.info(f'Updated sigma: {self.sigma}')
        logger.info(f'Updated gamma: {self.gamma}')

        self.z_old = self.z
        self.z = Lz.find_z(self.num_institutions, self.gamma, self.sigma, self.rho, self.Rt, self.z,
                           self.num_institutions, self.event_times)

        z_per_institution = self.compute_z_per_institution(self.gamma_per_institution,
                                                           sigma_per_institution, self.z)

        logger.debug(f'z per institution: {z_per_institution}')

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
        sigma_gamma_all_institutions = sigma_gamma_all_institutions.sum(axis=0)
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

    def get_betas(self):
        betas = []
        for institution in self.institutions:
            betas.append(institution.getBeta())

        return np.array(betas)


class Lz:
    # TODO: It might be easier if the fixed parameters are class variables
    # TODO: Vectorizing might make things easier
    @staticmethod
    def parametrized(z: ArrayLike, K: int, gamma: ArrayLike, sigma: ArrayLike, rho: float,
                     Rt: Dict[Union[int, float], List[int]]):
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
            result += dt * np.log((K * np.exp(z_at_risk)).sum())

        return result

    @staticmethod
    def component2(z, K, sigma, gamma, rho):
        element_wise = np.square(z) / 2 - sigma + (gamma / rho) * z
        return K * rho * element_wise.sum()

    @staticmethod
    def find_z(num_parties, gamma: ArrayLike, sigma: ArrayLike, rho: float,
               Rt: Dict[int, List[int]], z_start: ArrayLike, K: int, event_times: ArrayLike):
        logger.debug('Creating L(z) with parameters:')
        logger.debug(f'num_parties: {num_parties}  gamma: {gamma[:ARRAY_LOG_LIMIT]}..., sigma: '
                     f'{sigma[:ARRAY_LOG_LIMIT]}... '
                     f'rho:'
                     f' {rho} '
                     f'Rt: {list(Rt.keys())[:5]}...')
        L_z = lambda z: Lz.parametrized(z=z, K=num_parties, gamma=gamma, sigma=sigma, rho=rho,
                                        Rt=Rt)

        logger.debug(f'Finding minimum z starting at {z_start[:5]}...')

        jac = lambda z: Lz.jacobian(z, K, gamma, sigma, rho, Rt, event_times)

        minimum = minimize(L_z, z_start, jac=jac, method=OPTIMIZATION_METHOD)

        if not minimum.success:
            raise Exception('Could not find minimum z')

        logger.debug(f'Found minimum z at {minimum.x}')
        return minimum.x

    @staticmethod
    def derivative_1_parametrized(z: np.array, K: int, gamma: np.array, sigma: ArrayLike,
                                  rho: float,
                                  Rt: Dict[Union[int, float], List[int]], sample_idx: int,
                                  event_times: ArrayLike):
        """

        Args:
            z:
            K:
            gamma:
            sigma:
            rho:
            Rt:
            z_index:
            sample_idx:
            event_times:

        Returns:

        """
        dt = len(Rt)
        u_event_time = event_times[sample_idx]

        relevant_event_times = [t for t in Rt.keys() if t <= u_event_time]

        # First part
        # Numerator
        enumerator = K * np.exp(K * z[sample_idx])

        # Denominator
        denominator = 0
        for t in relevant_event_times:
            samples_at_risk = Rt[t]
            z_samples_at_risk = z[samples_at_risk]

            denominator += dt * (K * np.exp(K * z[sample_idx])) / np.sum(
                np.exp(K * z_samples_at_risk))

        first_part = enumerator / denominator

        # Second part
        second_part = K * rho * (z[sample_idx] - sigma[sample_idx] - (gamma[sample_idx] / rho))

        return first_part + second_part

    @staticmethod
    def jacobian(z: np.array, K: int, gamma: np.array, sigma: ArrayLike, rho: float,
                 Rt: Dict[Union[int, float], List[int]],
                 event_times: ArrayLike) -> ArrayLike:
        result = np.zeros(z.shape)

        for i in range(z.shape[0]):
            result[i] = Lz.derivative_1_parametrized(z, K=K, gamma=gamma, sigma=sigma, rho=rho,
                                                     Rt=Rt, sample_idx=i, event_times=event_times)

        return result
