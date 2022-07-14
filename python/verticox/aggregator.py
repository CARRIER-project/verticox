import logging
from collections import namedtuple
from typing import List, Dict, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from verticox.grpc.datanode_pb2 import Empty, AggregatedParameters, InitialValues
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

logger = logging.getLogger(__name__)

RHO = 0.25
E = .1
BETA = 0
Z = 0
GAMMA = 0

OPTIMIZATION_METHOD = 'Newton-CG'
OPTIMIZATION_OPTIONS = {'xtol': 0.1, 'eps':0.01}
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
        self.z = np.zeros(self.num_samples, dtype=np.float128)
        self.z_old = self.z
        self.gamma = np.ones(self.num_samples, dtype=np.float128)
        self.sigma = np.zeros(self.num_samples, dtype=np.float128)
        self.gamma_per_institution = np.ones((self.num_institutions, self.num_samples,),
                                             dtype=np.float128)
        self.num_iterations = 0

        self.prepare_datanodes(GAMMA, Z, BETA, RHO)

    def prepare_datanodes(self, gamma, z, beta, rho):
        initial_values = InitialValues(gamma=gamma, z=z, beta=beta, rho=rho)

        logger.debug(f'Preparing datanodes with: {initial_values}')

        for i in self.institutions:
            i.prepare(initial_values)

    def fit(self):

        while True:
            logger.info('\n\n----------------------------------------\n'
                        '          Starting new iteration...'
                        '\n----------------------------------------')
            self.fit_one()

            # Turning the while in the paper into a do-until type thing. This means I have to
            # flip the > sign
            z_diff = np.linalg.norm(self.z - self.z_old)
            z_sigma_diff = np.linalg.norm(self.z - self.sigma)

            logger.debug(f'z_diff: {z_diff}')
            logger.debug(f'sigma_diff: {z_sigma_diff}')

            if z_diff<= self.e and z_sigma_diff <= self.e:
                break

        logger.info(f'Finished training after {self.num_iterations} iterations')

    def fit_one(self):
        # TODO: Parallelize
        sigma_per_institution = np.zeros((self.num_institutions, self.num_samples,),
                                         dtype=np.float128)

        for idx, institution in enumerate(self.institutions):
            updated = institution.fit(Empty())
            sigma_per_institution[idx] = np.array(updated.sigma, dtype=np.float128)
            self.gamma_per_institution[idx] = np.array(updated.gamma, dtype=np.float128)

        self.sigma = self.aggregate_sigmas(sigma_per_institution)
        self.gamma = self.aggregate_gammas(self.gamma_per_institution)

        self.z_old = self.z
        self.z = Lz.find_z(self.gamma, self.sigma, self.rho, self.Rt, self.z,
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
        z_per_institution = np.zeros((self.num_institutions, self.num_samples), dtype=np.float128)
        sigma_gamma_all_institutions = sigma_per_institution + gamma_per_institution / self.rho
        sigma_gamma_all_institutions = sigma_gamma_all_institutions.sum(axis=0)

        logger.debug(f'Sigma per institution: \n{sigma_per_institution}')
        logger.debug(f'Gamma per institution: \n{gamma_per_institution}')
        # TODO: vectorize
        for i in range(self.num_institutions):
            z_per_institution[i] = z + sigma_per_institution[i] + gamma_per_institution[i] / \
                                   self.rho \
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

        return np.array(betas, dtype=np.float128)


class Lz:
    # TODO: Vectorizing might make things easier
    # TODO: Move to its own module

    Parameters = namedtuple('Parameters', ['gamma', 'sigma', 'rho', 'Rt', 'K', 'event_times'])

    @staticmethod
    def parametrized(z: ArrayLike, params: Parameters):
        """
        Equation 12
        Args:
            z:
            params:

        Returns:

        """
        dt = len(params.Rt)

        component1 = Lz.component1(z, params.K, params.Rt, dt)
        component2 = Lz.component2(z, params.K, params.sigma, params.gamma, params.rho)

        result = component1 + component2
        logger.debug(f'Lz: {result}')
        return result

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
    def find_z(gamma: ArrayLike, sigma: ArrayLike, rho: float,
               Rt: Dict[int, List[int]], z_start: ArrayLike, K: int, event_times: ArrayLike):

        params = Lz.Parameters(gamma, sigma, rho, Rt, K, event_times)

        logger.debug('Creating L(z) with parameters:')
        logger.debug(params)
        L_z = lambda z: Lz.parametrized(z=z, params=params)

        logger.debug(f'Finding minimum z starting at {z_start[:5]}...')

        jac = lambda z: Lz.jacobian(z, params)
        hessian = lambda z: Lz.hessian(z, params)

        minimum = minimize(L_z, z_start, jac=jac, hess=hessian, method=OPTIMIZATION_METHOD,
                           options=OPTIMIZATION_OPTIONS)

        if not minimum.success:
            raise Exception('Could not find minimum z')

        logger.debug(f'Found minimum z at {minimum.x}')
        return minimum.x

    @staticmethod
    def derivative_1(z: np.array, params: Parameters, sample_idx: int):
        """

        Args:
            z:
            params:
            sample_idx:


        Returns:

        """
        dt = len(params.Rt)
        u_event_time = params.event_times[sample_idx]

        relevant_event_times = [t for t in params.Rt.keys() if t <= u_event_time]

        # First part
        enumerator = params.K * np.exp(params.K * z[sample_idx])

        first_part = 0
        for t in relevant_event_times:
            denominator = Lz.bottom(z, params, t)

            first_part += dt * (enumerator / denominator)

        # Second part
        second_part = params.K * params.rho * (z[sample_idx] - params.sigma[sample_idx] - (
                params.gamma[sample_idx] / params.rho))

        return first_part + second_part

    @staticmethod
    def bottom(z, params, t):
        denominator = 0.
        for j in params.Rt[t]:
            denominator += np.exp(params.K * z[j])
        return denominator

    @staticmethod
    def jacobian(z: ArrayLike, params: Parameters) -> ArrayLike:
        result = np.zeros(z.shape)

        for i in range(z.shape[0]):
            result[i] = Lz.derivative_1(z, params, sample_idx=i)

        return result

    @staticmethod
    def hessian(z: ArrayLike, params: Parameters):
        # The hessian is a N x N matrix where N is the number of elements in z
        N = z.shape[0]
        mat = np.zeros((N, N))

        for u in range(N):
            for v in range(N):

                if u == v:
                    # Formula for diagonals
                    mat[u, v] = Lz.derivative_2_diagonal(z, params, u)
                else:
                    # Formula for off-diagonals
                    mat[u, v] = Lz.derivative_2_off_diagonal(z, params, u, v)

        return mat

    @staticmethod
    def derivative_2_diagonal(z: ArrayLike, params: Parameters, u):
        dt = len(params.Rt)
        u_event_time = params.event_times[u]

        relevant_event_times = [t for t in params.Rt.keys() if t <= u_event_time]

        summed = 0

        for t in relevant_event_times:
            denominator = Lz.bottom(z, params, t)

            first_part = np.square(params.K) * np.exp(params.K * z[u]) / denominator

            second_part = np.square(params.K) * np.square(np.exp(params.K * z[u])) / np.square(
                denominator)

            summed += dt * (first_part - second_part)

        return summed + params.K * params.rho

    @staticmethod
    def derivative_2_off_diagonal(z: ArrayLike, params: Parameters, u, v):
        dt = len(params.Rt)

        min_event_time = min(params.event_times[u], params.event_times[u])
        relevant_event_times = [t for t in params.Rt.keys() if t <= min_event_time]

        summed = 0

        for t in relevant_event_times:
            summed += dt * np.square(params.K) * np.exp(params.K * z[u]) * np.exp(params.K * z[v]) / \
                      np.square(np.exp(params.K * z[params.Rt[t]]).sum())

        return -1 * summed
