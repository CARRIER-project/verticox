import logging
import time
from typing import List
from numba import types, typed
import numpy as np
from numpy.typing import ArrayLike
from vantage6.common import info

from verticox.likelihood import find_z
from verticox.common import group_samples_at_risk, group_samples_on_event_time
from verticox.grpc.datanode_pb2 import Empty, AggregatedParameters, InitialValues
from verticox.grpc.datanode_pb2_grpc import DataNodeStub

logger = logging.getLogger(__name__)

RHO = 0.5
DEFAULT_PRECISION = 1e-5
BETA = 0
Z = 0
GAMMA = 0

# OPTIMIZATION_METHOD = 'Newton-CG'
# OPTIMIZATION_OPTIONS = {'maxiter': 10}
ARRAY_LOG_LIMIT = 5


class Aggregator:
    """
    Central node that aggregates the result of the datanodes and coordinates calculations

    TODO: This implementation is matching the paper quite closely. Unfortunately it relies a lot on
            state. This will have to be fixed later.
    """

    def __init__(self, institutions: List[DataNodeStub], event_times: types.float64[:],
                 event_happened: types.boolean[:], convergence_precision: float = DEFAULT_PRECISION,
                 newton_raphson_precision: float = DEFAULT_PRECISION, rho=RHO):
        """
        Initialize regular verticox aggregator. Note that this type of aggregator needs access to the
        event times of the samples.

        Args:
            institutions:
            event_times:
            event_happened:
            convergence_precision: threshold value
        """
        self.convergence_precision = convergence_precision
        self.newton_raphson_precision = newton_raphson_precision
        self.institutions = tuple(institutions)
        self.num_institutions = len(institutions)
        self.features_per_institution = self.get_features_per_institution()
        self.num_samples = self.get_num_samples()
        self.rho = rho  # TODO: Make dynamic
        self.event_times = event_times
        self.event_happened = event_happened
        self.Rt = group_samples_at_risk(event_times)
        self.Dt = group_samples_on_event_time(event_times, event_happened)
        self.deaths_per_t = Aggregator._compute_deaths_per_t(event_times, event_happened)
        # Initializing parameters
        self.z = np.zeros(self.num_samples)
        self.z_old = self.z
        self.gamma = np.ones(self.num_samples)
        self.sigma = np.zeros(self.num_samples)

        self.num_iterations = 0

        self.prepare_datanodes(GAMMA, Z, BETA, self.rho)

    @staticmethod
    def _compute_deaths_per_t(event_times: ArrayLike, event_happened: ArrayLike) -> \
            types.DictType(types.float64, types.int64):

        deaths_per_t = typed.Dict.empty(types.float64, types.int64)

        for t, event in zip(event_times, event_happened):
            if t not in deaths_per_t.keys():
                deaths_per_t[t] = 0

            if event:
                deaths_per_t[t] += 1

        return deaths_per_t

    def prepare_datanodes(self, gamma, z, beta, rho):
        initial_values = InitialValues(gamma=gamma, z=z, beta=beta, rho=rho)

        logger.debug(f'Preparing datanodes with: {initial_values}')

        for i in self.institutions:
            i.prepare(initial_values)

    def fit(self):
        start_time = time.time()
        current_time = start_time

        # Progress regarding the convergence criteria (sigma diff and z diff <= precision)
        # Since sigma and z diff will slowly shrink to match the order of magnitude of precision
        # I will track the log of the proportion of the precision variable compared to the
        # maximum of z_diff en sigma_diff and take the log. In the end result we should have
        # log(1) which is 0.
        progress = Progress(max_value=0)

        while True:
            logger.info('\n\n----------------------------------------\n'
                        '          Starting new iteration...'
                        '\n----------------------------------------')
            info(f'Starting iteration num {self.num_iterations}')

            self.fit_one()

            # Turning the while in the paper into a do-until type thing. This means I have to
            # flip the > sign
            z_diff = np.linalg.norm(self.z - self.z_old)
            z_sigma_diff = np.linalg.norm(self.z - self.sigma)

            logger.debug(f'z_diff: {z_diff}')
            logger.debug(f'sigma_diff: {z_sigma_diff}')
            info(f'z_diff: {z_diff}')
            info(f'sigma_diff: {z_sigma_diff}')

            if z_diff <= self.convergence_precision and z_sigma_diff <= self.convergence_precision:
                break

            previous_time = current_time
            current_time = time.time()
            diff = current_time - previous_time
            total_runtime = current_time - start_time
            progress_value = np.log10(self.convergence_precision / max(z_diff, z_sigma_diff))
            progress.update(progress_value)

            info(f'Completed current iteration after {diff} seconds')
            info(f'Iterations are taking on average {total_runtime / self.num_iterations} seconds '
                 f'per run')
            info(f'Current progress: {100 * progress.get_value():.2f}%')

        logger.info(f'Finished training after {self.num_iterations} iterations')

    def kill_all_datanodes(self):
        for institution in self.institutions:
            institution.kill(Empty())

    def fit_one(self):
        # TODO: Parallelize
        sigma_per_institution = np.zeros((self.num_institutions, self.num_samples,))
        gamma_per_institution = np.zeros((self.num_institutions, self.num_samples,))

        for idx, institution in enumerate(self.institutions):
            updated = institution.fit(Empty())
            sigma_per_institution[idx] = np.array(updated.sigma)
            gamma_per_institution[idx] = np.array(updated.gamma)

        self.sigma = self.aggregate_sigmas(sigma_per_institution)
        self.gamma = self.aggregate_gammas(gamma_per_institution)

        self.z_old = self.z
        self.z = find_z(self.gamma, self.sigma, self.rho, self.Rt, self.z,
                        self.num_institutions, self.event_times, self.Dt,
                        self.deaths_per_t, self.newton_raphson_precision)

        z_per_institution = self.compute_z_per_institution(gamma_per_institution,
                                                           sigma_per_institution, self.z)

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

        for institution in range(self.num_institutions):
            for sample_idx in range(self.num_samples):
                first_part = z[sample_idx] + sigma_per_institution[institution, sample_idx]
                second_part = gamma_per_institution[institution, sample_idx] / self.rho

                # TODO: This only needs to be computed once per sample
                third_part = sigma_per_institution[:, sample_idx] + \
                             gamma_per_institution[:, sample_idx] / self.rho
                third_part = third_part.sum()
                third_part = third_part / self.num_institutions

                z_per_institution[institution, sample_idx] = first_part + second_part - third_part

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
        names = []
        for institution in self.institutions:
            current_betas = institution.getBeta(Empty()).beta
            try:
                current_names = institution.getFeatureNames(Empty).names
            except Exception as e:
                print(e)
                current_names = [None] * len(current_betas)
            betas += current_betas
            names += current_names

        return list(zip(names, betas))


class Progress:
    """
    Helper class to keep track of progress
    """

    def __init__(self, max_value: object):
        self.initial_value = None
        self.current_value = None
        self.max_value = max_value

    def update(self, value):
        if self.initial_value is None:
            self.initial_value = value

        self.current_value = value

    def get_value(self):
        return (self.current_value - self.initial_value) / (self.max_value - self.initial_value)
