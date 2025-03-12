import logging
import time
from typing import List, Dict

import numpy as np
from numba import types, typed
from numpy.typing import ArrayLike
from sksurv.functions import StepFunction
from vantage6.algorithm.tools.util import info

from verticox.common import group_samples_at_risk, group_samples_on_event_time
from verticox.grpc.datanode_pb2 import (
    Empty,
    AggregatedParameters,
    InitialValues,
    Subset,
    RecordLevelSigmaRequest,
    AverageSigmaRequest,
)
from verticox.grpc.datanode_pb2_grpc import DataNodeStub
from verticox.likelihood import find_z, find_z_fast

logger = logging.getLogger(__name__)

RHO = 0.25
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

    def __init__(
            self,
            institutions: List[DataNodeStub],
            event_times: np.array,
            event_happened: np.array,
            convergence_precision: float = DEFAULT_PRECISION,
            newton_raphson_precision: float = DEFAULT_PRECISION,
            rho=RHO,
            total_num_iterations=None
    ):
        """
        Initialize regular verticox aggregator. Note that this type of aggregator needs access to
        the event times of the samples.

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
        self.num_samples = event_times.shape[0]
        self.rho = rho  # TODO: Make dynamic
        self.event_times = event_times
        self.event_happened = event_happened
        self.Rt = group_samples_at_risk(event_times)
        self.Dt = group_samples_on_event_time(event_times, event_happened)
        # I only need unique event times, therefore I use Rt.keys()
        self.relevant_event_times = Aggregator._group_relevant_event_times(
            list(self.Rt.keys())
        )
        self.deaths_per_t = Aggregator.compute_deaths_per_t(event_times, event_happened)
        # Initializing parameters
        self.z = np.zeros(self.num_samples)
        self.z_old = self.z
        self.gamma = np.ones(self.num_samples)
        self.sigma = np.zeros(self.num_samples)

        self.num_iterations = 0
        self.total_num_iterations = total_num_iterations
        logger.debug(f"Institution stubs: {self.institutions}")

        beta = np.full(self.num_samples, BETA)
        z = np.full(self.num_samples, Z)
        gamma = np.full(self.num_samples, GAMMA)

        self.baseline_hazard_function_ = None
        self.betas_ = None
        self.baseline_survival_function_ = None
        self.prepare_datanodes(gamma, z, beta, self.rho)

    def prepare_datanodes(self, gamma, z, beta, rho):
        initial_values = InitialValues(gamma=gamma, z=z, beta=beta, rho=rho)

        for i in self.institutions:
            i.prepare(initial_values)

    @staticmethod
    def _group_relevant_event_times(
            unique_event_times,
    ) -> types.DictType(types.float64, types.float64[:]):
        # Make sure event times are unique
        unique_event_times = np.array(np.unique(unique_event_times))   
        result = typed.Dict.empty(types.float64, types.float64[:])

        for current_t in unique_event_times:
            result[current_t] = np.array(
                [t for t in unique_event_times if t <= current_t],
                dtype=np.float64
            )

        return result

    @staticmethod
    def compute_deaths_per_t(
            event_times: ArrayLike, event_happened: ArrayLike
    ) -> types.DictType(types.float64, types.int64):
        deaths_per_t = typed.Dict.empty(types.float64, types.int64)

        for t, event in zip(event_times, event_happened):
            if t not in deaths_per_t.keys():
                deaths_per_t[t] = 0

            if event:
                deaths_per_t[t] += 1

        return deaths_per_t

    def fit(self):
        start_time = time.time()
        current_time = start_time

        # Progress regarding the convergence criteria (sigma diff and z diff <= precision)
        # Since sigma and z diff will slowly shrink to match the order of magnitude of precision
        # I will track the log of the proportion of the precision variable compared to the
        # maximum of z_diff and sigma_diff then take the log. In the end result we should have
        # log(1) which is 0.
        progress = Progress(max_value=0)

        while True:
            logger.debug(
                "\n\n----------------------------------------\n"
                "          Starting new iteration..."
                "\n----------------------------------------"
            )
            logger.debug(f"Starting iteration num {self.num_iterations}")

            self.fit_one()

            # Turning the while in the paper into a do-until type thing. This means I have to
            # flip the > sign
            z_diff = np.linalg.norm(self.z - self.z_old)
            z_sigma_diff = np.linalg.norm(self.z - self.sigma)

            if self._stopping_condtions_met(z_diff, z_sigma_diff):
                break

            previous_time = current_time
            current_time = time.time()
            diff = current_time - previous_time
            total_runtime = current_time - start_time
            progress_value = np.log10(
                self.convergence_precision / max(z_diff, z_sigma_diff)
            )
            progress.update(progress_value)

            logger.debug(f"Completed current iteration after {diff} seconds")
            logger.debug(
                f"Iterations are taking on average {total_runtime / self.num_iterations} seconds "
                f"per run"
            )
            logger.info(f"Current progress: {100 * progress.get_value():.2f}%")

            info(f"Current progress: {100 * progress.get_value():.2f}%")

        logger.info(f"Finished training after {self.num_iterations} iterations")

        logger.debug("Retrieving betas...")
        self.betas_ = self.retrieve_betas()
        logger.info("Done")

        logger.debug("Computing baseline hazard...")
        self.baseline_hazard_function_ = self.compute_baseline_hazard_function(
            Subset.TRAIN
        )
        logger.debug(f"Baseline hazard_function: {self.baseline_hazard_function_}")
        logger.info("Done")

        logger.info("Computing baseline survival...")
        self.baseline_survival_function_ = self.compute_baseline_survival_function()
        logger.info("Done")

    def _stopping_condtions_met(self, z_diff, z_sigma_diff):
        if self.total_num_iterations is not None:
            return self.num_iterations > self.total_num_iterations
        else:
            return (z_diff <= self.convergence_precision and
                    z_sigma_diff <= self.convergence_precision)

    def fit_one(self):
        # TODO: Parallelize
        sigma_per_institution = np.zeros(
            (
                self.num_institutions,
                self.num_samples,
            )
        )
        gamma_per_institution = np.zeros(
            (
                self.num_institutions,
                self.num_samples,
            )
        )

        for idx, institution in enumerate(self.institutions):
            updated = institution.fit(Empty())
            sigma_per_institution[idx] = np.array(updated.sigma)
            gamma_per_institution[idx] = np.array(updated.gamma)

        self.sigma = self.aggregate_sigmas(sigma_per_institution)
        self.gamma = self.aggregate_gammas(gamma_per_institution)

        self.z_old = self.z

        start = time.time()
        self.z = find_z_fast(
            self.gamma,
            self.sigma,
            self.rho,
            self.Rt,
            self.z,
            self.num_institutions,
            self.event_times,
            self.Dt,
            self.deaths_per_t,
            self.relevant_event_times,
            self.newton_raphson_precision,
        )
        end = time.time()

        print(f"Time taken for find_z: {end - start} seconds")

        z_per_institution = self.compute_z_per_institution(
            gamma_per_institution, sigma_per_institution, self.z
        )

        # Update parameters at datanodes
        for idx, node in enumerate(self.institutions):
            params = AggregatedParameters(
                gamma=self.gamma.tolist(),
                sigma=self.sigma.tolist(),
                z=z_per_institution[idx].tolist(),
            )
            node.updateParameters(params)
            node.computeGamma(Empty())

        self.num_iterations += 1
        logger.debug(f"Num iterations: {self.num_iterations}")

    def compute_z_per_institution(
            self, gamma_per_institution, sigma_per_institution, z
    ):
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
                first_part = (
                        z[sample_idx] + sigma_per_institution[institution, sample_idx]
                )
                second_part = gamma_per_institution[institution, sample_idx] / self.rho

                # TODO: This only needs to be computed once per sample
                third_part = (
                        sigma_per_institution[:, sample_idx]
                        + gamma_per_institution[:, sample_idx] / self.rho
                )
                third_part = third_part.sum()
                third_part = third_part / self.num_institutions

                z_per_institution[institution, sample_idx] = (
                        first_part + second_part - third_part
                )

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

    def retrieve_betas(self) -> Dict[str, float]:
        betas = []
        names = []
        logger.debug(f"Getting betas from {len(self.institutions)} institutions")
        for institution in self.institutions:
            current_betas = institution.getBeta(Empty()).beta
            try:
                current_names = institution.getFeatureNames(Empty()).names
            except Exception:
                current_names = [None] * len(current_betas)
            betas += current_betas
            names += current_names

        return dict(zip(names, betas))

    def compute_baseline_hazard_function(self, subset: Subset) -> StepFunction:
        record_level_sigmas = self.compute_record_level_sigmas(subset)

        baseline_hazard = {}

        for t, group in self.Rt.items():
            summed_sigmas = record_level_sigmas[group]
            baseline_hazard[t] = 1 / np.exp(summed_sigmas).sum()

        baseline_x, baseline_y = zip(*sorted(baseline_hazard.items()))
        return StepFunction(np.array(baseline_x), np.array(baseline_y))

    def compute_record_level_sigmas(self, subset: Subset):
        """
        Compute the risk score per record in the specified subset
        Args:
            subset:

        Returns:

        """
        record_level_sigmas = None
        request = RecordLevelSigmaRequest(subset=subset)
        for idx, institution in enumerate(self.institutions):
            record_level_sigma_institution = institution.getRecordLevelSigma(request)

            if record_level_sigmas is None:
                record_level_sigmas = np.array(record_level_sigma_institution.sigma)
            else:
                record_level_sigmas = record_level_sigmas + np.array(
                    record_level_sigma_institution.sigma)

        return record_level_sigmas

    def sum_average_sigmas(self, subset: Subset):
        summed = 0
        for datanode in self.institutions:
            result = datanode.getAverageSigma(AverageSigmaRequest(subset=subset))

            summed += result.sigma

        return summed

    def predict_average_cumulative_survival(self, subset: Subset):
        average_sigmas = self.sum_average_sigmas(subset)

        return self.compute_cumulative_survival(
            self.baseline_survival_function_, average_sigmas
        )

    @staticmethod
    def compute_cumulative_survival(
            baseline_survival_function: StepFunction, subpopulation_sigmas
    ):
        cum_survival = np.power(
            baseline_survival_function.y, np.exp(subpopulation_sigmas)
        )
        return StepFunction(x=baseline_survival_function.x, y=cum_survival)

    def compute_baseline_survival_function(
            self,
    ):
        cum_hazard = Aggregator.compute_cumulative_hazard_function(
            self.deaths_per_t, self.baseline_hazard_function_
        )

        cum_survival = np.exp(cum_hazard.y * -1)

        cumulative_survival = StepFunction(
            self.baseline_hazard_function_.x, cum_survival
        )

        return cumulative_survival

    @staticmethod
    def compute_cumulative_hazard_function(
            deaths_per_t: Dict[float, int], baseline_hazard: StepFunction
    ) -> StepFunction:
        result = np.zeros(baseline_hazard.x.shape[0])

        for idx, t in enumerate(baseline_hazard.x):
            summed = 0
            for previous_idx in range(idx + 1):
                previous_t = baseline_hazard.x[previous_idx]
                summed += deaths_per_t.get(previous_t, 0) * baseline_hazard.y[previous_idx]

            result[idx] = summed
        return StepFunction(x=baseline_hazard.x, y=result)

    def compute_auc(self):
        """
        Computes area under curve on the test data
        Returns:

        """
        record_level_sigmas_test = self.compute_record_level_sigmas(Subset.TEST)

        # Compute cumulative hazard per record
        record_level_cum_survival = []
        for sigma in record_level_sigmas_test:
            cum_survival = self.compute_cumulative_survival(
                self.baseline_survival_function_, np.array([sigma])
            )
            record_level_cum_survival.append(cum_survival.y)

        auc = []
        record_level_cum_survival = np.array(record_level_cum_survival)

        num_test_records = record_level_sigmas_test.shape[0]

        # Lambda1 is the sum of all cumulative survival values
        lambda1 = record_level_cum_survival.sum(axis=0) / num_test_records

        # For every time t
        for t_index in range(self.baseline_survival_function_.x.shape[0]):

            lambda2 = 0

            # For every record in TEST
            for idx, single_cum_survival in enumerate(record_level_cum_survival):

                # For every other record
                for idx2, record_cum_survival2 in enumerate(record_level_cum_survival):
                    if record_level_sigmas_test[idx] > record_level_sigmas_test[idx2]:
                        lambda2 += (
                                           1 - single_cum_survival[t_index]
                                   ) * record_cum_survival2[t_index]

            lambda2 = lambda2 / np.square(num_test_records)

            auc.append(lambda2 / ((1 - lambda1[t_index]) * lambda1[t_index]))

        auc = np.array(auc)

        return StepFunction(x=self.baseline_survival_function_.x, y=auc)


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
        return (self.current_value - self.initial_value) / (
                self.max_value - self.initial_value
        )
