import numba
import numpy as np
from numba import types
from numpy.typing import ArrayLike

from verticox import NPartyScalarProductClient

constant_param_spec = [
    ('gamma', types.float64[:]),
    ('sigma', types.float64[:]),
    ('Dt', types.float64[:]),
    ('distinct_event_times', types.types.float64[:]),
    ('deaths_per_t', types.int64[:]),
]


@numba.experimental.jitclass(constant_param_spec)
class ConstantParameters:
    gamma: np.ndarray
    sigma: np.ndarray
    rho: float
    K: int
    Dt: np.ndarray
    distinct_event_times: np.ndarray
    deaths_per_t: np.ndarray

    def __init__(self, gamma, sigma, rho, K, distinct_event_times, deaths_per_t):
        self.gamma = gamma
        self.sigma = sigma
        self.rho = rho
        self.K = K
        self.deaths_per_t = deaths_per_t
        self.distinct_event_times = distinct_event_times


z_spec = [
    ('z', types.float64[:]),
    ('exp_k_z', types.float64[:]),
    ('aggregated_hazard_per_t', types.float64[:])
]


@numba.experimental.jitclass(z_spec)
class ZBasedComponents:
    z: np.ndarray
    exp_k_z: np.ndarray
    aggregated_hazard_per_t: np.ndarray

    def __init__(self, z, exp_k_z, aggregated_hazard_per_t):
        self.z = z
        self.exp_k_z = exp_k_z
        self.aggregated_hazard_per_t = aggregated_hazard_per_t


class NPartyMaxLikelihoodFinder:

    def __init__(self, commodity_address: str, gamma: ArrayLike, sigma: ArrayLike, rho: float,
                 K: int, deaths_per_t: ArrayLike, distinct_event_times: ArrayLike,
                 event_time_column: str):
        self._commodity_address = commodity_address
        self.n_party_client = NPartyScalarProductClient(commodity_address)
        self.gamma = gamma
        self.sigma = sigma
        self.rho = rho
        self.K = K
        self.deaths_per_t = deaths_per_t
        self.distinct_event_times = distinct_event_times
        self.event_time_column = event_time_column

    def get_z_based_components(self, z) -> ZBasedComponents:
        """
        Compute all necessary components that are based on the vector z.
        These parts should be computed only once per Newton-Rhapson iteration to prevent
        redundant calls to the n-party-scalar product.

        TODO: List z-based components
        Args:
            z:

        Returns:

        """
        exp_k_z = np.exp(self.K * z)

        aggregated_hazard = np.zeros_like(self.distinct_event_times)

        for idx, t in enumerate(self.distinct_event_times):
            aggregated_hazard[idx] = self.n_party_client. \
                sum_records_at_risk(exp_k_z.tolist(), t, self.event_time_column)

        return ZBasedComponents(z, exp_k_z, aggregated_hazard)

    # def find_z(self, z_start):
    #     minimum = minimize_newton_raphson(z_start,
    #                                       jacobian_parametrized, hessian_parametrized,
    #                                       params=params, eps=eps)
    #
    #     return minimum


@numba.njit()
def jacobian_parametrized(z_components: ZBasedComponents,
                          params: ConstantParameters) -> types.float64[:]:
    z = z_components.z

    result = np.zeros(z.shape)

    for i in range(z.shape[0]):
        result[i] = derivative_1(z_components, params, sample_idx=i)

    return result


@numba.njit()
def derivative_1(z_components, params: ConstantParameters, sample_idx: int):
    """

    Args:
        z_components:
        z:
        params:
        sample_idx:


    Returns:

    """


    # First part
    enumerator = params. params.K * np.exp(params.K * z_components.z[sample_idx])

    first_part = 0

    # TODO: I need to limit the following for loop to not go over the event time of the current
    #  sample, without actually getting the event time of the sample

    for aggregated_hazard, deaths in zip(z_components.aggregated_hazard_per_t, params.deaths_per_t):
        first_part += deaths * (enumerator / aggregated_hazard)

    # Second part
    second_part = params.K * params.rho * (z_components.z[sample_idx] - params.sigma[sample_idx] -
                                           (params.gamma[sample_idx] / params.rho))

    return first_part + second_part
