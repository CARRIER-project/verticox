import logging
from typing import Dict

import numba
import numpy as np
from numba import types, prange
from numpy.typing import ArrayLike
import scalarproduct

EPSILON = 1e-4

_logger = logging.getLogger(__name__)

spec = [
    ('gamma', types.float64[:]),
    ('sigma', types.float64[:]),
    ('Dt', types.DictType(types.float64, types.int64[:])),
    ('deaths_per_t', types.DictType(types.float64, types.int64)),
    ('relevant_event_times', types.DictType(types.float64, types.float64[:]))
]


@numba.experimental.jitclass(spec)
class Parameters:
    gamma: np.ndarray
    sigma: np.ndarray
    rho: float
    K: int
    Dt: Dict[float, np.ndarray]
    deaths_per_t: Dict[float, int]
    relevant_event_times: Dict[float, np.ndarray]

    def __init__(self, gamma, sigma, rho, Rt, K, event_times, Dt, deaths_per_t,
                 relevant_event_times):
        self.gamma = gamma
        self.sigma = sigma
        self.rho = rho
        self.K = K
        self.Dt = Dt
        self.deaths_per_t = deaths_per_t


def get_deaths_per_t():
    # TODO: implement
    pass


@numba.njit
def auxiliary_variables_component(z, K, sigma, gamma, rho):
    element_wise = np.square(z) / 2 - sigma + (gamma / rho) * z
    return K * rho * element_wise.sum()


# TODO: Implement privacy preserving version of relevant event times
def find_z(gamma: ArrayLike, sigma: ArrayLike, rho: float,
           Rt: types.DictType(types.float64, types.int64[:]), z_start: types.float64[:], K: int,
           event_times: types.float64[:], Dt: types.DictType(types.float64, types.int64[:]),
           relevant_event_times: types.DictType(types.float64, types.float64[:]),
           eps: float = EPSILON):
    deaths_per_t = get_deaths_per_t()
    # TODO: Refactor. There are some parameters that I don't actually need
    params = Parameters(gamma, sigma, rho, Rt, K, event_times, Dt, deaths_per_t)

    minimum = minimize_newton_raphson(z_start,
                                      jacobian_parametrized, hessian_parametrized,
                                      params=params, eps=eps)

    return minimum


@numba.njit()
def derivative_1(exp_K_z, params: Parameters, sample_idx: int):
    """

    Args:
        z:
        params:
        sample_idx:


    Returns:

    """
    u_event_time = params.event_times[sample_idx]

    relevant_event_times = params.relevant_event_times[u_event_time]

    # First part
    enumerator = params.K * np.exp(params.K * z[sample_idx])

    first_part = 0
    for t in relevant_event_times:
        denominator = aggregated_hazard_at_t(exp_K_z, params, t)

        first_part += params.deaths_per_t[t] * (enumerator / denominator)

    # Second part
    second_part = params.K * params.rho * (z[sample_idx] - params.sigma[sample_idx] - (
            params.gamma[sample_idx] / params.rho))

    return first_part + second_part


@numba.njit()
def jacobian_parametrized(z: types.float64[:], exp_K_z, params: Parameters) -> types.float64[:]:
    result = np.zeros(z.shape)

    for i in range(z.shape[0]):
        result[i] = derivative_1(z, exp_K_z, params, sample_idx=i)

    return result


@numba.njit()
def derivative_2_diagonal(z: types.float64, params: Parameters, u: int) -> types.float64[:, :]:
    u_event_time = params.event_times[u]

    relevant_event_times = params.relevant_event_times[u_event_time]

    summed = 0

    for t in relevant_event_times:
        denominator = aggregated_hazard_at_t(z, params, t)

        first_part = np.square(params.K) * np.exp(params.K * z[u]) / denominator

        second_part = np.square(params.K) * np.square(np.exp(params.K * z[u])) / \
                      np.square(denominator)

        summed += params.deaths_per_t[t] * (first_part - second_part)

    return summed + params.K * params.rho


@numba.njit()
def derivative_2_off_diagonal(z: ArrayLike, params, u, v):
    min_event_time = min(params.event_times[u], params.event_times[v])
    relevant_event_times = params.relevant_event_times[min_event_time]

    elements = np.zeros(relevant_event_times.shape[0])
    K_squared = params.K * params.K

    for i in range(relevant_event_times.shape[0]):
        t = relevant_event_times[i]
        elements[i] = params.deaths_per_t[t] * K_squared * np.exp(params.K * z[u]) * \
                      np.exp(params.K * z[v]) / \
                      np.square(np.exp(params.K * z[params.Rt[t]]).sum())

    return -1 * elements.sum()


@numba.njit(parallel=True)
def hessian_parametrized(z: types.float64[:], params: Parameters):
    # The hessian is a N x N matrix where N is the number of elements in z
    N = z.shape[0]
    mat = np.zeros((N, N))

    for u in range(N):
        mat[u, u] = derivative_2_diagonal(z, params, u)

    for u in prange(N):
        for v in range(u + 1, N):
            # Formula for off-diagonals
            mat[u, v] = derivative_2_off_diagonal(z, params, u, v)
            mat[v, u] = mat[u, v]

    return mat


def compute_sum_exp_k_z_for_all_t(z: ArrayLike, K: int, unique_t, t_column, commodity_address):
    n_party_client = scalarproduct.NPartyScalarProductClient(commodity_address)

    exp_K_z = np.exp(K * z)

    result = np.zeros_like(unique_t)
    for idx, t in enumerate(unique_t):
        result[idx] = n_party_client.sum_records_at_risk(exp_K_z, t, t_column)

    return result


@numba.njit()
def minimize_newton_raphson(x_0: types.float64[:], jacobian, hessian, params: Parameters,
                            eps: float) -> \
        types.float64[:]:
    """
    The terminology is a little confusing here. We are trying to find the minimum,
    but newton-raphson is a root-finding algorithm. Therefore we are looking for the x where the
    norm of the first-order derivative (jacobian) is 0. In the context of newton-rhapson this
    would be function F(x), while our hessian matrix would be F'(x).
    Args:
        x_0:
        jacobian:
        hessian:
        params:
        eps:

    Returns:

    """
    x = x_0
    compute_sum_exp_k_z_for_all_t(x, params.K, )

    current_jac = jacobian(x, params)
    while np.linalg.norm(current_jac) > eps:
        current_hess = hessian(x, params)
        current_jac = jacobian(x, params)

        x = x - np.linalg.inv(current_hess) @ current_jac
    return x
