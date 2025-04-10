import logging
from typing import Dict

import numba
import numpy as np
from numba import prange, types
from numpy.typing import ArrayLike

EPSILON = 1e-4

_logger = logging.getLogger(__name__)

spec = [
    ("gamma", types.float64[:]),
    ("sigma", types.float64[:]),
    ("Rt", types.DictType(types.float64, types.int64[:])),
    ("event_times", types.float64[:]),
    ("Dt", types.DictType(types.float64, types.int64[:])),
    ("deaths_per_t", types.DictType(types.float64, types.int64)),
    ("relevant_event_times", types.DictType(types.float64, types.float64[:])),
]


@numba.experimental.jitclass(spec)
class Parameters:
    gamma: np.ndarray
    sigma: np.ndarray
    rho: float
    Rt: Dict[float, np.ndarray]
    K: int
    event_times: np.ndarray
    Dt: Dict[float, np.ndarray]
    deaths_per_t: Dict[float, int]
    relevant_event_times: Dict[float, np.ndarray]

    def __init__(
        self,
        gamma,
        sigma,
        rho,
        Rt,
        K,
        event_times,
        Dt,
        deaths_per_t,
        relevant_event_times,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.rho = rho
        self.Rt = Rt
        self.K = K
        self.event_times = event_times
        self.Dt = Dt
        self.deaths_per_t = deaths_per_t
        self.relevant_event_times = relevant_event_times


@numba.njit
def parametrized(z: types.float64[:], params: Parameters) -> types.float64:
    """
    Equation 12
    Args:
        z:
        params:

    Returns:

    """
    c1 = aggregated_hazard(z, params)
    c2 = auxiliary_variables_component(
        z, params.K, params.sigma, params.gamma, params.rho
    )

    result = c1 + c2
    return result


@numba.njit
def aggregated_hazard(z, params: Parameters):
    result = 0
    for t, group in params.Rt.items():
        z_at_risk = z[group]
        result += params.deaths_per_t[t] * np.log((np.exp(params.K * z_at_risk)).sum())

    return result


@numba.njit
def auxiliary_variables_component(z, K, sigma, gamma, rho):
    element_wise = np.square(z) / 2 - sigma + (gamma / rho) * z
    return K * rho * element_wise.sum()


def find_z_fast(
    gamma: ArrayLike,
    sigma: ArrayLike,
    rho: float,
    Rt: types.DictType(types.float64, types.int64[:]),
    z_start: types.float64[:],
    K: int,
    event_times: types.float64[:],
    Dt: types.DictType(types.float64, types.int64[:]),
    deaths_per_t: types.DictType(types.float64, types.int64),
    relevant_event_times: types.DictType(types.float64, types.float64[:]),
    eps: float = EPSILON,
):
    params = Parameters(
        gamma, sigma, rho, Rt, K, event_times, Dt, deaths_per_t, relevant_event_times
    )

    minimum = minimize_newton_raphson_fast(
        z_start, params=params, eps=eps
    )

    return minimum


def find_z(
    gamma: ArrayLike,
    sigma: ArrayLike,
    rho: float,
    Rt: types.DictType(types.float64, types.int64[:]),
    z_start: types.float64[:],
    K: int,
    event_times: types.float64[:],
    Dt: types.DictType(types.float64, types.int64[:]),
    deaths_per_t: types.DictType(types.float64, types.int64),
    relevant_event_times: types.DictType(types.float64, types.float64[:]),
    eps: float = EPSILON,
):
    params = Parameters(
        gamma, sigma, rho, Rt, K, event_times, Dt, deaths_per_t, relevant_event_times
    )

    minimum = minimize_newton_raphson(
        z_start, jacobian_parametrized, hessian_parametrized, params=params, eps=eps
    )

    return minimum

@numba.njit()
def derivative_1(z, params: Parameters, sample_idx: int):
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
        denominator = aggregated_hazard_at_t(z, params, t)

        first_part += params.deaths_per_t[t] * (enumerator / denominator)

    # Second part
    second_part = (
        params.K
        * params.rho
        * (
            z[sample_idx]
            - params.sigma[sample_idx]
            - (params.gamma[sample_idx] / params.rho)
        )
    )

    return first_part + second_part


@numba.njit()
def aggregated_hazard_at_t(z, params, t):
    denominator = 0.0
    for j in params.Rt[t]:
        denominator += np.exp(params.K * z[j])
    return denominator


@numba.njit(parallel=True)
def jacobian_parametrized(z: types.float64[:], params: Parameters) -> types.float64[:]:
    result = np.zeros(z.shape)

    for i in prange(z.shape[0]):
        result[i] = derivative_1(z, params, sample_idx=i)

    return result


@numba.njit()
def derivative_2_diagonal(
    z: types.float64, params: Parameters, u: int
) -> types.float64[:, :]:
    u_event_time = params.event_times[u]

    relevant_event_times = params.relevant_event_times[u_event_time]

    summed = 0

    for t in relevant_event_times:
        denominator = aggregated_hazard_at_t(z, params, t)

        first_part = np.square(params.K) * np.exp(params.K * z[u]) / denominator

        second_part = (
            np.square(params.K)
            * np.square(np.exp(params.K * z[u]))
            / np.square(denominator)
        )

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
        elements[i] = (
            params.deaths_per_t[t]
            * K_squared
            * np.exp(params.K * z[u]) # exp_k_z of u
            * np.exp(params.K * z[v]) # exp_k_z of v
            / np.square(np.exp(params.K * z[params.Rt[t]]).sum()) # derivative2_denominator
        )

    return -1 * elements.sum()

@numba.njit()
def derivative_2_off_diagonal_fast(z: ArrayLike, params, u, v, precomp_derivative_denominator, precomp_exp_k_z):
    min_event_time = min(params.event_times[u], params.event_times[v])
    relevant_event_times = params.relevant_event_times[min_event_time]

    elements = np.zeros(relevant_event_times.shape[0])
    K_squared = params.K * params.K

    for i in range(relevant_event_times.shape[0]):
        t = relevant_event_times[i]
        elements[i] = (
            params.deaths_per_t[t]
            * K_squared
            * precomp_exp_k_z[u]
            * precomp_exp_k_z[v]
            * precomp_derivative_denominator[t]
        )

    return -1 * elements.sum()

@numba.njit()
def derivative2_denominator(z, params):
    event_times = params.Rt.keys()

    result = {}

    for t in event_times:
        result[t] = 1/np.square(np.exp(params.K * z[params.Rt[t]]).sum())

    return result

@numba.njit()
def exp_k_z(z, k):
    return np.exp(k * z)


@numba.njit(parallel=True)
def hessian_parametrized_fast(z: types.float64[:], params: Parameters):
    # The hessian is a N x N matrix where N is the number of elements in z
    N = z.shape[0]
    mat = np.zeros((N, N))
    
    precomputed_derivative_denominator = derivative2_denominator(z, params)
    precomputed_exp_k_z = exp_k_z(z, params.K)

    for u in range(N):
        mat[u, u] = derivative_2_diagonal(z, params, u)

    for u in prange(N):
        for v in range(u + 1, N):
            # Formula for off-diagonals
            mat[u, v] = derivative_2_off_diagonal_fast(z, params, u, v, precomputed_derivative_denominator, precomputed_exp_k_z)
            mat[v, u] = mat[u, v]

    return mat

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


@numba.njit()
def minimize_newton_raphson(
    x_0: types.float64[:], jacobian, hessian, params: Parameters, eps: float
) -> types.float64[:]:
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
    current_jac = jacobian(x, params)
    while np.linalg.norm(current_jac) > eps:
        # before = time.time()
        current_hess = hessian(x, params)
        # after = time.time()

        #print(f"Time taken for hessian: {after - before}")

        # before = time.time()
        current_jac = jacobian(x, params)
        # after = time.time()

        #print(f"Time taken for jacobian: {after - before}")

        # before = time.time()
        x = x - np.linalg.inv(current_hess) @ current_jac
        # after = time.time()

        #print(f"Time taken for inversion: {after - before}")
    return x

@numba.njit()
def minimize_newton_raphson_fast(
    x_0: types.float64[:], params: Parameters, eps: float
) -> types.float64[:]:
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
    current_jac = jacobian_parametrized(x, params)
    while np.linalg.norm(current_jac) > eps:
        # before = time.time()
        current_hess = hessian_parametrized_fast(x, params)
        # after = time.time()

        #print(f"Time taken for hessian: {after - before}")

        # before = time.time()
        current_jac = jacobian_parametrized(x, params)
        # after = time.time()

        #print(f"Time taken for jacobian: {after - before}")

        # before = time.time()
        x = x - np.linalg.inv(current_hess) @ current_jac
        # after = time.time()

        #print(f"Time taken for inversion: {after - before}")
    return x