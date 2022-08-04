import logging
from collections import namedtuple
from typing import Dict, List, Any

import numba
import numpy as np
from numba import types
from numpy.typing import ArrayLike
from viztracer import log_sparse

EPSILON = 1e-4

Parameters = namedtuple('Parameters', ['gamma', 'sigma', 'rho', 'Rt', 'K', 'event_times', 'Dt',
                                       'deaths_per_t'])
_logger = logging.getLogger(__name__)

spec = [
    ('gamma', types.float64[:]),
    ('sigma', types.float64[:]),
    ('Rt', types.DictType(types.float64, types.int64[:])),
    ('event_times', types.float64[:]),
    ('Dt', types.DictType(types.float64, types.int64[:])),
    ('deaths_per_t', types.DictType(types.float64, types.int64))
]


@numba.experimental.jitclass(spec)
class NumbaParameters:
    gamma: np.ndarray
    sigma: np.ndarray
    rho: float
    Rt: Dict[float, np.ndarray]
    K: int
    event_times: np.ndarray
    Dt: Dict[float, np.ndarray]
    deaths_per_t: Dict[float, int]

    def __init__(self, gamma, sigma, rho, Rt, K, event_times, Dt, deaths_per_t):
        self.gamma = gamma
        self.sigma = sigma
        self.rho = rho
        self.Rt = Rt
        self.K = K
        self.event_times = event_times
        self.Dt = Dt
        self.deaths_per_t = deaths_per_t


@log_sparse
def parametrized(z: ArrayLike, params: NumbaParameters):
    """
    Equation 12
    Args:
        z:
        params:

    Returns:

    """
    c1 = component1(z, params)
    c2 = component2(z, params.K, params.sigma, params.gamma, params.rho)

    result = c1 + c2
    return result


def component1(z, params: NumbaParameters):
    result = 0
    for t, group in params.Rt.items():
        z_at_risk = z[group]
        result += params.deaths_per_t[t] * np.log((np.exp(params.K * z_at_risk)).sum())

    return result


def component2(z, K, sigma, gamma, rho):
    element_wise = np.square(z) / 2 - sigma + (gamma / rho) * z
    return K * rho * element_wise.sum()


@log_sparse
def find_z(gamma: ArrayLike, sigma: ArrayLike, rho: float,
           Rt: types.DictType(types.float64, types.int64[:]), z_start: np.ndarray, K: int,
           event_times: types.float64[:], Dt: types.DictType(types.float64, types.int64[:]),
           deaths_per_t: types.DictType(types.float64, types.int64), eps: float = EPSILON):
    params = NumbaParameters(gamma, sigma, rho, Rt, K, event_times, Dt, deaths_per_t)

    _logger.debug(f'Rt: {params.Rt}')

    def L_z(z):
        return parametrized(z=z, params=params)

    def jac(z):
        return jacobian_parametrized(z, params)

    def hessian(z):
        return hessian_parametrized(z, params)

    # minimum = minimize(L_z, z_start, jac=jac, hess=hessian, method=OPTIMIZATION_METHOD,
    #                    options=OPTIMIZATION_OPTIONS)
    minimum = minimize_newton_raphson(z_start, L_z, jac, hessian, eps=eps)

    return minimum


def derivative_1(z, params: Parameters, sample_idx: int):
    """

    Args:
        z:
        params:
        sample_idx:


    Returns:

    """
    u_event_time = params.event_times[sample_idx]

    relevant_event_times = [t for t in params.Rt.keys() if t <= u_event_time]

    # First part
    enumerator = params.K * np.exp(params.K * z[sample_idx])

    first_part = 0
    for t in relevant_event_times:
        denominator = bottom(z, params, t)

        first_part += params.deaths_per_t[t] * (enumerator / denominator)

    # Second part
    second_part = params.K * params.rho * (z[sample_idx] - params.sigma[sample_idx] - (
            params.gamma[sample_idx] / params.rho))

    return first_part + second_part


def bottom(z, params, t):
    denominator = 0.
    for j in params.Rt[t]:
        denominator += np.exp(params.K * z[j])
    return denominator


@log_sparse
def jacobian_parametrized(z: ArrayLike, params: Parameters) -> ArrayLike:
    result = np.zeros(z.shape)

    for i in range(z.shape[0]):
        result[i] = derivative_1(z, params, sample_idx=i)

    return result


@numba.njit()
def bottom(z, params, t):
    denominator = 0.
    for j in params.Rt[t]:
        denominator += np.exp(params.K * z[j])
    return denominator


@numba.njit()
def derivative_2_diagonal(z: ArrayLike, params: Parameters, u):
    u_event_time = params.event_times[u]

    relevant_event_times = [t for t in params.Rt.keys() if t <= u_event_time]

    summed = 0

    for t in relevant_event_times:
        denominator = bottom(z, params, t)

        first_part = np.square(params.K) * np.exp(params.K * z[u]) / denominator

        second_part = np.square(params.K) * np.square(np.exp(params.K * z[u])) / \
                      np.square(denominator)

        summed += params.deaths_per_t[t] * (first_part - second_part)

    return summed + params.K * params.rho


@numba.njit()
def derivative_2_off_diagonal(z: ArrayLike, params, u, v):
    min_event_time = min(params.event_times[u], params.event_times[v])
    relevant_event_times = [t for t in params.Rt.keys() if t <= min_event_time]

    summed = 0

    for t in relevant_event_times:
        summed += params.deaths_per_t[t] * np.square(params.K) * np.exp(params.K * z[u]) * \
                  np.exp(params.K * z[v]) / \
                  np.square(np.exp(params.K * z[params.Rt[t]]).sum())

    return -1 * summed


@numba.njit()
def hessian_parametrized(z: ArrayLike, params: NumbaParameters):
    # The hessian is a N x N matrix where N is the number of elements in z
    N = z.shape[0]
    mat = np.zeros((N, N))

    for u in range(N):
        for v in range(N):

            if u == v:
                # Formula for diagonals
                mat[u, v] = derivative_2_diagonal(z, params, u)
            else:
                # Formula for off-diagonals
                mat[u, v] = derivative_2_off_diagonal(z, params, u, v)

    return mat


def minimize_newton_raphson(x_0, func, jacobian, hessian, eps) -> ArrayLike:
    """
    The terminology is a little confusing here. We are trying to find the minimum,
    but newton-raphson is a root-finding algorithm. Therefore we are looking for the x where the
    norm of the first-order derivative (jacobian) is 0. In the context of newton-rhapson this
    would be function F(x), while our hessian matrix would be F'(x).
    Args:
        x_0:
        jacobian:
        hessian:
        eps:

    Returns:

    """
    x = x_0
    current_jac = jacobian(x)
    while np.linalg.norm(current_jac) > eps:
        # logger.debug(f'Old x: {old_x}')
        # logger.debug(f'new x: {x}')

        current_hess = hessian(x)
        current_jac = jacobian(x)

        x = x - np.matmul(np.linalg.inv(current_hess), current_jac)

        _logger.debug(f'Jacobian: {current_jac}')
        _logger.debug(f'Norm of jacobian: {np.linalg.norm(current_jac)}')
        _logger.debug(f'Lz_inner: {func(x)}')
    return x
