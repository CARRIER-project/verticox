import logging
from collections import namedtuple
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

EPSILON = 1e-4

Parameters = namedtuple('Parameters', ['gamma', 'sigma', 'rho', 'Rt', 'K', 'event_times', 'Dt'])
_logger = logging.getLogger(__name__)


def parametrized(z: ArrayLike, params: Parameters):
    """
    Equation 12
    Args:
        z:
        params:

    Returns:

    """
    c1 = component1(z, params.K, params.Rt, params.Dt)
    c2 = component2(z, params.K, params.sigma, params.gamma, params.rho)

    result = c1 + c2
    return result


def component1(z, K, Rt, Dt):
    result = 0
    for t, group in Rt.items():
        z_at_risk = z[group]
        result += _get_dt(t, Dt) * np.log((np.exp(K * z_at_risk)).sum())

    return result


def component2(z, K, sigma, gamma, rho):
    element_wise = np.square(z) / 2 - sigma + (gamma / rho) * z
    return K * rho * element_wise.sum()


def find_z(gamma: ArrayLike, sigma: ArrayLike, rho: float,
           Rt: Dict[int, List[int]], z_start: ArrayLike, K: int, event_times: ArrayLike,
           Dt: Dict[int, List[int]], eps: float = EPSILON):
    params = Parameters(gamma, sigma, rho, Rt, K, event_times, Dt)

    _logger.debug(f'Rt: {params.Rt}')

    def L_z(z):
        return parametrized(z=z, params=params)

    _logger.debug(
        f'Finding minimum z starting at\n{z_start.tolist()}\nwith parameters\n{params}')

    def jac(z):
        return jacobian_parametrized(z, params)

    def hessian(z):
        return hessian_parametrized(z, params)

    # minimum = minimize(L_z, z_start, jac=jac, hess=hessian, method=OPTIMIZATION_METHOD,
    #                    options=OPTIMIZATION_OPTIONS)
    minimum = minimize_newton_raphson(z_start, L_z, jac, hessian, eps=eps)

    _logger.debug(f'Found minimum z at {minimum}')
    _logger.debug(f'Lz_outer: {L_z(minimum)}')
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

        first_part += _get_dt(t, params.Dt) * (enumerator / denominator)

    # Second part
    second_part = params.K * params.rho * (z[sample_idx] - params.sigma[sample_idx] - (
            params.gamma[sample_idx] / params.rho))

    return first_part + second_part


def bottom(z, params, t):
    denominator = 0.
    for j in params.Rt[t]:
        denominator += np.exp(params.K * z[j])
    return denominator


def jacobian_parametrized(z: ArrayLike, params: Parameters) -> ArrayLike:
    result = np.zeros(z.shape)

    for i in range(z.shape[0]):
        result[i] = derivative_1(z, params, sample_idx=i)

    return result


def hessian_parametrized(z: ArrayLike, params: Parameters):
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


def derivative_2_diagonal(z: ArrayLike, params: Parameters, u):
    u_event_time = params.event_times[u]

    relevant_event_times = [t for t in params.Rt.keys() if t <= u_event_time]

    summed = 0

    for t in relevant_event_times:
        denominator = bottom(z, params, t)

        first_part = np.square(params.K) * np.exp(params.K * z[u]) / denominator

        second_part = np.square(params.K) * np.square(np.exp(params.K * z[u])) / np.square(
            denominator)

        summed += _get_dt(t, params.Dt) * (first_part - second_part)

    return summed + params.K * params.rho


def derivative_2_off_diagonal(z: ArrayLike, params: Parameters, u, v):
    min_event_time = min(params.event_times[u], params.event_times[v])
    relevant_event_times = [t for t in params.Rt.keys() if t <= min_event_time]

    summed = 0

    for t in relevant_event_times:
        summed += _get_dt(t, params.Dt) * np.square(params.K) * np.exp(params.K * z[u]) * \
                  np.exp(params.K * z[
                      v]) / \
                  np.square(np.exp(params.K * z[params.Rt[t]]).sum())

    return -1 * summed


def _get_dt(t, Dt):
    # If there is no entry for time t it means that it was from a right-censored sample
    # I'm making it an empty list so I can call len on it but it's semantically sound
    d = Dt.get(t, [])
    return len(d)


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
