from unittest import TestCase

import numpy as np
from numpy import array
from numpy.testing import assert_array_equal
from pytest import mark

from verticox.aggregator import group_samples_at_risk, group_samples_on_event_time
from verticox.likelihood import Parameters, parametrized, derivative_1, minimize_newton_raphson, \
    jacobian_parametrized, hessian_parametrized

NUM_PATIENTS = 3
NUM_FEATURES = 2
K = 2
RT = {1: [0], 2: [1]}
EVENT_TIMES = np.arange(NUM_PATIENTS)
Z = np.arange(NUM_PATIENTS)
GAMMA = Z
SIGMA = Z
RHO = 2
DT = {t: [t] for t in EVENT_TIMES}

PARAMS = Parameters(GAMMA, SIGMA, RHO, RT, K, EVENT_TIMES, DT)


def test_lz_outputs_scalar():
    result = parametrized(Z, PARAMS)

    assert np.isscalar(result)


def test_group_samples_at_risk():
    event_times = [1, 2, 2, 3]
    result = group_samples_at_risk(event_times)

    target = {1: [0, 1, 2, 3], 2: [1, 2, 3], 3: [3]}

    assert result.keys() == target.keys()

    for k, v in result.items():
        print(f'Comparing {k}, {v} with {target[k]}')
        assert_array_equal(v, np.array(target[k]))


def test_lz_derivative_1_output_scalar():
    u_index = 2

    result = derivative_1(Z, PARAMS, u_index)

    assert np.isscalar(result)


def test_one_dim_newton_rhapson():
    def f(x):
        return np.square(x).sum()

    def jac(x):
        return np.array([2 * x[0], 2 * x[1]])

    def hess(x):
        return np.array([
            [2, 0],
            [0, 2]
        ])

    result = minimize_newton_raphson(np.array([1, 1]), f, jac, hess, 1e-4)

    np.testing.assert_array_almost_equal(result, np.array([0, 0]))


def test_newton_rhapson_finds_root_when_starting_at_root():
    def f(x):
        return np.square(x).sum()

    def jac(x):
        return np.array([2 * x[0], 2 * x[1]])

    def hess(x):
        return np.array([
            [2, 0],
            [0, 2]
        ])

    result = minimize_newton_raphson(np.array([1, 1]), f, jac, hess, 1e-4)

    np.testing.assert_array_almost_equal(result, np.array([0, 0]))


def test_group_deaths():
    include = [1, 1, 0, 1]
    event_times = [2, 2, 3, 4]

    result = group_samples_on_event_time(event_times, include)
    target = {2: [0, 1], 4: [3]}

    TestCase().assertDictEqual(result, target)


def test_newton_rhaphson_problematic_start_z():
    params = Parameters(gamma=array([-2.38418579e-07, 3.81469727e-06, 8.99999809e+00]),
                           sigma=array([-7255.40722656, -7255.40722656, -5700.67675781]), rho=1,
                           Rt={1.0: array([0, 1, 2]), 297.0: array([0, 2]), 1496.0: array([2])},
                           K=1, event_times=array([2.970e+02, 1.000e+00, 1.496e+03]),
                           Dt={2.970e+02: [0], 1.000e+00: [1], 1.496e+03: [2]})
    start_x = np.array([-7253.769531488418579, -7253.7695274353027344, -5699.390626907348633])

    def f(x):
        return parametrized(x, params)

    def jacobian(x):
        return jacobian_parametrized(x, params)

    def hessian(x):
        return hessian_parametrized(x, params)

    minimum = minimize_newton_raphson(start_x, f, jacobian, hessian, 1e-4)

    assert not np.isnan(minimum).any()


@mark.skip('Will fail due to overflow error with float64')
def test_problematic_jacobian_not_nan():
    # This test currently fails because of overflow
    params = Parameters(gamma=array([-2.38418579e-07, 3.81469727e-06, 8.99999809e+00]),
                           sigma=array([-7255.40722656, -7255.40722656, -5700.67675781]), rho=1,
                           Rt={1.0: array([0, 1, 2]), 297.0: array([0, 2]), 1496.0: array([2])},
                           K=1, event_times=array([2.970e+02, 1.000e+00, 1.496e+03]),
                           Dt={2.970e+02: [0], 1.000e+00: [1], 1.496e+03: [2]})
    x = np.array([-7253.769531488418579, -7253.7695274353027344, -5699.390626907348633])

    jac = jacobian(x, params)

    assert not np.isnan(jac).any()
