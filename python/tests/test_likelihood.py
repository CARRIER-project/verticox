import numba
import numpy as np
from numba import typed, types
from numpy import array
from pytest import mark

from verticox.likelihood import (
    Parameters,
    parametrized,
    derivative_1,
    minimize_newton_raphson,
    jacobian_parametrized,
)

NUM_PATIENTS = 3
NUM_FEATURES = 2
K = 2
RT = {1.0: np.array([0]), 2.0: np.array([1])}
EVENT_TIMES = np.arange(NUM_PATIENTS, dtype=float)
Z = numba.float64(np.arange(NUM_PATIENTS))
GAMMA = Z
SIGMA = Z
RHO = 2.0
DT = {t: np.array([t], dtype=int) for t in EVENT_TIMES}
EPSILON = 1e-5
DEATHS_PER_T = {t: len(list_) for t, list_ in DT.items()}
RELEVANT_EVENT_TIMES = {
    k: np.array([t for t in RT.keys() if t <= k]) for k in RT.keys()
}


def get_typed_relevant_t():
    typed_rel_t = typed.Dict.empty(types.float64, types.float64[:])

    for k, v in RELEVANT_EVENT_TIMES.items():
        typed_rel_t[k] = v
    return typed_rel_t


def get_typed_Rt():
    typed_Rt = typed.Dict.empty(types.float64, types.int64[:])

    for key, value in RT.items():
        typed_Rt[key] = value

    return typed_Rt


def get_typed_Dt() -> types.DictType(types.float64, types.int64[:]):
    typed_Dt = typed.Dict.empty(types.float64, types.int64[:])

    for key, value in DT.items():
        typed_Dt[key] = value
    return typed_Dt


def get_typed_deaths_per_t():
    typed_deaths_per_t = typed.Dict.empty(types.float64, types.int64)

    for key, value in DEATHS_PER_T.items():
        typed_deaths_per_t[key] = value

    return typed_deaths_per_t


PARAMS = Parameters(
    GAMMA,
    SIGMA,
    RHO,
    get_typed_Rt(),
    K,
    EVENT_TIMES,
    get_typed_Dt(),
    get_typed_deaths_per_t(),
    get_typed_relevant_t(),
)


def test_lz_outputs_scalar():
    result = parametrized(Z, PARAMS)

    assert np.isscalar(result)


def test_lz_derivative_1_output_scalar():
    u_index = 2

    result = derivative_1(Z, PARAMS, u_index)

    assert np.isscalar(result)


def test_one_dim_newton_rhapson():
    @numba.njit
    def f(x: types.float64[:], params) -> types.float64:
        return np.square(x).sum()

    @numba.njit
    def jac(x: types.float64[:], params) -> types.float64[:]:
        return numba.float64([2 * x[0], 2 * x[1]])

    @numba.njit
    def hess(x: types.float64, params) -> types.float64[:, :]:
        return numba.float64([[2, 0], [0, 2]])

    result = minimize_newton_raphson(
        np.array([1, 1], dtype=float), jac, hess, PARAMS, eps=EPSILON
    )

    np.testing.assert_array_almost_equal(result, np.array([0, 0]))


def test_newton_rhapson_finds_root_when_starting_at_root():
    @numba.njit
    def f(x: types.float64[:], params: Parameters) -> types.float64:
        return np.square(x).sum()

    @numba.njit
    def jac(x: types.float64[:], params: Parameters) -> types.float64[:]:
        return numba.float64([2 * x[0], 2 * x[1]])

    @numba.njit
    def hess(x: types.float64[:], params: Parameters) -> types.float64[:, :]:
        return numba.float64([[2, 0], [0, 2]])

    start_z = np.array([1, 1], dtype=float)

    result = minimize_newton_raphson(start_z, jac, hess, PARAMS, EPSILON)

    np.testing.assert_array_almost_equal(result, np.array([0, 0]))


@mark.skip("Will fail due to overflow error with float64")
def test_problematic_jacobian_not_nan():
    # This test currently fails because of overflow
    params = Parameters(
        gamma=array([-2.38418579e-07, 3.81469727e-06, 8.99999809e00]),
        sigma=array([-7255.40722656, -7255.40722656, -5700.67675781]),
        rho=1,
        Rt={1.0: array([0, 1, 2]), 297.0: array([0, 2]), 1496.0: array([2])},
        K=1,
        event_times=array([2.970e02, 1.000e00, 1.496e03]),
        Dt={2.970e02: [0], 1.000e00: [1], 1.496e03: [2]},
    )
    x = np.array([-7253.769531488418579, -7253.7695274353027344, -5699.390626907348633])

    jac = jacobian_parametrized(x, params)

    assert not np.isnan(jac).any()
