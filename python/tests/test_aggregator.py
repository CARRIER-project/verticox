import numpy as np
from numpy.testing import assert_array_equal
from verticox.aggregator import Lz, group_samples_at_risk

NUM_PATIENTS = 3
NUM_FEATURES = 2
K = 2
RT = {1: [0], 2: [1]}
EVENT_TIMES = np.arange(NUM_PATIENTS)
Z = np.arange(NUM_PATIENTS)
GAMMA = Z
SIGMA = Z
RHO = 2

PARAMS = Lz.Parameters(GAMMA, SIGMA, RHO, RT, K, EVENT_TIMES)

def test_lz_outputs_scalar():
    result = Lz.parametrized(Z, PARAMS)

    assert np.isscalar(result)


def test_group_samples_at_risk():
    event_times = [1, 2, 2, 3]
    result = group_samples_at_risk(event_times, [])

    target = {1: [0, 1, 2, 3], 2: [1, 2, 3], 3: [3]}

    assert result.keys() == target.keys()

    for k, v in result.items():
        print(f'Comparing {k}, {v} with {target[k]}')
        assert_array_equal(v, np.array(target[k]))


def test_lz_derivative_1_output_scalar():
    u_index = 2

    result = Lz.derivative_1(Z, PARAMS)

    assert np.isscalar(result)
