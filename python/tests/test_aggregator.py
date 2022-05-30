import numpy as np
from numpy.testing import assert_array_equal
from verticox.aggregator import L_z_parametrized, group_samples_at_risk


def test_lz_outputs_scalar():
    num_patients, num_features = 3, 2
    num_parties = 1
    samples_at_risk = {1: [0], 2: [1]}

    z = np.arange(num_patients)
    gamma = z
    sigma = z
    rho = 2

    result = L_z_parametrized(z, num_parties, gamma, sigma, rho, samples_at_risk)

    assert np.isscalar(result)


def test_group_samples_at_risk():
    event_times = [1, 2, 2, 3]
    result = group_samples_at_risk(event_times, [])

    target = {1: [0, 1, 2, 3], 2: [1, 2, 3], 3: [3]}

    assert result.keys() == target.keys()

    for k, v in result.items():
        print(f'Comparing {k}, {v} with {target[k]}')
        assert_array_equal(v, np.array(target[k]))
