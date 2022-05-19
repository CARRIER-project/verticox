import numpy as np
from verticox.datanode import DataNode


def test_sum_covariates_returns_one_dim_array():
    num_patients = 2
    num_features = 2

    covariates = np.arange(num_patients * num_features).reshape((num_patients, num_features))

    result = DataNode.sum_covariates(covariates)
    assert result.shape == (
        num_features,), f'Result is not one dimensional but shape {result.shape}'


def test_multiply_covariates_returns_scalar():
    num_patients = 2
    num_features = 2

    covariates = np.arange(num_patients * num_features).reshape((num_patients, num_features))

    result = DataNode.multiply_covariates(covariates)
    assert np.isscalar(result), f'Result is not scalar but shape {result.shape}'


def test_elementwise_multiply_sum():
    two_dim = np.array([[1, 2], [3, 4], [5, 6]])
    one_dim = np.array([1, 2, 3])

    result = DataNode.elementwise_multiply_sum(one_dim, two_dim)

    assert result.shape == (two_dim.shape[
                                1],), f'Result shape is not same as number of columns in two_dim ({two_dim.shape[1]}) but {result.shape}'

    np.testing.assert_array_equal(result, np.array([22, 28]))


def test_local_update():
    num_patients = 3
    num_features = 2

    rho = 1
    covariates = np.arange(num_patients * num_features).reshape((num_patients, num_features))
    z = np.arange(num_patients)
    gamma = np.arange(num_patients)
    multiplied_cov = DataNode.multiply_covariates(covariates)
    summed_cov = DataNode.sum_covariates(covariates)

    sigma = DataNode.local_update(covariates, z, gamma, rho, multiplied_cov, summed_cov)

    assert sigma.shape == (
    num_patients,), f'Updated value is not an array of shape {(num_features,)} but of shape: ' \
                    f'{sigma.shape}'
