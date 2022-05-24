import numpy as np
from verticox.datanode import DataNode
from verticox.grpc.datanode_pb2 import Empty

NUM_PATIENTS = 3
NUM_FEATURES = 5


def test_sum_covariates_returns_one_dim_array():
    covariates = np.arange(NUM_PATIENTS * NUM_FEATURES).reshape((NUM_PATIENTS, NUM_FEATURES))

    result = DataNode.sum_covariates(covariates)
    assert result.shape == (
        NUM_FEATURES,), f'Result is not one dimensional but shape {result.shape}'


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
    rho = 1
    covariates = np.arange(NUM_PATIENTS * NUM_FEATURES).reshape((NUM_PATIENTS, NUM_FEATURES))
    z = np.arange(NUM_PATIENTS)
    gamma = np.arange(NUM_PATIENTS)
    multiplied_cov = DataNode.multiply_covariates(covariates)
    summed_cov = DataNode.sum_covariates(covariates)

    sigma = DataNode.local_update(covariates, z, gamma, rho, multiplied_cov, summed_cov)

    assert sigma.shape == (
        NUM_PATIENTS,), f'Updated value is not an array of shape {(NUM_FEATURES,)} but of shape: ' \
                        f'{sigma.shape}'


def test_get_num_features_returns_num_features():
    data = np.arange(NUM_PATIENTS * NUM_FEATURES).reshape((NUM_PATIENTS, NUM_FEATURES))

    datanode = DataNode(features=data)

    assert datanode.getNumFeatures(Empty()).numFeatures == NUM_FEATURES
