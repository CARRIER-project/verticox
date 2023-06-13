from multiprocessing import Process
from time import sleep
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import verticox.ssl
from verticox.datanode import DataNode, serve
from verticox.grpc.datanode_pb2 import Empty, Subset

NUM_PATIENTS = 3
NUM_FEATURES = 5
PORT = 9999


@pytest.fixture()
def data():
    return data_nofixture()


def data_nofixture():
    data = np.arange(4).reshape((2, 2))
    feature_names = ['blood_pressure', 'heart_rate']

    return data, feature_names


@pytest.fixture()
def beta():
    return np.array([0.1, 0.2])


def test_sum_covariates_returns_one_dim_array():
    covariates = np.arange(NUM_PATIENTS * NUM_FEATURES).reshape((NUM_PATIENTS, NUM_FEATURES))

    result = DataNode._sum_covariates(covariates)
    assert result.shape == (
        NUM_FEATURES,), f'Result is not one dimensional but shape {result.shape}'


def test_multiply_covariates_returns_matrix():
    num_patients = 2
    num_features = 3

    covariates = np.arange(num_patients * num_features).reshape((num_patients, num_features))

    result = DataNode._multiply_features(covariates)
    assert result.shape == (num_features, num_features)


def test_local_update_sigma_shape_is_num_patients():
    rho = 1
    covariates = np.arange(NUM_PATIENTS * NUM_FEATURES).reshape((NUM_PATIENTS, NUM_FEATURES))
    z = np.arange(NUM_PATIENTS)
    gamma = np.arange(NUM_PATIENTS)
    multiplied_cov = DataNode._multiply_features(covariates)
    summed_cov = DataNode._sum_covariates(covariates)

    sigma, beta = DataNode._local_update(covariates, z, gamma, rho, multiplied_cov, summed_cov)

    assert sigma.shape == (NUM_PATIENTS,), \
        f'Updated value is not an array of shape {(NUM_PATIENTS,)} but of shape: {sigma.shape}'

    assert beta.shape == (NUM_FEATURES,), \
        f'Updated value is not an array of shape {(NUM_FEATURES,)} but of shape: {beta.shape}'


def test_get_num_features_returns_num_features():
    data = np.arange(NUM_PATIENTS * NUM_FEATURES).reshape((NUM_PATIENTS, NUM_FEATURES))

    datanode = DataNode(features=data)

    assert datanode.getNumFeatures(Empty()).numFeatures == NUM_FEATURES


def test_get_feature_names_gives_names_if_they_exist(data):
    features, feature_names = data
    datanode = DataNode(features=features, feature_names=feature_names)
    result = datanode.getFeatureNames(request=Empty(), context=None)

    assert result.names == ['blood_pressure', 'heart_rate']


def test_get_feature_names_aborts_if_not_exist(data):
    data, _ = data
    datanode = DataNode(features=data, )

    mock_context = MagicMock()
    datanode.getFeatureNames(request=Empty(), context=mock_context)

    mock_context.abort.assert_called_once()


# Because this test is juggling multiple processes it doesn't go well with the pytest runner.
@pytest.mark.skip
def test_can_make_secure_connection_with_datanode(data):
    features, feature_names = data
    port = PORT
    server_process = Process(target=serve,
                             kwargs={'features': features, 'feature_names': feature_names,
                                     'port': port, 'address': '127.0.0.1'})
    server_process.start()

    # Wait until server has started
    print('Waiting for server to start...')
    sleep(5)
    print('Continuing....')

    def get_feature_names():
        host = '127.0.0.1'
        stub = verticox.ssl.get_secure_stub(host, port)

        names = stub.getFeatureNames(Empty())

        if list(names.names) != feature_names:
            raise Exception('Result is not as expected')

    client_process = Process(target=get_feature_names)

    client_process.start()
    client_process.join()

    client_process.kill()
    server_process.join()
    server_process.kill()


def test_get_record_level_sigma_full_dataset(data):
    features, _ = data
    beta = np.array([0.1, 0.2])

    datanode = DataNode(features=features, beta=beta)
    num_records = features.shape[0]
    indices = np.arange(num_records)
    subset = Subset(indices=indices)

    result = datanode.getRecordLevelSigma(subset).sigma

    target = [0.2, 0.8]

    assert_array_almost_equal(target, result)


def test_get_average_sigma(data, beta):
    features, _ = data

    datanode = DataNode(features=features, beta=beta)
    num_records = features.shape[0]
    indices = np.arange(num_records)

    subset = Subset(indices=indices)
    result = datanode.getAverageSigma(subset).sigma

    target = 0.5
    np.testing.assert_almost_equal(target, result)


def test_compute_partial_hazard_ratio_1_record(data, beta):
    features, _ = data
    subset = [0]
    datanode = DataNode(features, beta=beta)
    request = Subset(indices=subset)

    result = datanode.computePartialHazardRatio(request)
    ratios = np.array(result.partialHazardRatios)

    target = np.dot(features[0], beta)
    target = target.reshape((1,))

    np.testing.assert_array_almost_equal(ratios, target)


def test_compute_partial_hazard_ratio_multiple_records(beta):
    total_rows = 6 * 2
    subset = [0, 1, 2]

    features = np.arange(total_rows).reshape((-1, 2))
    datanode = DataNode(features, beta=beta)
    request = Subset(indices=subset)

    result = datanode.computePartialHazardRatio(request)
    ratios = np.array(result.partialHazardRatios)

    subset_features = features[subset]
    target = [np.dot(subset_features[i], beta) for i in range(subset_features.shape[0])]
    target = np.array(target)

    np.testing.assert_array_almost_equal(ratios, target)


if __name__ == '__main__':
    test_can_make_secure_connection_with_datanode(data_nofixture())
