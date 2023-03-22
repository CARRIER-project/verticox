from unittest import TestCase
from unittest.mock import MagicMock

from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np

from verticox import common
from verticox.aggregator import Aggregator
from verticox.grpc.datanode_pb2 import RecordLevelSigma, NumSamples


def test_compute_deaths_per_t_two_event_times_no_right_censored():
    event_times = np.array([1, 1, 2])
    event_happened = np.array([True, True, True])

    result = Aggregator._compute_deaths_per_t(event_times, event_happened)
    result = dict(result)

    TestCase().assertDictEqual(result, {1: 2, 2: 1})


def test_compute_deaths_per_t_with_right_censored():
    event_times = np.array([1, 1, 2])
    event_happened = np.array([False, True, True])

    result = Aggregator._compute_deaths_per_t(event_times, event_happened)
    result = dict(result)

    TestCase().assertDictEqual(result, {1: 1, 2: 1})


def test_compute_deaths_per_t_with_right_censored_returns_0_deaths():
    """
    Right censored samples shouldn't count as a death. Therefore, if all
    samples with the same event time are right censored, there should be 0 deaths
    at that timestep.
    """
    event_times = np.array([1, 1, 2])
    event_happened = np.array([True, True, False])

    result = Aggregator._compute_deaths_per_t(event_times, event_happened)
    result = dict(result)

    TestCase().assertDictEqual(result, {1: 2, 2: 0})


# TODO: test with more institutions and features
def test_compute_baseline_hazard():
    # We need enough data to be able to do a full analysis
    num_records = 60
    num_features = 2
    num_institutions = 2
    features_per_institution = num_features // num_institutions
    features, events, names = common.get_test_dataset(num_records, num_features)

    event_times, event_happened = common.unpack_events(events)

    centralized_model = CoxPHSurvivalAnalysis()
    centralized_model.fit(features, events)

    mock_datanodes = []
    # Mock the datanodes
    for i in range(num_institutions):
        feature_idx = i * features_per_institution
        feature_idx_end = feature_idx + features_per_institution
        subfeatures = features[:, feature_idx:feature_idx_end]
        coefs = centralized_model.coef_[feature_idx:feature_idx_end]

        record_level_sigma = np.apply_along_axis(lambda x: np.dot(x, coefs),
                                                 axis=1, arr=subfeatures)

        datanode = MagicMock()
        datanode.getNumSamples.return_value = NumSamples(numSamples=num_records)
        datanode.getRecordLevelSigma.return_value = RecordLevelSigma(sigma=record_level_sigma)
        mock_datanodes.append(datanode)

    aggregator = Aggregator(institutions=mock_datanodes, event_times=event_times,
                            event_happened=event_happened)

    baseline_hazard = aggregator.compute_baseline_hazard_function()
    print(baseline_hazard)
