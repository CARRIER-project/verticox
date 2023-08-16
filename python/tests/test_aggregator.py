from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from pytest import mark
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis

from verticox import common
from verticox.aggregator import Aggregator
from verticox.grpc.datanode_pb2 import RecordLevelSigma, NumSamples, Subset


def test_compute_deaths_per_t_two_event_times_no_right_censored():
    event_times = np.array([1, 1, 2])
    event_happened = np.array([True, True, True])

    result = Aggregator.compute_deaths_per_t(event_times, event_happened)
    result = dict(result)

    TestCase().assertDictEqual(result, {1: 2, 2: 1})


def test_compute_deaths_per_t_with_right_censored():
    event_times = np.array([1, 1, 2])
    event_happened = np.array([False, True, True])

    result = Aggregator.compute_deaths_per_t(event_times, event_happened)
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

    result = Aggregator.compute_deaths_per_t(event_times, event_happened)
    result = dict(result)

    TestCase().assertDictEqual(result, {1: 2, 2: 0})


@mark.parametrize(
    "num_records,num_features,num_institutions", [(60, 2, 2), (100, 3, 3), (400, 4, 4)]
)
def test_compute_baseline_hazard(num_records, num_features, num_institutions):
    features_per_institution = num_features // num_institutions
    features, events, names = common.get_test_dataset(num_records, num_features)

    event_times, event_happened = common.unpack_events(events)

    centralized_model = CoxPHSurvivalAnalysis()
    centralized_model.fit(features, events)

    # Compute hazard ratio in centralized way
    predictions = predict(features, centralized_model.coef_)

    centralized_baseline_hazard = compute_baseline_hazard(events, predictions)
    centralized_t = centralized_baseline_hazard.x
    centralized_hazard = centralized_baseline_hazard.y

    mock_datanodes = []
    # Mock the datanodes
    for i in range(num_institutions):
        feature_idx = i * features_per_institution
        feature_idx_end = feature_idx + features_per_institution
        subfeatures = features[:, feature_idx:feature_idx_end]
        coefs = centralized_model.coef_[feature_idx:feature_idx_end]

        record_level_sigma = np.apply_along_axis(
            lambda x: np.dot(x, coefs), axis=1, arr=subfeatures
        )

        datanode = MagicMock()
        datanode.getNumSamples.return_value = NumSamples(numSamples=num_records)
        datanode.getRecordLevelSigma.return_value = RecordLevelSigma(
            sigma=record_level_sigma
        )
        mock_datanodes.append(datanode)

    aggregator = Aggregator(
        institutions=mock_datanodes,
        event_times=event_times,
        event_happened=event_happened,
    )

    baseline_hazard = aggregator.compute_baseline_hazard_function(Subset.ALL)
    decentralized_t = baseline_hazard.x
    decentralized_hazard = baseline_hazard.y

    np.testing.assert_almost_equal(decentralized_t, centralized_t)
    np.testing.assert_almost_equal(decentralized_hazard, centralized_hazard, decimal=5)


def predict(features, coefs):
    return np.apply_along_axis(lambda x: compute_hazard_ratio(x, coefs), 1, features)


def test_compute_cumulative_survival_right_length():
    subpopulation_sigmas = 2

    survival = np.array([1, 0.5, 0.1])
    survival_func = StepFunction(np.arange(3), survival)

    cum_survival = Aggregator.compute_cumulative_survival(
        survival_func, subpopulation_sigmas
    )

    assert len(cum_survival.x) == 3
    assert len(cum_survival.y) == 3


def test_compute_cumulative_survival_decreases():
    subpopulation_sigmas = 2

    survival = np.array([1, 0.5, 0.1])
    survival_func = StepFunction(np.arange(3), survival)

    cum_survival = Aggregator.compute_cumulative_survival(
        survival_func, subpopulation_sigmas
    )

    previous = cum_survival.y[0]
    for s in cum_survival.y:
        assert previous >= s
        previous = s


def compute_central_summed_average_sigmas(coefs, features: np.array) -> float:
    return np.dot(coefs, features.mean(axis=0))


def compute_hazard_ratio(features, coefficients):
    return np.exp(np.dot(features, coefficients))


def compute_baseline_hazard(events, predictions):
    event_times, event_happened = common.unpack_events(events)

    at_risk_per_event_time = common.group_samples_at_risk(event_times)

    baseline_hazard_function = {}
    # We need to weight the risk set using the estimated coefficients
    for time, risk_set in at_risk_per_event_time.items():
        risk_set = at_risk_per_event_time[time]
        ratios = predictions[risk_set]
        weighted_risk = ratios.sum()

        baseline_hazard_score = 1 / weighted_risk
        baseline_hazard_function[time] = baseline_hazard_score

    steps, hazard = zip(*sorted(baseline_hazard_function.items()))

    return StepFunction(steps, hazard)
