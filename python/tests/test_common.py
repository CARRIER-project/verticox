from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from verticox import common
from verticox.common import load_aids_data_with_dummies, get_test_dataset


def test_group_samples_at_risk_numbers_descend():
    event_times = np.array(
        [
            4,
            7,
            7,
            6,
            7,
            23,
            2,
            4,
        ]
    )
    # Testing if the resulting list descends in numbers
    previous_length = len(event_times) + 1

    Rt = common.group_samples_at_risk(event_times)
    for t in sorted(Rt.keys()):
        length = len(Rt[t])

        assert length < previous_length

        previous_length = length


def test_group_samples_at_risk():
    """
    Testing with duplicate event times on t=2
    """
    event_times = [1, 2, 2, 3]
    result = common.group_samples_at_risk(event_times)

    target = {1: [0, 1, 2, 3], 2: [1, 2, 3], 3: [3]}

    assert result.keys() == target.keys()

    for k, v in result.items():
        print(f"Comparing {k}, {v} with {target[k]}")
        assert_array_equal(v, np.array(target[k]))


def test_group_deaths():
    include = [1, 1, 0, 1]
    event_times = [2, 2, 3, 4]

    result = common.group_samples_on_event_time(event_times, include)

    result = {key: list(value) for key, value in result.items()}

    target = {2: [0, 1], 4: [3]}

    TestCase().assertDictEqual(result, target)


def test_load_aids_data_converts_dummies():
    covariates, outcome = load_aids_data_with_dummies("aids")

    datatypes = covariates.dtypes
    assert pd.Categorical not in datatypes


def test_get_test_dataset_get_all_features_dummies():
    features, outcome, column_names = get_test_dataset(10, feature_limit=10, dataset="aids")

    assert len(column_names) == 10
