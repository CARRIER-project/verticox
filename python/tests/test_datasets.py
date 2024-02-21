import pandas as pd

from verticox.datasets import load_aids_data_with_dummies, get_test_dataset


def test_load_aids_data_converts_dummies():
    covariates, outcome = load_aids_data_with_dummies("aids")

    datatypes = covariates.dtypes
    assert pd.Categorical not in datatypes


def test_get_test_dataset_get_all_features_dummies():
    features, outcome, column_names = get_test_dataset(10, feature_limit=10, dataset="aids")

    assert len(column_names) == 10
