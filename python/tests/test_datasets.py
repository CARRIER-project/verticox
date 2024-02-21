import pandas as pd

from verticox.datasets import load_aids_data_with_dummies, get_test_dataset, \
    get_prioritized_features


def test_load_aids_data_converts_dummies():
    covariates, outcome = load_aids_data_with_dummies("aids")

    datatypes = covariates.dtypes
    assert pd.Categorical not in datatypes


def test_get_test_dataset_get_all_features_dummies():
    features, outcome, column_names = get_test_dataset(10, feature_limit=10, dataset="aids")

    assert len(column_names) == 10


def test_dummy_features_are_mixed():
    # Dataframe with 2 categorical columns
    df = pd.DataFrame({"animal": ["dog", "cat"], "sex": ["female", "male"]})

    prioritized = get_prioritized_features(df)

    target_column_order = ["cat", "female"]

    assert list(prioritized.columns) == target_column_order
