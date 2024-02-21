from typing import Tuple, List

import numpy as np
import pandas as pd
from numpy._typing import ArrayLike
from sksurv.datasets import load_aids, get_x_y, load_whas500

from verticox.common import SEER, NotEnoughFeaturesException, _uncensored


def load_aids_data_with_dummies(endpoint: str = "aids") -> (pd.DataFrame, np.array):
    """
    Load the aids dataset from sksurv. Categorical features will be converted to one-hot encoded
    columns.

    Args:
        endpoint: either "aids" or "death". Default is "aids".

    Returns:

    """
    covariates, outcome = load_aids(endpoint)

    categorical_columns = [name for name, dtype in covariates.dtypes.items() if dtype == "category"]
    dummies = pd.get_dummies(covariates[categorical_columns]).astype(float)
    numerical_df = covariates.drop(categorical_columns, axis=1)

    combined = pd.concat([numerical_df, dummies], axis=1)
    return combined, outcome


def load_seer() -> (pd.DataFrame, np.array):
    """
    Load the seer dataset from zenodo.

    Zhandos Sembay. (2021). Seer Breast Cancer Data [Data set]. Zenodo.
    https://doi.org/10.5281/zenodo.5120960
    Returns:

    """

    df = pd.read_csv(
        "https://zenodo.org/records/5120960/files/SEER%20Breast%20Cancer%20Dataset%20.csv?download=1")
    # Remove empty column
    df = df.drop(columns=["Unnamed: 3"])

    # Convert string columns to categorical.
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")

    # Split off outcome
    outcome = df[["Survival Months", "Status"]]
    df = df.drop(columns=outcome.columns)

    # Categorical to dummies
    for name in df.columns:
        column = df[name]
        if column.dtype.name == "category":
            dummies = pd.get_dummies(column)
            df = df.drop(columns=name)
            df = pd.concat([df, dummies], axis=1)

    _, events = get_x_y(outcome, attr_labels=["Status", "Survival Months"], pos_label="Dead")

    return df, events


def get_test_dataset(
        limit=None, feature_limit=None, include_right_censored=True, dataset: str = SEER
) -> Tuple[np.array, np.array, List]:
    """
    Prepare and provide the whas500, aids or SEER dataset for testing purposes.

    Args:
        dataset: there are two datasets available: "whas500" and "aids". Whas500 is the default.
        limit: Limit on the number of samples, by default all 500 samples will be used
        feature_limit:  Limit on the features that should be included
        include_right_censored: Whether to include right censored data. By default it is True

    Returns: A tuple containing features, outcome and column names as
     (FEATURES, OUTCOME, COLUMN_NAMES)

    """
    match dataset:
        case "whas500":
            features, events = load_whas500()
        case "aids":
            features, events = load_aids_data_with_dummies()
        case "seer":
            features, events = load_seer()
        case other:
            raise Exception(f"Dataset \"{other}\" is not available.")

    if feature_limit is not None and len(features.columns) < feature_limit:
        raise NotEnoughFeaturesException(f"Desired number of features ({feature_limit})"
                                         f" is larger than number of available features "
                                         f"({len(features.columns)}).")

    if not include_right_censored:
        features = features[_uncensored(events)]
        events = events[_uncensored(events)]
    if include_right_censored and limit:
        # Make sure there's both right censored and non-right censored data
        # Since the behavior should be deterministic we will still just take the first samples we
        # that meets the requirements.
        non_censored = _uncensored(events)
        non_censored_idx = np.argwhere(non_censored).flatten()
        right_censored_idx = np.argwhere(~non_censored).flatten()

        limit_per_type = limit // 2

        non_censored_idx = non_censored_idx[:limit_per_type]
        right_censored_idx = right_censored_idx[: (limit - limit_per_type)]

        all_idx = np.concatenate([non_censored_idx, right_censored_idx])

        events = events[all_idx]
        features = features.iloc[all_idx]

    features = features.select_dtypes(include="number")

    if limit:
        features = features.head(limit)
        events = events[:limit]

    if feature_limit:
        columns = features.columns[:feature_limit]
        features = features[columns]

    return features.values.astype(float), events, list(features.columns)
