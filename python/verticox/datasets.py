from typing import Tuple, List

import numpy as np
import pandas as pd
from sksurv.datasets import load_aids, get_x_y, load_whas500

from verticox.common import SEER, _uncensored


def get_dummies(categorical_features: pd.DataFrame):
    for c in categorical_features.columns:
        categorical_features[c] = categorical_features[c].astype("category")

    all_dummies = []
    columns = categorical_features.columns

    for c in columns:
        dummies = pd.get_dummies(categorical_features[c])
        all_dummies.append(dummies)

    idx = 0
    num_dummies = len(all_dummies)

    while not all_single_columns(all_dummies):
        current_dummies = all_dummies[idx]

        # Always leave out the last category
        if len(current_dummies.columns) <= 1:
            idx = (idx + 1) % num_dummies
            continue

        column_name = current_dummies.columns[0]
        column = current_dummies[column_name]
        current_dummies.drop(columns=[column_name], inplace=True)
        idx = (idx + 1) % num_dummies
        yield column


def get_prioritized_features(df: pd.DataFrame):
    numerical_features = df.select_dtypes(include="number")
    other_features = df.drop(columns=numerical_features.columns)

    sorted_dummies = list(get_dummies(categorical_features=other_features))

    return pd.concat([numerical_features] + sorted_dummies, axis=1)


def all_single_columns(df):
    for df in df:
        if len(df.columns) > 1:
            return False

    return True


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

    # Split off outcome
    outcome = df[["Survival Months", "Status"]]
    df = df.drop(columns=outcome.columns)

    df = get_prioritized_features(df)

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


def unpack_events(events):
    """
    Unpacks outcome arrays from sksurv into two separate arrays with censor and event time
    :param events:
    :return: (times array, status array)
    """
    times = []
    right_censored = []

    for event in events:
        times.append(event[1])
        right_censored.append(event[0])

    right_censored = np.array(right_censored)
    if right_censored.dtype != bool:
        raise Exception('Status is not boolean.')
    return np.array(times), right_censored


class NotEnoughFeaturesException(Exception):
    pass
