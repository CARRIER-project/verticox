from collections import namedtuple
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import typed, types
from numpy.typing import ArrayLike
from sksurv.datasets import load_whas500, load_aids, get_x_y

Split = namedtuple("Split", ("train", "test", "all"))
WHAS500 = "whas500"
AIDS = "aids"
SEER = "seer"


@np.vectorize
def _uncensored(event):
    return event[0]


def group_samples_at_risk(
        event_times: ArrayLike,
) -> types.DictType(types.float64, types.int64[:]):
    """
    Groups the indices of samples on whether they are at risk at a certain time.

    A sample is at risk at a certain time when its event- or censor time is greater or equal that
    time.

    Ri is the set of indices of samples with death or censor times occurring
    after ti.
    Args:
        event_times:

    Returns:

    """
    unique_times = np.unique(event_times)

    grouped = typed.Dict.empty(types.float64, types.int64[:])

    for t in unique_times:
        grouped[t] = np.argwhere(event_times >= t).flatten()

    return grouped


def group_samples_on_event_time(
        event_times, event_happened
) -> types.DictType(types.float64, types.int64[:]):
    """
    Group samples based on event time. Right-censored samples are excluded.

    Args:
        event_times:
        event_happened: Include means the samples is NOT right censored

    Returns:

    """
    Dt = {}

    for idx, events in enumerate(zip(event_times, event_happened)):
        t, i = events
        if i:
            Dt[t] = Dt.get(t, []) + [idx]

    typed_Dt = typed.Dict.empty(types.float64, types.int64[:])
    for key, value in Dt.items():
        typed_Dt[key] = np.array(value)

    return typed_Dt


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
) -> Tuple[ArrayLike, ArrayLike, List]:
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

    if len(features.columns) < feature_limit:
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

    print(f"Features dtypes: {features.dtypes}")

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
