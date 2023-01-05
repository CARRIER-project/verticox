import numpy as np
from numba import typed, types
from numpy.typing import ArrayLike
from sksurv.datasets import load_whas500


@np.vectorize
def _uncensored(event):
    return event[0]


def group_samples_at_risk(event_times: ArrayLike) -> types.DictType(types.float64, types.int64[:]):
    """
    Groups the indices of samples on whether they are at risk at a certain time.

    A sample is at risk at a certain time when its event time is greater or equal that time.

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


def group_samples_on_event_time(event_times, event_happened) -> \
        types.DictType(types.float64, types.int64[:]):
    """

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


def get_test_dataset(limit=None, feature_limit=None, include_right_censored=True):
    """
    Prepare and provide the whas500 dataset for testing purposes.

    Args:
        limit: Limit on the number of samples, by default all 500 samples will be used
        feature_limit:  Limit on the features that should be included
        include_right_censored: Whether to include right censored data. By default it is True

    Returns: A tuple containing features, outcome and column names as
     (FEATURES, OUTCOME, COLUMN_NAMES)

    """
    features, events = load_whas500()

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
        right_censored_idx = right_censored_idx[:(limit - limit_per_type)]

        all_idx = np.concatenate([non_censored_idx, right_censored_idx])

        events = events[all_idx]
        features = features.iloc[all_idx]

    numerical_columns = features.columns[features.dtypes == float]

    features = features[numerical_columns]

    if limit:
        features = features.head(limit)
        events = events[:limit]

    features = features.values.astype(float)
    if feature_limit:
        features = features[:, :feature_limit]
        numerical_columns = numerical_columns[:feature_limit]
    return features, events, list(numerical_columns)
