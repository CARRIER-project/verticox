from collections import namedtuple

import numpy as np
from numba import typed, types
from numpy.typing import ArrayLike

Split = namedtuple("Split", ("train", "test", "all"))
WHAS500 = "whas500"
AIDS = "aids"
SEER = "seer"


@np.vectorize
def is_uncensored(event):
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
