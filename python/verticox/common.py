import numpy as np
from numba import typed, types
from numpy.typing import ArrayLike
from viztracer import log_sparse


@log_sparse
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


@log_sparse
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
