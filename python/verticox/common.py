from typing import Dict, Union, List

import numpy as np
from numpy.typing import ArrayLike


def group_samples_at_risk(event_times: ArrayLike) -> Dict[Union[float, int], List[int]]:
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

    grouped = {}

    for t in unique_times:
        grouped[t] = np.argwhere(event_times >= t).flatten()

    return grouped


def group_samples_on_event_time(event_times, event_happened):
    """

    Args:
        event_times:
        event_happened: Include means the samples is NOT right censored

    Returns:

    """
    Dt = {}

    for idx, (t, i) in enumerate(zip(event_times, event_happened)):
        if i:
            Dt[t] = Dt.get(t, []) + [idx]

    return Dt
