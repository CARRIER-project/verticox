from typing import Dict, Union, List

import numpy as np
from numpy.typing import ArrayLike


def group_samples_at_risk(event_times: ArrayLike,
                          include: ArrayLike) -> Dict[Union[float, int], List[int]]:
    """
    Groups the indices of samples on whether they are at risk at a certain time.

    A sample is at risk at a certain time when its event time is greater or equal that time.

    TODO: Figure out what to do with right-censored samples
    Args:
        event_times:
        right_censored:

    Returns:

    """
    unique_times = np.unique(event_times)

    grouped = {}

    for t in unique_times:
        grouped[t] = np.argwhere(event_times >= t).flatten()

    return grouped

def group_samples_on_event_time(event_times, include):
    """

    Args:
        event_times:
        include: Include means the samples is NOT right censored

    Returns:

    """
    Dt = {}

    for idx, (t, i) in enumerate(zip(event_times, include)):
        if i:
            Dt[t] = Dt.get(t, []) + [idx]

    return Dt
