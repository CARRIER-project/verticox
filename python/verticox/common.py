import numpy as np


def group_samples_at_risk(event_times: np.array):
    """
    Group the indices of samples on whether they are at risk at a certain time.

    A sample is at risk at a certain time when its event time is greater or equal that time.

    :param event_times:
    :return:
    Args:
        event_times:

    Returns:

    """
    unique_times = np.unique(event_times)

    grouped = {}

    for t in unique_times:
        grouped[t] = np.argwhere(event_times >= t)

    return grouped

