from unittest import TestCase

from verticox.aggregator import Aggregator
import numpy as np


def test_compute_deaths_per_t_two_event_times_no_right_censored():
    event_times = np.array([1, 1, 2])
    event_happened = np.array([True, True, True])

    result = Aggregator._compute_deaths_per_t(event_times, event_happened)

    TestCase().assertDictEqual(result, {1: 2, 2: 1})


def test_compute_deaths_per_t_with_right_censored():
    event_times = np.array([1, 1, 2])
    event_happened = np.array([False, True, True])

    result = Aggregator._compute_deaths_per_t(event_times, event_happened)

    TestCase().assertDictEqual(result, {1: 1, 2: 1})


def test_compute_deaths_per_t_with_right_censored_returns_0_deaths():
    event_times = np.array([1, 1, 2])
    event_happened = np.array([True, True, False])

    result = Aggregator._compute_deaths_per_t(event_times, event_happened)

    TestCase().assertDictEqual(result, {1: 2, 2: 0})
