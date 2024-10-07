#! /usr/bin/env python3

from verticox.aggregator import Aggregator
from verticox.likelihood import find_z
from verticox.datasets import get_test_dataset, unpack_events
from verticox.common import group_samples_at_risk, group_samples_on_event_time
import numpy as np
from numba import types, typed
import time
import logging

_logger = logging.getLogger(__name__)


RHO = 1
EPSILON = 1e-4

def main():
    # Prepare data

    features, events, columns = get_test_dataset(feature_limit=3, dataset="aids", limit=50)
    event_times, event_happened = unpack_events(events)
    rho = RHO
    Rt = group_samples_at_risk(event_times)
    Dt = group_samples_on_event_time(event_times, event_happened)
    K = 1
    deaths_per_t = Aggregator.compute_deaths_per_t(event_times, event_happened)
    eps = EPSILON
    relevant_event_times = Aggregator._group_relevant_event_times(event_times)
    
    # Relevant event times should have the same event times as deaths_per_t (so no right-censored samples)
    assert set(relevant_event_times.keys()) == set(deaths_per_t.keys())


    z_start = np.zeros(len(events), dtype=float)
    gamma = z_start.copy()
    sigma = z_start.copy()

    typed_deaths_per_t = typed.Dict.empty(types.float64, types.int64)

    for k,v in deaths_per_t.items():
        typed_deaths_per_t[k] = v

   
    # call find_z

    # TODO: first time is slow, subsequent calls are faster. We need to call this multiple times.
    find_z(
        gamma=gamma,
        sigma=sigma,
        rho=rho,
        Rt=Rt,
        z_start=z_start,
        K=1,
        event_times=event_times,
        Dt=Dt,
        deaths_per_t=typed_deaths_per_t,
        relevant_event_times=relevant_event_times,
        eps=eps
    )
   
    # start timingsu
    start = time.time()
    find_z(
        gamma=gamma,
        sigma=sigma,
        rho=rho,
        Rt=Rt,
        z_start=z_start,
        K=1,
        event_times=event_times,
        Dt=Dt,
        deaths_per_t=typed_deaths_per_t,
        relevant_event_times=relevant_event_times,
        eps=eps
    )
     
    # end timing
    end = time.time()
    print(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    main()
