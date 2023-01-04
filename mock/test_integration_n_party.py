import os
from pathlib import Path
from numba import typed, types
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from verticox.common import group_samples_on_event_time
from verticox.datanode import DataNode
from verticox.scalarproduct import NPartyScalarProductClient
from verticox import likelihood, likelihood_n_party_2, common

CENSOR_FEATURE = 'event_happened'
CENSOR_VALUE = True
TIME_FEATURE = 'event_time'
COMMODITY_ADDRESS = 'commodity'
DATANODE1 = 'javanode1'
DATANODE2 = 'javanode2'
NUM_PARTIES = 2
N_PARTY_PRECISION = 4


def main():
    data_dir = os.environ['DATA_DIR']
    data_dir = Path(data_dir)
    data_1 = pd.read_parquet(data_dir / 'data_1.parquet')
    outcome = pd.read_parquet(data_dir / 'outcome.parquet')

    print(outcome.event_time.values)

    # Initialize java nodes
    n_party_client = NPartyScalarProductClient(commodity_address=COMMODITY_ADDRESS,
                                               external_commodity_address=COMMODITY_ADDRESS,
                                               other_addresses=[DATANODE1, DATANODE2])
    n_party_client.initialize_servers()

    test_sum_dt(data_1, outcome)

    print("Testing sum_records_at_risk")
    test_sum_records_at_risk(n_party_client, outcome)

    print('\n\nTesting jacobian')
    test_jacobian(COMMODITY_ADDRESS, outcome)


def test_sum_dt(data_1, outcome):
    Dt = group_samples_on_event_time(outcome.event_time.values, outcome.event_happened.values)
    regular_sum_Dt = DataNode.compute_sum_Dt(Dt, data_1.values)
    print(f'Regular sum_dt: {regular_sum_Dt}')
    local_feature_names = data_1.columns
    n_party_sum_Dt = DataNode.compute_sum_Dt_n_party_scalar_product(local_feature_names,
                                                                    CENSOR_FEATURE,
                                                                    CENSOR_VALUE,
                                                                    COMMODITY_ADDRESS)
    print(f'N party sum dt: {n_party_sum_Dt}')
    assert_array_almost_equal(regular_sum_Dt, n_party_sum_Dt)


def test_sum_records_at_risk(n_party_client: NPartyScalarProductClient, outcome: pd.DataFrame):
    # Pick a time t somewhere in the middle
    some_t = outcome.event_time.median()

    print(f'Some t: {some_t}')

    z = np.ones(len(outcome))

    exp_k_z = np.exp(NUM_PARTIES * z)
    selection = outcome.event_time.values >= some_t
    print(f'Selection: {selection}')
    regular_result = exp_k_z[selection].sum()

    n_party_result = n_party_client.sum_records_at_risk(exp_k_z.tolist(), some_t, TIME_FEATURE)
    print(n_party_result)
    np.testing.assert_almost_equal(n_party_result, regular_result, N_PARTY_PRECISION)


def test_jacobian(commodity_address, outcome: pd.DataFrame):
    event_times = outcome.event_time.values
    z = np.zeros(len(outcome), dtype=float)
    sigma = np.array(z)
    gamma = np.array(z)
    rho = 2.
    Rt = common.group_samples_at_risk(event_times)
    K = 2
    dt = outcome.groupby('event_time').size().to_dict()

    distinct_event_times = np.unique(event_times)
    deaths_per_t = [dt[t] for t in distinct_event_times]
    deaths_per_t = np.array(deaths_per_t, dtype=int)

    print(f'Dt: {dt}')
    print(f'Deaths per t: {deaths_per_t}')

    dt_typed = typed.Dict.empty(types.float64, types.float64)
    for k, v in dt.items():
        dt_typed[k] = v

    relevant_event_times = typed.Dict.empty(types.float64, types.float64[:])

    for k in Rt.keys():
        relevant_event_times[k] = np.array([t for t in Rt.keys() if t <= k])

    regular_params = likelihood.Parameters(gamma, sigma, rho, Rt, K, event_times, dt_typed,
                                           relevant_event_times)
    regular_jacobian = likelihood.jacobian_parametrized(z, regular_params)

    max_likelihood_finder = likelihood_n_party_2.NPartyMaxLikelihoodFinder(commodity_address,
                                                                           gamma, sigma, rho,
                                                                           K, deaths_per_t,
                                                                           distinct_event_times,
                                                                           'event_time')

    n_party_params = likelihood_n_party_2.ConstantParameters(gamma, sigma, rho, K,
                                                             distinct_event_times, deaths_per_t)


    z_based_components = max_likelihood_finder.get_z_based_components(z)
    print(f'z based components: z: {z_based_components.z}'
          f'\nexp_k_z: {z_based_components.exp_k_z}'
          f'\naggregated_hazard_per_t: {z_based_components.aggregated_hazard_per_t}')
    n_party_jacobian = likelihood_n_party_2.jacobian_parametrized(z_based_components,
                                                                  n_party_params)

    np.testing.assert_array_almost_equal(regular_jacobian, n_party_jacobian, 4)


if __name__ == '__main__':
    main()
