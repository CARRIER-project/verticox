import pandas as pd
from numpy.testing import assert_array_almost_equal
import os
from verticox.common import group_samples_on_event_time
from verticox.datanode import DataNode
from pathlib import Path
from verticox.scalarproduct import NPartyScalarProductClient
import numpy as np

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

    print("Testing sum_records_at_risk")
    test_sum_records_at_risk(n_party_client, data_1, outcome)


def test_sum_records_at_risk(n_party_client: NPartyScalarProductClient, covariates: pd.DataFrame,
                             outcome: pd.DataFrame):
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


if __name__ == '__main__':
    main()
