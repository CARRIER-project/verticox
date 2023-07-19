import json

import numpy as np
from numpy.testing import assert_array_almost_equal
import vantage6.client as v6client
from clize import run
from sksurv.datasets import load_whas500
from sksurv.linear_model import CoxPHSurvivalAnalysis

from verticox.client import VerticoxClient

OUTCOME = 'event_happened'

OUTCOME_TIME_COLUMN = 'event_time'

FEATURE_COLUMNS = ['age', 'bmi', 'sysbp']

IMAGE = 'harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox'
DATABASE = 'parquet'
TIMEOUT = 20 * 60
PRECISION = 1e-3


def run_verticox_v6(host, port, user, password, *, private_key=None, tag='latest'):
    image = f'{IMAGE}:{tag}'

    client = v6client.Client(host, port)

    client.authenticate(user, password)
    client.setup_encryption(private_key)
    nodes = client.node.list(is_online=True)

    orgs = [n['id'] for n in nodes['data']]
    orgs = sorted(orgs)
    central_node = orgs[0]
    datanodes = orgs[1:]

    verticox_client = VerticoxClient(client, image=image)

    task = verticox_client.get_column_names(database=DATABASE)

    column_name_results = task.get_result()

    for r in column_name_results:
        print(f'organization: {r.organization}, columns: {r.content}')

    task = verticox_client.compute(FEATURE_COLUMNS, OUTCOME_TIME_COLUMN, OUTCOME,
                                   datanodes=datanodes, central_node=central_node,
                                   precision=PRECISION,
                                   database=DATABASE)

    results = task.get_result(timeout=TIMEOUT)
    results = results[0]

    local_coefs = run_local_analysis()

    print(f'Organization: {results.organization}')
    print(f'Log: {results.log}')
    print(f'Content: {json.dumps(results.content)}')
    federated_coefs = results.content[0]
    federated_coefs = [el[1] for el in federated_coefs]
    federated_coefs = np.array(federated_coefs)

    assert_array_almost_equal(local_coefs, federated_coefs, decimal=3)


def run_local_analysis():
    df, outcome = load_whas500()
    df = df[FEATURE_COLUMNS]

    model = CoxPHSurvivalAnalysis()
    model.fit(df, outcome)

    return model.coef_

    # Results should be close to [ 0.06169848, -0.00783813]

if __name__ == '__main__':
    run(run_verticox_v6)
