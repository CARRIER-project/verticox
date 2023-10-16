import json
import numpy as np
import vantage6.client as v6client
from clize import run

from test_constants import OUTCOME_TIME_COLUMN, OUTCOME, PRECISION
from verticox.client import VerticoxClient

IMAGE = "harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox"
DATABASE = "parquet"
TIMEOUT = 20 * 60
TARGET_COEFS = {"age": 0.05566997593047372, "bmi": -0.0908968266847538}


def run_verticox_v6(host, port, user, password, *, private_key=None, tag="latest", method="fit"):
    image = f"{IMAGE}:{tag}"

    client = v6client.Client(host, port, log_level="warning")

    client.authenticate(user, password)
    client.setup_encryption(private_key)
    nodes = client.node.list(is_online=True)

    orgs = [n["id"] for n in nodes["data"]]
    orgs = sorted(orgs)
    central_node = orgs[0]
    datanodes = orgs[1:]

    verticox_client = VerticoxClient(client, image=image)

    task = verticox_client.get_column_names(database=DATABASE)

    column_name_results = task.get_result()

    for r in column_name_results:
        print(f"organization: {r.organization}, columns: {r.content}")

    feature_columns = list(TARGET_COEFS.keys())

    print(f'Using feature columns: {feature_columns}')

    match method:
        case "fit":
            task = verticox_client.fit(
                feature_columns,
                OUTCOME_TIME_COLUMN,
                OUTCOME,
                datanodes=datanodes,
                central_node=central_node,
                precision=PRECISION,
                database=DATABASE,
            )
        case "crossval":
            task = verticox_client.cross_validate(
                feature_columns,
                OUTCOME_TIME_COLUMN,
                OUTCOME,
                datanodes=datanodes,
                central_node=central_node,
                precision=PRECISION,
                database=DATABASE,
            )

    results = task.get_result(timeout=TIMEOUT)
    for result in results:
        print(f"Organization: {result.organization}")
        print(f"Log: {result.log}")
        print(f"Content: {json.dumps(result.content)}")

    # Results should be close to [ 0.06169848, -0.00783813]
    coefs = dict(results[0].content[0])

    for key, value in coefs.items():
        np.testing.assert_almost_equal(value, TARGET_COEFS[key], decimal=4)


if __name__ == "__main__":
    run(run_verticox_v6)
