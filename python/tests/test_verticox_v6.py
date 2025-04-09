#! /usr/bin/env python3

import json
import vantage6.client as v6client
from clize import run

from test_constants import OUTCOME_TIME_COLUMN, OUTCOME, PRECISION
from verticox.client import VerticoxClient
from verticox.config import DOCKER_IMAGE

IMAGE = f"{DOCKER_IMAGE}:latest"
DATABASE = "default"
TIMEOUT = 20 * 60
TARGET_COEFS = {"age": 0.05566997593047372, "bmi": -0.0908968266847538}


def run_verticox_v6(host, port, user, password, *, private_key=None, image: str=IMAGE,
                    method="fit", precision: float = PRECISION):

    client = v6client.Client(host, port, log_level="warning")

    client.authenticate(user, password)
    client.setup_encryption(private_key)

    verticox_client = VerticoxClient(client, image=image)

    task = verticox_client.get_column_names(database=DATABASE)

    column_name_results = task.get_results()

    feature_orgs = []
    central_node = None
    feature_columns = []

    for r in column_name_results:
        run_id = r["run"]["id"]
        run = client.run.get(run_id)
        organization = run["organization"]["id"]
        columns = json.loads(r['result'])
        print(f"organization: {organization}, columns: {columns}")

        if {OUTCOME, OUTCOME_TIME_COLUMN} == set(columns):
            central_node = organization
        else:
            feature_orgs.append(organization)
            feature_columns += columns

    if central_node is None:
        raise Exception("Central node not found")

    # feature_columns = list(TARGET_COEFS.keys())

    print(f'Using feature columns: {feature_columns}')

    match method:
        case "fit":
            task = verticox_client.fit(
                feature_columns,
                OUTCOME_TIME_COLUMN,
                OUTCOME,
                feature_nodes=feature_orgs,
                outcome_node=central_node,
                precision=precision,
                database=DATABASE,
            )
        case "crossval":
            task = verticox_client.cross_validate(
                feature_columns,
                OUTCOME_TIME_COLUMN,
                OUTCOME,
                feature_nodes=feature_orgs,
                outcome_node=central_node,
                precision=precision,
                database=DATABASE,
            )

    results = task.get_results()

    print("Results: ", results)


if __name__ == "__main__":
    run(run_verticox_v6)
