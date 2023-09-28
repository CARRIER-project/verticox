import logging
from pathlib import Path

import clize
import numpy as np
import pandas as pd
from numpy import vectorize
from sksurv.datasets import get_x_y
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from vantage6.common import info

from common import compare_stepfunctions
from test_constants import CONVERGENCE_PRECISION
from verticox.common import unpack_events
from verticox.node_manager import LocalNodeManager

TEST_DATA_PATH = "mock/data"
COVARIATE_FILES = ["data_1.parquet", "data_2.parquet"]
OUTCOME_FILE = "outcome.parquet"
DECIMAL_PRECISION = 4
TARGET_COEFS = {"age": 0.05566997593047372, "bmi": -0.0908968266847538}
SELECTED_TARGET_COEFS = {"bmi": -0.15316136, "age": 0.05067197}

NUM_SELECTED_ROWS = 20


def select_rows(data_length, num_rows=NUM_SELECTED_ROWS):
    """
    Select a subset of data so that it includes both censored and non-censored data. Assuming the
    censored data follows after the non-censored data.
    Args:
        data:

    Returns:
    """
    if num_rows > data_length:
        raise Exception(
            f"Selecting too many rows. There are only {data_length} available."
        )

    num_start = num_rows // 2
    num_end = num_rows - num_start

    censored_selection = range(num_start)
    uncensored_selection = range(data_length - num_end, data_length)

    return np.array(list(censored_selection) + list(uncensored_selection))


def compute_centralized():
    all_tables = []
    for f in COVARIATE_FILES:
        df = pd.read_parquet(Path(TEST_DATA_PATH) / f)
        all_tables.append(df)

    full_covariates = pd.concat(all_tables, axis=1)

    outcome = pd.read_parquet(Path(TEST_DATA_PATH) / OUTCOME_FILE)
    outcome_structured = Surv.from_dataframe("event_happened", "event_time", outcome)

    model = CoxPHSurvivalAnalysis()

    model.fit(full_covariates, outcome_structured)

    # Turn results into dict so that we know which covariates belong to which column
    return dict(zip(full_covariates.columns, model.coef_))


def run_test_full_dataset(
        node_manager: LocalNodeManager, all_data_features, all_data_outcome
):
    node_manager.reset()
    node_manager.fit()
    coefs = node_manager.betas
    logging.info(f"Betas: {coefs}")
    logging.info(f"Baseline hazard ratio {node_manager.baseline_hazard}")

    for key, value in TARGET_COEFS.items():
        np.testing.assert_almost_equal(value, coefs[key], decimal=DECIMAL_PRECISION)


def collect_all_test_data(data_path):
    data_path = Path(data_path)

    dfs = []
    for f in data_path.iterdir():
        dfs.append(pd.read_parquet(f))

    return pd.concat(dfs, axis=1)


def run_test_selection(
        node_manager: LocalNodeManager,
        full_data_length,
        all_data_features,
        all_data_outcome,
):
    selected_idx = select_rows(full_data_length)

    # TODO: This flow is not ideal
    node_manager.reset(selected_idx)
    node_manager.fit()
    coefs = node_manager.betas

    logging.info(f"Betas: {coefs}")
    logging.info(f"Baseline hazard ratio {node_manager.baseline_hazard}")
    for key, value in SELECTED_TARGET_COEFS.items():
        np.testing.assert_almost_equal(value, coefs[key], decimal=DECIMAL_PRECISION)

    auc, cum_survival = node_manager.test()

    all_data_features_train = all_data_features.iloc[selected_idx]
    all_data_outcome_train = all_data_outcome[selected_idx]

    all_data_features_test = all_data_features.iloc[~selected_idx]
    all_data_outcome_test = all_data_outcome[~selected_idx]

    central_model = CoxPHSurvivalAnalysis()
    central_model.fit(all_data_features_train, all_data_outcome_train)

    # Because of how auc is computed all test times need to be within the range of the training set
    within_range = outcome_lower_equal_than_x(
        all_data_outcome_test, np.max(node_manager.baseline_hazard.x)
    )

    all_data_features_test = all_data_features_test.iloc[within_range]
    all_data_outcome_test = all_data_outcome_test[within_range]

    central_predictions = central_model.predict(all_data_features_test)

    times = node_manager.baseline_hazard.x

    info(f"AUC: {auc}")

    #compare_stepfunctions(auc, target_auc)


@vectorize
def outcome_lower_equal_than_x(outcome, x):
    return outcome[1] <= x


def run_locally(local_data, all_data, event_times_column, event_happened_column):
    df = pd.read_parquet(local_data)
    all_data_df = collect_all_test_data(all_data)

    all_data_features, all_data_outcome = get_x_y(
        all_data_df, (event_happened_column, event_times_column), pos_label=True
    )

    node_manager = LocalNodeManager(
        df,
        event_times_column,
        event_happened_column,
        {"convergence_precision": CONVERGENCE_PRECISION},
    )

    node_manager.start_nodes()

    run_test_full_dataset(node_manager, all_data_features, all_data_outcome)
    run_test_selection(node_manager, df.shape[0], all_data_features, all_data_outcome)

    print("Test has passed.")


if __name__ == "__main__":
    clize.run(run_locally)
