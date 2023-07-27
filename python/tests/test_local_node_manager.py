import logging
from pathlib import Path
import clize
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from test_constants import CONVERGENCE_PRECISION
from verticox.node_manager import LocalNodeManager
import numpy as np

TEST_DATA_PATH = '../mock/data'
COVARIATE_FILES = ['data_1.parquet', 'data_2.parquet']
OUTCOME_FILE = 'outcome.parquet'
DECIMAL_PRECISION = 4
TARGET_COEFS = {'bmi': -0.16787336343511397, 'sysbp': -0.015998133958014818}
logger = logging.getLogger(__name__)


def compute_centralized():
    all_tables = []
    for f in COVARIATE_FILES:
        df = pd.read_parquet(Path(TEST_DATA_PATH) / f)
        all_tables.append(df)

    full_covariates = pd.concat(all_tables, axis=1)

    outcome = pd.read_parquet(Path(TEST_DATA_PATH) / OUTCOME_FILE)
    outcome_structured = Surv.from_dataframe('event_happened', 'event_time', outcome)

    model = CoxPHSurvivalAnalysis()

    model.fit(full_covariates, outcome_structured)

    # Turn results into dict so that we know which covariates belong to which column
    return dict(zip(full_covariates.columns, model.coef_))


def run_locally(data, event_times_column, event_happened_column):
    df = pd.read_parquet(data)

    node_manager = LocalNodeManager(df, event_times_column, event_happened_column,
                                    {'convergence_precision': CONVERGENCE_PRECISION})
    node_manager.start_nodes()
    node_manager.fit()

    coefs = node_manager.betas
    logging.info(f'Betas: {coefs}')
    logging.info(f'Baseline hazard ratio {node_manager.baseline_hazard}')

    for key, value in TARGET_COEFS.items():
        np.testing.assert_almost_equal(value, coefs[key], decimal=DECIMAL_PRECISION)

    assert False
    print('Test has passed.')


if __name__ == '__main__':
    clize.run(run_locally)
