import logging
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Tuple

import clize
import numpy as np
import pandas as pd
from numpy import vectorize
from sklearn.model_selection import KFold
from sksurv.datasets import get_x_y
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from test_constants import CONVERGENCE_PRECISION
from verticox.common import unpack_events
from verticox.cross_validation import kfold_cross_validate
from verticox.node_manager import LocalNodeManager

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

TEST_DATA_PATH = "mock/data"
COVARIATE_FILES = ["data_1.parquet", "data_2.parquet"]
OUTCOME_FILE = "outcome.parquet"
DECIMAL_PRECISION = 3
SELECTED_TARGET_COEFS = {"bmi": -0.15316136, "age": 0.05067197}

NUM_SELECTED_ROWS = 20


def compute_central_coefs(all_data_features, all_data_outcome):
    central_model = CoxPHSurvivalAnalysis()
    central_model.fit(all_data_features, all_data_outcome)

    coef_dict = dict(zip(all_data_features.columns, central_model.coef_))
    return coef_dict


class IntegrationTest(ABC):

    def run(self, local_data, all_data, event_times_column, event_happened_column):
        """
        Run an integration test
        Args:
            local_data: the outcome data file to pass to the aggregator
            all_data: the directory that contains all the data, for validation
            event_times_column: the column name of outcome event times
            event_happened_column: the column name of whether the outcome event has happened
            benchmark: If True, run time will be returned as additional value. Default = False

        Returns:

        """
        all_data_features, all_data_outcome, node_manager = prepare_test(all_data,
                                                                         event_happened_column,
                                                                         event_times_column,
                                                                         local_data)

        start_time = datetime.now()
        results = self.run_integration_test(all_data_features, all_data_outcome, node_manager)
        end_time = datetime.now()
        runtime = end_time - start_time

        print(f"Runtime: {runtime.total_seconds()}")

        return runtime.total_seconds()

    @staticmethod
    def run_integration_test(all_data_features, all_data_outcome, node_manager):
        pass


class OnlyTrain(IntegrationTest):
    """
    Train the model on the full dataset. Compare resulting coefficients to a central model.
    """

    @staticmethod
    def run_integration_test(all_data_features, all_data_outcome, node_manager):
        _logger.info(
            "\n\n----------------------------------------\n"
            "       Starting test on full dataset..."
            "\n----------------------------------------"
        )
        node_manager.reset()
        node_manager.fit()
        coefs = node_manager.coefs
        print(f"Betas: {coefs}")
        print(f"Baseline hazard ratio {node_manager.baseline_hazard}")

        target_coefs = compute_central_coefs(all_data_features, all_data_outcome)

        for key, value in target_coefs.items():
            np.testing.assert_almost_equal(value, coefs[key], decimal=DECIMAL_PRECISION)


class TrainTest(IntegrationTest):
    @staticmethod
    def run_integration_test(all_data_features, all_data_outcome, node_manager):
        """
        Split the data in a training and test set. Train on the training set, test performance on the
        test set.
        Args:
            all_data_features:
            all_data_outcome:
            node_manager:

        Returns:

        """

        _logger.info(
            "\n\n----------------------------------------\n"
            "          Starting test on selection..."
            "\n----------------------------------------"
        )
        full_data_length = node_manager.num_total_records
        selected_idx = select_rows(full_data_length)
        mask = np.zeros(full_data_length, dtype=bool)
        mask[selected_idx] = True

        # TODO: This flow is not ideal
        node_manager.reset(selected_idx)
        node_manager.fit()
        coefs = node_manager.coefs

        _logger.info(f"Betas: {coefs}")
        _logger.info(f"Baseline hazard ratio {node_manager.baseline_hazard}")
        for key, value in SELECTED_TARGET_COEFS.items():
            np.testing.assert_almost_equal(value, coefs[key], decimal=DECIMAL_PRECISION)

        c_index = node_manager.test()

        all_data_features_train = all_data_features.iloc[mask]
        all_data_outcome_train = all_data_outcome[mask]

        all_data_features_test = all_data_features.iloc[~mask]
        all_data_outcome_test = all_data_outcome[~mask]
        print(f'Number of test samples: {all_data_features_test.shape[0]}')
        event_time, event_indicator = unpack_events(all_data_outcome_test)

        central_model = CoxPHSurvivalAnalysis()
        central_model.fit(all_data_features_train, all_data_outcome_train)

        central_predictions = central_model.predict(all_data_features_test)
        central_c_index, _, _, _, _ = concordance_index_censored(event_indicator, event_time,
                                                                 central_predictions)

        np.testing.assert_almost_equal(c_index, central_c_index, decimal=DECIMAL_PRECISION)


class CrossValidation(IntegrationTest):
    """
    Performs the crossvalidation integration test.
    """

    @staticmethod
    def run_integration_test(all_data_features, all_data_outcome, node_manager):
        n_splits = 5
        random_state = 0
        # In the dataset we are using the uncensored data is at the beginning and the censored data
        # at the end. We need to mix it up
        shuffle = True

        central_c_indices = cross_validate_central(all_data_features, all_data_outcome, n_splits,
                                                   random_state, shuffle)
        _logger.info(
            "\n\n--------------------------------------------\n"
            "      Starting test with cross validation..."
            "\n--------------------------------------------"
        )
        c_indices, coefs, baseline_hazards = kfold_cross_validate(node_manager, n_splits,
                                                                  random_state,
                                                                  shuffle)
        print("Cross validation done")
        print(f"C scores: {c_indices}")
        print(f"Baseline hazards: {baseline_hazards}")
        print(f"Coefs: {coefs}")

        # Compare against central version

        np.testing.assert_almost_equal(c_indices, central_c_indices, decimal=DECIMAL_PRECISION)


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


def collect_all_test_data(data_path):
    data_path = Path(data_path)

    dfs = []
    for f in data_path.iterdir():
        dfs.append(pd.read_parquet(f))

    return pd.concat(dfs, axis=1)


@vectorize
def outcome_lower_equal_than_x(outcome, x):
    return outcome[1] <= x


def cross_validate_central(all_data_features, all_data_outcome, n_splits, random_state, shuffle):
    kfold = KFold(n_splits, random_state=random_state, shuffle=shuffle)
    folds = kfold.split(all_data_outcome)
    central_c_indices = []
    for idx, (train_indices, test_indices) in enumerate(folds):
        _logger.info(f'Fold {idx}')
        # Select data
        train_features = all_data_features.iloc[train_indices]
        train_outcome = all_data_outcome[train_indices]
        # Train model
        model = CoxPHSurvivalAnalysis()
        model.fit(train_features, train_outcome)
        # Evaluate model
        test_features = all_data_features.iloc[test_indices]
        test_outcome = all_data_outcome[test_indices]

        estimates = model.predict(test_features)

        event_time, event_indicator = unpack_events(test_outcome)
        c_index, _, _, _, _ = concordance_index_censored(event_indicator, event_time, estimates)
        central_c_indices.append(c_index)
    return central_c_indices


def run_all(local_data, all_data, event_times_column, event_happened_column):
    """
    Run all integration tests:
        - Training on the full dataset
        - Training on one split, testing on the other
        - Performing crossvalidation
    Args:
        local_data:
        all_data:
        event_times_column:
        event_happened_column:

    Returns:

    """
    OnlyTrain().run(local_data, all_data, event_times_column, event_happened_column)
    TrainTest().run(local_data, all_data, event_times_column, event_happened_column)
    CrossValidation().run(local_data, all_data, event_times_column, event_happened_column)
    print("Test has passed.")


def prepare_test(all_data, event_happened_column, event_times_column, local_data) \
        -> Tuple[pd.DataFrame, np.array, LocalNodeManager]:
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
    return all_data_features, all_data_outcome, node_manager


if __name__ == "__main__":
    subcommands = {"all": run_all, "train": OnlyTrain().run, "split": TrainTest.run,
                   "crossval": CrossValidation.run}
    clize.run(subcommands)
