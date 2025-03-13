from typing import NamedTuple
from vantage6.algorithm.tools.util import info

import pandas as pd
from pathlib import Path
from verticox.datasets import get_prioritized_features

OUTPUT_FILE = "preprocessed_data.parquet"

Columns = NamedTuple("Columns", [("feature_columns", list[str]),
                                 ("event_times_column", str),
                                 ("event_happened_column", str)])
def preprocess_data(df: pd.DataFrame,  columns: [Columns, list[str]], output_dir: str|None = None)\
        -> tuple[pd.DataFrame,Columns, str]|tuple[pd.DataFrame, Columns]:
    """
    Takes the data and preprocesses it. Returned the preprocessed dataframe,
    as well as the file location of the preprocessed data.

    Preprocessing involves:
    - Converting categorical data to dummies

    :param df: The data to be preprocessed.
    :param columns: The columns used for data analysis. Might not all be present in this
    particular dataset.
    :param output_dir: The directory where the preprocessed data will be stored.
    :return: The preprocessed data as a dataframe, the new column names as Columns object,
    additionally, the file location of the preprocessed data if output_dir was provided
    """
    preprocessed = get_prioritized_features(df)
    preprocessed_names = list(preprocessed.columns)
    new_features = []

    info(f"Columns: {columns}")

    match columns:
        case Columns(feature_columns, event_times_column, event_happened_column):
            feature_columns = feature_columns
            event_times_column = event_times_column
            event_happened_column = event_happened_column
        case _:
            # Must be a list
            feature_columns = columns
            event_happened_column = None
            event_times_column = None

    info(f"Preprocessed names: {preprocessed_names}")

    for feature in feature_columns:
        new_names = [name for name in preprocessed_names if name.startswith(feature)]

        # If column is not present in the data, we still want to keep the old value in feature names
        # This is because the columns represent all columns present in the federated dataset,
        # not just the columns present in the current datanode.
        if new_names == []:
            new_names = [feature]
        new_features += new_names

    # Check outcome as well
    if event_happened_column is not None:
        new_event_happened = [name for name in preprocessed_names if name.startswith(event_happened_column)]
        if new_event_happened == []:
            new_event_happened = event_happened_column
        else:
            new_event_happened = new_event_happened[0]
    else:
        new_event_happened = None

    new_columns = Columns(new_features, event_times_column, new_event_happened)

    if output_dir is  None:
        return preprocessed, new_columns
    else:
        preprocessed_file = Path(output_dir)/OUTPUT_FILE
        preprocessed.to_parquet(preprocessed_file)

        return preprocessed, new_columns, str(preprocessed_file.absolute())
