from typing import NamedTuple

import pandas as pd
from pathlib import Path
from verticox.datasets import get_prioritized_features

OUTPUT_FILE = "preprocessed_data.parquet"

Columns = NamedTuple("Columns", [("feature_columns", list[str]),
                                 ("event_times_column", str),
                                 ("event_happened_column", str)])
def preprocess_data(df: pd.DataFrame,  columns: Columns, output_dir: str|None = None)\
        -> tuple[ pd.DataFrame,Columns, str]:
    """
    Takes the data and preprocesses it. Returned the preprocessed dataframe,
    as well as the file location of the preprocessed data.

    Preprocessing involves:
    - Converting categorical data to dummies

    :param df: The data to be preprocessed.
    :param output_dir: The directory where the preprocessed data will be stored.
    :return: The preprocessed data as a dataframe and the file location of the preprocessed data.
    """
    preprocessed = get_prioritized_features(df)
    new_names = list(preprocessed.columns)
    new_features = []
    for feature in columns.feature_columns:
        new_features += [name for name in new_names if name.startswith(feature)]

    # Check outcome as well
    new_event_happened = [name for name in new_names if name.startswith(columns.event_happened_column)]

    new_columns = Columns(new_features, columns.event_times_column, new_event_happened[0])


    if output_dir is  None:
        return preprocessed, new_columns
    else:
        preprocessed_file = Path(output_dir)/OUTPUT_FILE
        preprocessed.to_parquet(preprocessed_file)

        return preprocessed, new_columns, str(preprocessed_file.absolute())
