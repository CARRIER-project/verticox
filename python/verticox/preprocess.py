import pandas as pd
from pathlib import Path
from verticox.datasets import get_prioritized_features

OUTPUT_FILE = "preprocessed_data.parquet"

def preprocess_data(df: pd.DataFrame, output_dir: str|None = None)-> tuple[pd.DataFrame, str] :
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

    if output_dir is  None:
        return preprocessed
    else:
        preprocessed_file = Path(output_dir)/OUTPUT_FILE
        preprocessed.to_parquet(preprocessed_file)

        return preprocessed, str(preprocessed_file.absolute())
