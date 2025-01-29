import pandas as pd
from pathlib import Path
from verticox.datasets import get_prioritized_features

OUTPUT_FILE = "preprocessed_data.parquet"

def preprocess_data(df: pd.DataFrame, output_dir: str)-> tuple[pd.DataFrame, str] :
    """
    Takes the data and preprocesses it. Returned the preprocessed dataframe,
    as well as the file location of the preprocessed data.

    :param df:
    :return:
    """
    preprocessed = get_prioritized_features(df)

    preprocessed_file = Path(output_dir)/OUTPUT_FILE
    preprocessed.to_parquet(preprocessed_file)

    return preprocessed, str(preprocessed_file.absolute())
