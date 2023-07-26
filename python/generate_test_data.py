from verticox.common import get_test_dataset
import pandas as pd
from pathlib import Path

ROWS = 50
NUM_COLUMNS = 3
OUTPUT_DIR = Path('../mock/data/')
FILENAMES = ['data_1', 'data_2']
OUTCOME_FILENAME = 'outcome'
OUTCOME_COLUMN_NAMES = ['event_time', 'event_happened']


def main():
    data, outcome, column_names = get_test_dataset(limit=ROWS, feature_limit=3)

    df = pd.DataFrame(data, columns=column_names)

    dataset_1 = df[column_names[:-1]]
    dataset_2 = df[column_names[-1:]]

    datasets = [dataset_1, dataset_2]

    for subset, name in zip(datasets, FILENAMES):
        file_path = OUTPUT_DIR / f'{name}.parquet'
        subset.to_parquet(file_path.absolute())

    event_happened, event_time = zip(*outcome)

    outcome_df = pd.DataFrame({'event_time': event_time, 'event_happened': event_happened})

    outcome_df.to_parquet(OUTPUT_DIR/'outcome.parquet')


if __name__ == '__main__':
    main()

