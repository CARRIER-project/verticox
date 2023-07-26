from verticox.common import get_test_dataset
import pandas as pd
from pathlib import Path
from test_constants import FEATURE_COLUMNS

ROWS = 50
OUTPUT_DIR = Path('../mock/data/')
FILENAMES = ['data_1', 'data_2']
OUTCOME_FILENAME = 'outcome'
OUTCOME_COLUMN_NAMES = ['event_time', 'event_happened']


def main():
    data, outcome, column_names = get_test_dataset(limit=ROWS)

    df = pd.DataFrame(data, columns=column_names)
    df = df[FEATURE_COLUMNS]
    print(df)

    dataset_1 = df[FEATURE_COLUMNS[:-1]]
    dataset_2 = df[FEATURE_COLUMNS[-1:]]

    datasets = [dataset_1, dataset_2]

    for subset, name in zip(datasets, FILENAMES):
        parquet_path = OUTPUT_DIR / f'{name}.parquet'
        subset.to_parquet(parquet_path.absolute(), index=False)
        csv_path = OUTPUT_DIR / f'{name}.csv'
        subset.to_csv(csv_path.absolute(), index=False)

    event_happened, event_time = zip(*outcome)

    outcome_df = pd.DataFrame({'event_time': event_time, 'event_happened': event_happened})

    print(outcome_df)
    outcome_df.to_parquet(OUTPUT_DIR / 'outcome.parquet', index=False)
    outcome_df.to_csv(OUTPUT_DIR / 'outcome.csv', index=False)


if __name__ == '__main__':
    main()
