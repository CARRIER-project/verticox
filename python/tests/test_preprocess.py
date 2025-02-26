import pandas as pd
from verticox.preprocess import preprocess_data, Columns


def test_preprocess_makes_dummies():
    df = pd.DataFrame({"species": ["cat", "dog", "human"],
                       "event_time": [1, 2, 3], "event_happened": [True, False, True]})

    columns = Columns(["species"], "event_time", "event_happened")

    result, new_columns = preprocess_data(df, columns)

    # One dummy category is always removed
    assert set(result.columns.tolist()) == {"species_cat", "species_dog", "event_time",
                                        "event_happened"}
    assert set(new_columns.feature_columns) == {"species_cat", "species_dog"}
    assert new_columns.event_times_column == "event_time"
    assert new_columns.event_happened_column == "event_happened"
