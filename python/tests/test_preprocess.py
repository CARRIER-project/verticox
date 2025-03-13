import pandas as pd
from verticox.preprocess import preprocess_data, Columns

SAMPLE_DF = pd.DataFrame({"species": ["cat", "dog", "human"],
                       "event_time": [1, 2, 3], "event_happened": [True, False, True]})

def test_preprocess_makes_dummies():
    columns = Columns(["species"], "event_time", "event_happened")
    result, new_columns = preprocess_data(SAMPLE_DF, columns)

    # One dummy category is always removed
    assert set(result.columns.tolist()) == {"species_cat", "species_dog", "event_time",
                                        "event_happened"}
    assert set(new_columns.feature_columns) == {"species_cat", "species_dog"}
    assert new_columns.event_times_column == "event_time"
    assert new_columns.event_happened_column == "event_happened"

def test_preprocess_keeps_absent_columns():
    columns = Columns(["species", "sex"], "event_time", "event_happened")

    result, new_columns = preprocess_data(SAMPLE_DF, columns)

    assert set(new_columns.feature_columns) == {"species_cat", "species_dog", "sex"}