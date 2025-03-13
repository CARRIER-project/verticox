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

def test_preprocess_imputes_missing_numbers_median():
    df = pd.DataFrame({"age": [40, 50, 60, None],
                       "event_time": [1, 2, 3, 4], "event_happened": [True, False, True, True]})
    columns = Columns(["age"], "event_time", "event_happened")

    result, _ = preprocess_data(df, columns)

    assert result["age"].tolist() == [40, 50, 60, 50]

def test_preprocess_imputes_missing_categoricals_mode():
    df = pd.DataFrame({"animal": ["dog", "dog", "cat", None],
                       "event_time": [1, 2, 3, 4], "event_happened": [True, False, True, True]})
    columns = Columns(["animal"], "event_time", "event_happened")

    result, _ = preprocess_data(df, columns)

    # Categoricals are converted to dummies after imputation
    assert result["animal_cat"].tolist() == [0, 0, 1, 0]

def test_categorical_imputed_to_mode():
    df = pd.DataFrame({"animal": ["dog", "dog", "cat", None],
                       "event_time": [1, 2, 3, 4], "event_happened": [True, False, True, True]})
    df["animal"] = df["animal"].astype("category")

    columns = Columns(["animal"], "event_time", "event_happened")
    result, _ = preprocess_data(df, columns)

    assert result["animal_cat"].tolist() == [0, 0, 1, 0]