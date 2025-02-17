import pandas as pd
from verticox.preprocess import preprocess_data

def test_preprocess_makes_dummies():
    df = pd.DataFrame({"species": ["cat", "dog", "human"]})

    result = preprocess_data(df)

    # One dummy category is always removed
    assert result.columns.tolist() == ["species_cat", "species_dog"]