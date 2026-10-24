import pandas as pd
from data_loader import clean


def test_clean_normalizes_columns_and_drops_duplicates():
    df = pd.DataFrame({"Dose mg": [1, 1, 2], " Response ": [0.1, 0.1, 0.5]})
    out = clean(df)
    assert list(out.columns) == ["dose_mg", "response"]
    assert len(out) == 2
