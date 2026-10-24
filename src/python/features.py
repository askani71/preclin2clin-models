"""Feature engineering and scaling."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize the given columns; returns the transformed frame and the fitted scaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
