"""Pipeline to load and clean raw preclinical inputs."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw(filename: str) -> pd.DataFrame:
    """Load a raw CSV file from ``data/raw``."""
    return pd.read_csv(RAW_DIR / filename)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: normalize column names and drop duplicate rows."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df.drop_duplicates().reset_index(drop=True)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save a dataset to ``data/processed`` in Parquet format."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_parquet(path, index=False)
    return path
