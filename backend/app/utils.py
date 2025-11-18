import os
import pandas as pd

def safe_parquet_load(path: str):
    """Load parquet safely with real validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Path exists but is not a file: {path}")

    if os.path.getsize(path) == 0:
        raise ValueError(f"File exists but is EMPTY: {path}")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise ValueError(f"Failed to load parquet: {path}. Error: {e}")

    if df.empty:
        raise ValueError(f"Loaded EMPTY dataframe from {path}")

    return df
