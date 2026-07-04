"""
src/preprocessing.py
---------------------
Data loading and cleaning utilities for the phishing detection pipeline.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame | None:
    """
    Load a CSV dataset.

    Expects at minimum a 'url' column and (for training) a 'label' column
    where 1 = phishing and 0 = legitimate.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame, or None on failure.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns  ({filepath})")
        return df
    except Exception as exc:
        print(f"[Error] Could not load '{filepath}': {exc}")
        return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw URL dataset.

    Steps
    -----
    1. Drop rows with missing 'url' values.
    2. Remove exact-duplicate URLs (keep first occurrence).
    3. Cast 'url' to str.
    4. Cast 'label' to int if the column is present.
    5. Reset the index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with at least a 'url' column.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.dropna(subset=["url"])
    df = df.drop_duplicates(subset=["url"])
    df["url"] = df["url"].astype(str)

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    df = df.reset_index(drop=True)
    print(f"Data cleaned  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/urls.csv"
    data = load_data(path)
    if data is not None:
        data = clean_data(data)
        print(data.head())
