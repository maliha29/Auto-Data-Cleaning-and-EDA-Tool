"""cleaning.py - everything that FIXES the data (no website code here)."""
import numpy as np
import pandas as pd


def make_sample():
    """Creates a small messy dataset for testing."""
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame({
        "age": rng.integers(18, 60, n).astype(float),
        "salary": rng.integers(20000, 90000, n).astype(float),
        "city": rng.choice(["Delhi", "Mumbai", "Pune", "Delhi "], n),
    })
    df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "city"] = None
    df.loc[3, "salary"] = 900000  # one extreme value (an outlier)
    df = pd.concat([df, df.head(8)], ignore_index=True)
    return df


def health(df):
    """Returns (missing cells, duplicate rows, simple quality score)."""
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    score = 100 - (missing / df.size * 100) * 2 - (duplicates / len(df) * 100) * 2
    return missing, duplicates, max(0, round(score, 1))


def remove_duplicates(df):
    return df.drop_duplicates()


def fill_missing_numbers(df):
    df = df.copy()
    for col in df.select_dtypes("number").columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def fill_missing_text(df):
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            mode = df[col].mode()
            if len(mode) > 0:
                df[col] = df[col].fillna(mode.iloc[0])
    return df


def trim_spaces(df):
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
    return df