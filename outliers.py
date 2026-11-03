"""outliers.py - finds and handles extreme values (outliers)."""
import pandas as pd


def iqr_mask(s):
    """True for values that are far outside the 'normal middle' of the data (IQR method)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def zscore_mask(s, threshold=3):
    """True for values more than 3 'standard deviations' away from the average (Z-score method)."""
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=s.index)
    return ((s - s.mean()).abs() / std) > threshold


def outlier_summary(df):
    """A small table: how many outliers each numeric column has, by both methods."""
    rows = []
    for col in df.select_dtypes("number").columns:
        rows.append({
            "column": col,
            "IQR outliers": int(iqr_mask(df[col]).sum()),
            "Z-score outliers": int(zscore_mask(df[col]).sum()),
        })
    return pd.DataFrame(rows)


def cap_column(df, col):
    """Pulls extreme values back to the edge of the normal range (keeps all rows)."""
    df = df.copy()
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


def cap_all(df):
    for col in df.select_dtypes("number").columns:
        df = cap_column(df, col)
    return df


def remove_outlier_rows(df):
    """Deletes every row that has an IQR outlier in any numeric column."""
    bad = pd.Series(False, index=df.index)
    for col in df.select_dtypes("number").columns:
        bad = bad | iqr_mask(df[col])
    return df[~bad]