"""insights.py - writes findings about the data in plain English (no website code here)."""
import pandas as pd

from outliers import iqr_mask


def top_correlations(df, n=5):
    """The n strongest relationships between numeric columns, as (column A, column B, r)."""
    num = df.select_dtypes("number")
    pairs = []
    cols = list(num.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = num[a].corr(num[b])
            if pd.notna(r):
                pairs.append((a, b, float(r)))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs[:n]


def _strength(r):
    r = abs(r)
    if r >= 0.7:
        return "strong"
    if r >= 0.4:
        return "moderate"
    return "weak"


def generate_insights(df):
    """Returns a list of sentences describing the most important things in the data."""
    out = []

    # Strongest relationships
    for a, b, r in top_correlations(df, 3):
        if abs(r) >= 0.4:
            direction = "positive" if r > 0 else "negative"
            out.append(f"{a} and {b} show a {_strength(r)} {direction} correlation of {r:.2f}.")

    # Missing values
    for c in df.columns:
        pct = df[c].isna().mean() * 100
        if pct >= 5:
            out.append(f"{c} has {pct:.1f}% missing values.")

    # Lopsided numbers and outliers
    for c in df.select_dtypes("number").columns:
        s = df[c].dropna()
        if len(s) < 3:
            continue
        skew = s.skew()
        if pd.notna(skew) and abs(skew) > 1:
            side = "right (a few very large values)" if skew > 0 else "left (a few very small values)"
            out.append(f"{c} is skewed to the {side}; consider a log transform before modelling.")
        n_out = int(iqr_mask(s).sum())
        if n_out:
            out.append(f"{c} contains {n_out} outlier value(s) that may be anomalies "
                       f"(range {s.min():,.0f} to {s.max():,.0f}).")

    # Dominant categories
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1:
            share = df[c].value_counts(normalize=True)
            if share.iloc[0] >= 0.6:
                out.append(f"In {c}, the value '{share.index[0]}' makes up {share.iloc[0]:.0%} of all rows.")

    return out or ["No notable patterns were detected."]