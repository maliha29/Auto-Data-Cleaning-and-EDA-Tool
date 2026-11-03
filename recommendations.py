"""recommendations.py - the 'smart advisor'. Looks at the data and explains what to fix and why."""
import pandas as pd

from outliers import iqr_mask, cap_column


def get_recommendations(df):
    """Returns a list of advice cards. Each card says WHAT, WHY and how CONFIDENT we are."""
    recs = []

    # 1. Duplicate rows
    dups = int(df.duplicated().sum())
    if dups:
        recs.append({
            "title": f"Remove {dups} duplicate rows",
            "why": "Exact copies of rows add no information and can bias your results.",
            "confidence": 99, "action": "remove_duplicates", "column": None,
        })

    for col in df.columns:
        s = df[col]
        pct = s.isna().mean() * 100

        # 2. Missing values
        if 0 < pct <= 60:
            if pd.api.types.is_numeric_dtype(s):
                skew = s.skew()
                if abs(skew) > 1:
                    recs.append({
                        "title": f"'{col}' has {pct:.1f}% missing values: use MEDIAN",
                        "why": f"The data is lopsided (skew = {skew:.2f}), so the median is safer than the mean.",
                        "confidence": 92, "action": "fill_median", "column": col,
                    })
                else:
                    recs.append({
                        "title": f"'{col}' has {pct:.1f}% missing values: use MEAN",
                        "why": f"The data is fairly balanced (skew = {skew:.2f}), so the average is a good fill value.",
                        "confidence": 85, "action": "fill_mean", "column": col,
                    })
            else:
                recs.append({
                    "title": f"'{col}' has {pct:.1f}% missing values: use MODE",
                    "why": "This is a text column, so the most common value is the best guess.",
                    "confidence": 80, "action": "fill_mode", "column": col,
                })
        elif pct > 60:
            recs.append({
                "title": f"'{col}' is {pct:.0f}% empty: drop the column",
                "why": "Too little real data is left to fill it in reliably.",
                "confidence": 75, "action": "drop_column", "column": col,
            })

        # 3. Extra spaces in text
        if not pd.api.types.is_numeric_dtype(s):
            t = s.dropna().astype(str)
            spaces = int((t != t.str.strip()).sum())
            if spaces:
                recs.append({
                    "title": f"'{col}' has {spaces} values with extra spaces",
                    "why": "'Delhi' and 'Delhi ' look the same to you, but the computer sees two different values.",
                    "confidence": 98, "action": "trim", "column": col,
                })

        # 4. Outliers
        elif s.nunique() > 10:
            out = int(iqr_mask(s).sum())
            if out:
                out_pct = out / len(s) * 100
                recs.append({
                    "title": f"'{col}' has {out} outlier(s) ({out_pct:.1f}%): cap them",
                    "why": "A few extreme values can distort averages and charts. Capping keeps the rows.",
                    "confidence": 90 if out_pct < 5 else 75, "action": "cap_outliers", "column": col,
                })

    return recs


def apply_recommendation(df, rec):
    """Carries out one piece of advice and returns the new table."""
    action, col = rec["action"], rec["column"]
    df = df.copy()

    if action == "remove_duplicates":
        return df.drop_duplicates()
    if action == "fill_median":
        df[col] = df[col].fillna(df[col].median())
    elif action == "fill_mean":
        df[col] = df[col].fillna(df[col].mean())
    elif action == "fill_mode":
        mode = df[col].mode()
        if len(mode) > 0:
            df[col] = df[col].fillna(mode.iloc[0])
    elif action == "trim":
        df[col] = df[col].astype("string").str.strip()
    elif action == "cap_outliers":
        df = cap_column(df, col)
    elif action == "drop_column":
        df = df.drop(columns=col)
    return df