"""ml_readiness.py - checks whether the data is ready for machine learning."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def suggest_targets(df):
    """Columns that could be 'the thing to predict'. Returns a list of (column, type)."""
    found = []
    for c in df.columns:
        n = df[c].nunique()
        if n < 2 or n / max(len(df), 1) > 0.95:  # constant or ID-like columns are skipped
            continue
        if n <= 10:
            found.append((c, "classification"))
        elif pd.api.types.is_numeric_dtype(df[c]):
            found.append((c, "regression"))
    return found


def class_balance(df, target):
    """Returns (counts table, is_imbalanced). Imbalanced = biggest class is 3x+ the smallest."""
    counts = df[target].value_counts()
    ratio = counts.max() / counts.min()
    return counts, ratio > 3


def ml_checks(df, target=None):
    """Runs all checks. Returns (score, list of (ok?, title, advice))."""
    checks = []
    score = 100

    # 1. Missing values left
    missing_pct = df.isna().mean().mean() * 100
    if missing_pct > 0:
        score -= min(40, missing_pct * 3)
        checks.append((False, f"{missing_pct:.1f}% of cells are still empty",
                       "Most ML models cannot handle empty cells. Fill or drop them first."))
    else:
        checks.append((True, "No missing values left", "Models can read every cell."))

    # 2. Duplicates
    dup_pct = df.duplicated().mean() * 100
    if dup_pct > 0:
        score -= min(10, dup_pct * 2)
        checks.append((False, f"{dup_pct:.1f}% duplicate rows",
                       "Duplicates can leak into the test set and give fake accuracy."))
    else:
        checks.append((True, "No duplicate rows", "Good."))

    # 3. Encoding (text columns must become numbers)
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != target]
    if text_cols:
        score -= min(20, len(text_cols) * 5)
        checks.append((False, f"{len(text_cols)} text column(s) need encoding: {', '.join(map(str, text_cols))}",
                       "Use One-Hot Encoding (few categories) or Label Encoding (many categories)."))
    else:
        checks.append((True, "No text columns to encode", "All features are already numbers."))

    # 4. Scaling (numeric columns on very different scales)
    num = df.select_dtypes("number")
    if num.shape[1] >= 2:
        stds = num.std().replace(0, np.nan).dropna()
        if len(stds) >= 2 and stds.max() / stds.min() > 10:
            score -= 10
            checks.append((False, "Numeric columns are on very different scales",
                           f"'{stds.idxmax()}' varies far more than '{stds.idxmin()}'. "
                           "Use StandardScaler or MinMaxScaler for distance-based models (KNN, SVM)."))
        else:
            checks.append((True, "Numeric scales look similar", "Scaling is not urgent."))

    # 5. Enough rows
    if len(df) < 100:
        score -= 10
        checks.append((False, f"Only {len(df)} rows",
                       "Small datasets make models unreliable. Aim for several hundred rows or more."))
    else:
        checks.append((True, f"{len(df):,} rows", "A reasonable amount of data."))

    # 6. Class imbalance (only if a target was chosen)
    if target and df[target].nunique() <= 10:
        _, imbalanced = class_balance(df, target)
        if imbalanced:
            score -= 10
            checks.append((False, f"Target '{target}' is imbalanced",
                           "Try class weights, oversampling (SMOTE), and use F1-score instead of accuracy."))
        else:
            checks.append((True, f"Target '{target}' is balanced", "Classes have similar sizes."))

    return max(0, round(score)), checks


def show_ml_readiness(df):
    st.header("🤖 Machine Learning Readiness")

    targets = suggest_targets(df)
    options = ["(none)"] + [c for c, _ in targets]
    if targets:
        st.caption("Suggested targets: " + ", ".join(f"**{c}** ({t})" for c, t in targets))
    choice = st.selectbox("Choose a target column (what you want to predict)", options)
    target = None if choice == "(none)" else choice

    score, checks = ml_checks(df, target)

    left, right = st.columns([1, 2])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": "ML Readiness Score"},
        gauge={"axis": {"range": [0, 100]}},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    left.plotly_chart(fig, key="ml_gauge")

    with right:
        for ok, title, advice in checks:
            if ok:
                st.success(f"✅ {title}")
            else:
                st.warning(f"⚠️ {title}\n\n{advice}")

    if target and df[target].nunique() <= 10:
        counts, _ = class_balance(df, target)
        data = counts.reset_index()
        data.columns = [target, "count"]
        st.plotly_chart(px.bar(data, x=target, y="count", title=f"Class balance of '{target}'"),
                        key="ml_balance")