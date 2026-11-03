"""quality.py - calculates the 4 quality scores and draws the gauges."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def quality_scores(df):
    # Completeness: how much of the data is NOT empty
    completeness = 100 - df.isna().mean().mean() * 100

    # Consistency: "Delhi", "delhi", "Delhi " should count as one value
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    ratios = []
    for c in text_cols:
        s = df[c].dropna().astype(str)
        if s.nunique() > 0:
            ratios.append(s.str.strip().str.lower().nunique() / s.nunique())
    consistency = 100 * sum(ratios) / len(ratios) if ratios else 100

    # Validity: numbers should not be stored as text
    bad_types = 0
    for c in text_cols:
        s = df[c].dropna().astype(str).str.strip()
        if len(s) > 0 and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9:
            bad_types += 1
    validity = 100 - bad_types / len(df.columns) * 100

    # Accuracy: fewer extreme outliers means higher accuracy
    num = df.select_dtypes("number")
    outliers = 0
    for c in num.columns:
        q1, q3 = num[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers += ((num[c] < q1 - 1.5 * iqr) | (num[c] > q3 + 1.5 * iqr)).sum()
    accuracy = 100 - outliers / num.size * 100 if num.size else 100

    overall = (completeness + consistency + validity + accuracy) / 4
    return {
        "Completeness": round(completeness, 1),
        "Consistency": round(consistency, 1),
        "Validity": round(validity, 1),
        "Accuracy": round(accuracy, 1),
        "Overall": round(overall, 1),
    }


def show_quality_dashboard(df):
    """Draws the 5 gauges on the page."""
    st.subheader("📈 Data Quality Dashboard")
    scores = quality_scores(df)
    boxes = st.columns(5)
    for box, (name, value) in zip(boxes, scores.items()):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": name},
            gauge={"axis": {"range": [0, 100]}},
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
        box.plotly_chart(fig, key=f"gauge_{name}")