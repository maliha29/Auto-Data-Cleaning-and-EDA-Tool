"""views.py - one function per page of the website. app.py decides which one to show."""
import streamlit as st

from chart_recommender import show_chart_recommender
from chatbot import show_chatbot
from cleaning import (fill_missing_numbers, fill_missing_text, health,
                      remove_duplicates, trim_spaces)
from eda import show_eda
from history import can_undo, commit, get_log, show_history, undo
from ml_readiness import show_ml_readiness
from outliers import cap_all, outlier_summary, remove_outlier_rows
from quality import show_quality_dashboard
from recommendations import apply_recommendation, get_recommendations
from report import build_report

import pandas as pd


def fmt_bytes(n):
    if n is None:
        return "built-in sample"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def health_row():
    """Three numbers that show the effect of cleaning. Used on the Overview and Clean pages."""
    original, clean = st.session_state.original, st.session_state.clean
    m0, d0, s0 = health(original)
    m1, d1, s1 = health(clean)
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing cells", m1, delta=m1 - m0, delta_color="inverse")
    c2.metric("Duplicate rows", d1, delta=d1 - d0, delta_color="inverse")
    c3.metric("Quality score", f"{s1} / 100", delta=round(s1 - s0, 1))
    st.progress(int(s1))


# ---------------------------------------------------------------- page 1
def page_overview():
    original, clean = st.session_state.original, st.session_state.clean
    st.header("🏠 Overview")

    c = st.columns(5)
    c[0].metric("Rows", f"{original.shape[0]:,}")
    c[1].metric("Columns", original.shape[1])
    c[2].metric("File size", fmt_bytes(st.session_state.get("file_size")))
    c[3].metric("Memory", fmt_bytes(int(original.memory_usage(deep=True).sum())))
    c[4].metric("Numeric columns", original.select_dtypes("number").shape[1])

    st.subheader("Data preview")
    st.dataframe(original.head(50), width="stretch")

    st.subheader("Columns and data types")
    types = pd.DataFrame({
        "column": original.columns,
        "type": original.dtypes.astype(str).values,
        "missing": original.isna().sum().values,
        "missing %": (original.isna().mean().values * 100).round(1),
        "unique values": original.nunique().values,
    })
    st.dataframe(types, hide_index=True, width="stretch")

    st.subheader("Data health: before vs after cleaning")
    health_row()
    show_quality_dashboard(clean)


# ---------------------------------------------------------------- page 2
def page_clean():
    original, clean = st.session_state.original, st.session_state.clean
    st.header("🧼 Clean your data")
    health_row()

    st.subheader("Cleaning Studio")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    if b1.button("Remove duplicates"):
        commit(remove_duplicates(clean), "Removed duplicate rows")
        st.rerun()
    if b2.button("Fill missing numbers"):
        commit(fill_missing_numbers(clean), "Filled missing numbers (median)")
        st.rerun()
    if b3.button("Fill missing text"):
        commit(fill_missing_text(clean), "Filled missing text (most common value)")
        st.rerun()
    if b4.button("Trim extra spaces"):
        commit(trim_spaces(clean), "Trimmed extra spaces in text")
        st.rerun()
    if b5.button("↩ Undo", disabled=not can_undo()):
        undo()
        st.rerun()
    if b6.button("Reset everything"):
        commit(original.copy(), "Reset to original")
        st.rerun()

    st.subheader("🤖 Smart Recommendations")
    recs = get_recommendations(clean)
    if not recs:
        st.success("No cleaning needed. Your data looks healthy!")
    for i, rec in enumerate(recs):
        with st.container(border=True):
            left, mid, right = st.columns([5, 2, 1])
            left.markdown(f"**{rec['title']}**")
            left.caption(rec["why"])
            mid.write(f"Confidence: {rec['confidence']}%")
            mid.progress(rec["confidence"] / 100)
            if right.button("Apply", key=f"rec_{i}"):
                commit(apply_recommendation(clean, rec), rec["title"])
                st.rerun()

    st.subheader("🎯 Outlier Handling")
    summary = outlier_summary(clean)
    if summary.empty:
        st.info("No numeric columns to check.")
    else:
        st.dataframe(summary, hide_index=True)
        o1, o2 = st.columns(2)
        if o1.button("Cap all outliers (keep rows)"):
            commit(cap_all(clean), "Capped outliers (IQR method)")
            st.rerun()
        if o2.button("Remove rows with outliers"):
            commit(remove_outlier_rows(clean), "Removed rows with outliers (IQR method)")
            st.rerun()

    st.subheader("Before vs after")
    left, right = st.columns(2)
    left.write(f"**Original data** ({original.shape[0]} rows)")
    left.dataframe(original.head(20))
    right.write(f"**Cleaned data** ({clean.shape[0]} rows)")
    right.dataframe(clean.head(20))
    st.download_button("⬇ Download cleaned CSV", clean.to_csv(index=False).encode(),
                       "cleaned.csv", "text/csv")

    show_history()


# ---------------------------------------------------------------- page 3
def page_explore():
    clean = st.session_state.clean
    show_eda(clean)
    show_chart_recommender(clean)


# ---------------------------------------------------------------- page 4
def page_ml_report():
    original, clean = st.session_state.original, st.session_state.clean
    show_ml_readiness(clean)

    st.header("📄 One-Click PDF Report")
    if st.button("Generate PDF report"):
        with st.spinner("Building your report..."):
            st.session_state.pdf = build_report(original, clean, get_log(), st.session_state.source)
    if "pdf" in st.session_state:
        st.download_button("⬇ Download PDF report", st.session_state.pdf,
                           "data_report.pdf", "application/pdf")
        st.caption("The report shows your data as it was when you clicked Generate.")


# ---------------------------------------------------------------- page 5
def page_chat():
    show_chatbot(st.session_state.clean)