"""app.py - the MAIN file. Loads the data, draws the sidebar, and shows the chosen page."""
import pandas as pd
import streamlit as st

from cleaning import make_sample
from history import init_history
from views import (page_chat, page_clean, page_explore, page_ml_report,
                   page_overview)

st.set_page_config(page_title="Auto Clean + EDA", page_icon="🧹", layout="wide")

PAGES = {
    "🏠 Overview": page_overview,
    "🧼 Clean": page_clean,
    "📊 Explore": page_explore,
    "🤖 ML & Report": page_ml_report,
    "💬 Ask your data": page_chat,
}
STATE_KEYS = ["source", "source_key", "original", "clean", "history",
              "pdf", "chat", "file_size"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧹 Auto Clean + EDA")
    st.caption("Upload, clean, explore and report, in one place.")
    file = st.file_uploader("Upload a CSV, Excel or JSON file", type=["csv", "xlsx", "json"])
    use_sample = st.checkbox("Or use sample messy data")

# ---------------------------------------------------------------- load data (only when the file changes)
if file is not None:
    key, name, size = f"{file.name}|{file.size}", file.name, file.size
elif use_sample:
    key, name, size = "sample", "sample", None
else:
    key = None

if key is None:  # nothing selected: forget any old data
    for k in STATE_KEYS:
        st.session_state.pop(k, None)
elif st.session_state.get("source_key") != key:
    try:
        if file is None:
            df = make_sample()
        elif name.endswith("csv"):
            df = pd.read_csv(file)
        elif name.endswith("json"):
            df = pd.read_json(file)
        else:
            df = pd.read_excel(file)
        init_history(df)
        st.session_state.source_key = key
        st.session_state.source = name
        st.session_state.file_size = size
    except Exception as e:
        st.sidebar.error(f"Could not read this file: {e}")
        st.session_state.pop("source_key", None)

# ---------------------------------------------------------------- show the page
if "clean" not in st.session_state:
    st.title("🧹 Auto Data Cleaning + EDA Tool")
    st.write("Clean messy data and explore it automatically, in a few clicks.")
    st.info("👈 Upload a file in the sidebar, or tick **Use sample messy data** to try it.")
    a, b, c = st.columns(3)
    a.markdown("**1. Check**\n\nSee quality scores, missing values, duplicates and outliers.")
    b.markdown("**2. Clean**\n\nApply smart recommendations with one click. Undo anytime.")
    c.markdown("**3. Explore**\n\nCharts, insights, an ML readiness check and a PDF report.")
else:
    with st.sidebar:
        st.divider()
        page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
        st.divider()
        d = st.session_state.clean
        st.caption(f"📁 {st.session_state.source}")
        st.caption(f"{d.shape[0]:,} rows × {d.shape[1]} columns")
        st.caption(f"🕘 Version {len(st.session_state.history) - 1}")
    PAGES[page]()