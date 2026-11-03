"""chart_recommender.py - suggests the best chart for the columns you pick, and draws it."""
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st


def column_kind(s):
    """Decides what kind of column this is: numeric, datetime, category or text."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    sample = s.dropna().astype(str).head(50)
    if len(sample) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return "datetime"
    return "category" if s.nunique() <= 30 else "text"


def recommend_charts(df, cols, kinds):
    """Returns (list of {chart, reason}, message). The FIRST chart in the list is the best one."""
    kinds_set = sorted(kinds)

    if "text" in kinds:
        return [], "A column with too many different text values is hard to chart. Pick a different column."

    if len(cols) == 1:
        k = kinds[0]
        if k == "numeric":
            return [
                {"chart": "Histogram", "reason": "Shows how the numbers are spread out."},
                {"chart": "Box plot", "reason": "Shows the middle, the spread and any outliers."},
            ], ""
        if k == "category":
            out = [{"chart": "Bar chart", "reason": "Compares how often each category appears."}]
            if df[cols[0]].nunique() <= 6:
                out.append({"chart": "Pie chart",
                            "reason": "Only a few categories, so shares of the whole are easy to read."})
            return out, ""
        if k == "datetime":
            return [{"chart": "Line chart", "reason": "Shows how many records fall in each month."}], ""

    if len(cols) == 2:
        if kinds_set == ["numeric", "numeric"]:
            return [
                {"chart": "Scatter plot", "reason": "Two numbers: shows whether they rise or fall together."},
                {"chart": "Line chart", "reason": "Only useful if the first column is an ordered sequence."},
            ], ""
        if kinds_set == ["category", "numeric"]:
            cat = cols[kinds.index("category")]
            out = [
                {"chart": "Box plot", "reason": "Compares the spread of the numbers in each category."},
                {"chart": "Bar chart", "reason": "Compares the average number in each category."},
            ]
            if df[cat].nunique() <= 6:
                out.append({"chart": "Pie chart", "reason": "Few categories, so each one's share of the total is clear."})
            return out, ""
        if kinds_set == ["datetime", "numeric"]:
            return [
                {"chart": "Line chart", "reason": "A number over time: a line shows the trend best."},
                {"chart": "Scatter plot", "reason": "Shows each individual point in time."},
            ], ""
        if kinds_set == ["category", "category"]:
            return [
                {"chart": "Grouped bar chart", "reason": "Compares one category inside the other."},
                {"chart": "Heatmap", "reason": "Colors show which combinations are most common."},
            ], ""

    return [], "This combination isn't supported yet. Try a number with a category, or two numbers."


def draw_chart(df, chart, cols, kinds):
    """Draws the chosen chart with Plotly."""
    d = df[cols].copy()
    for c, k in zip(cols, kinds):  # turn date-like text into real dates
        if k == "datetime":
            d[c] = pd.to_datetime(d[c], errors="coerce")

    def pick(kind):
        return cols[kinds.index(kind)] if kind in kinds else None

    num, cat, dt = pick("numeric"), pick("category"), pick("datetime")

    # ----- one column -----
    if len(cols) == 1:
        c = cols[0]
        if chart == "Histogram":
            return px.histogram(d, x=c, title=f"Distribution of {c}")
        if chart == "Box plot":
            return px.box(d, y=c, title=f"Box plot of {c}")
        if chart in ("Bar chart", "Pie chart"):
            vc = d[c].value_counts().head(15).reset_index()
            vc.columns = [c, "count"]
            if chart == "Bar chart":
                return px.bar(vc, x=c, y="count", title=f"Count of each {c}")
            return px.pie(vc, names=c, values="count", title=f"Share of each {c}")
        if chart == "Line chart":
            per = d[c].dropna().dt.to_period("M").astype(str).value_counts().sort_index().reset_index()
            per.columns = [c, "count"]
            return px.line(per, x=c, y="count", markers=True, title=f"Records per month ({c})")

    # ----- two columns -----
    a, b = cols
    if chart in ("Scatter plot", "Line chart") and (num or dt):
        x, y = (dt, num) if dt else (a, b)
        data = d.sort_values(x) if chart == "Line chart" else d
        fn = px.line if chart == "Line chart" else px.scatter
        return fn(data, x=x, y=y, title=f"{y} vs {x}")
    if chart == "Box plot":
        return px.box(d, x=cat, y=num, title=f"{num} by {cat}")
    if chart == "Bar chart":
        g = d.groupby(cat)[num].mean().reset_index()
        return px.bar(g, x=cat, y=num, title=f"Average {num} by {cat}")
    if chart == "Pie chart":
        g = d.groupby(cat)[num].sum().reset_index()
        return px.pie(g, names=cat, values=num, title=f"Share of total {num} by {cat}")
    if chart == "Grouped bar chart":
        return px.histogram(d, x=a, color=b, barmode="group", title=f"{a} split by {b}")
    if chart == "Heatmap":
        return px.imshow(pd.crosstab(d[a], d[b]), text_auto=True, title=f"{a} vs {b}")


def show_chart_recommender(df):
    st.header("🎨 Smart Chart Recommender")
    cols = st.multiselect("Pick 1 or 2 columns", list(df.columns),
                          max_selections=2, key="cr_cols")
    if not cols:
        st.info("Choose a column and I'll suggest the best chart for it.")
        return

    kinds = [column_kind(df[c]) for c in cols]
    st.caption(" · ".join(f"**{c}** looks like *{k}*" for c, k in zip(cols, kinds)))

    recs, message = recommend_charts(df, cols, kinds)
    if not recs:
        st.warning(message)
        return

    best = recs[0]["chart"]
    reasons = {r["chart"]: r["reason"] for r in recs}
    chart = st.radio("Recommended charts", list(reasons), horizontal=True, key="cr_chart",
                     format_func=lambda n: f"⭐ {n} (best)" if n == best else n)
    st.caption(f"💡 Why: {reasons[chart]}")
    st.plotly_chart(draw_chart(df, chart, cols, kinds), key="cr_figure")