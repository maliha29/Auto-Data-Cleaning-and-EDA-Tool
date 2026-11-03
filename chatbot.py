"""chatbot.py - answers plain-English questions about the data (rule-based, no API key needed)."""
import re

import pandas as pd
import streamlit as st

from insights import top_correlations
from ml_readiness import ml_checks
from outliers import outlier_summary
from recommendations import get_recommendations


# ------------------------------------------------------------------ understanding the question
def find_columns(df, q):
    """Which column names does the question mention? (also understands plurals: 'products' -> product)"""
    found = []
    for c in df.columns:
        name = str(c).lower().replace("_", " ")
        variants = {name, name + "s", name + "es"}
        if name.endswith("s"):
            variants.add(name[:-1])
        positions = [m.start() for v in variants
                     for m in [re.search(rf"\b{re.escape(v)}\b", q)] if m]
        if positions:
            found.append((min(positions), c))
    return [c for _, c in sorted(found, key=lambda x: x[0])]


def reply(text, table=None):
    return {"text": text, "table": table}


def answer_question(df, question):
    """Reads the question, picks the matching skill, and returns {'text': ..., 'table': ...}."""
    q = question.lower().strip()
    cols = find_columns(df, q)
    is_num = lambda c: pd.api.types.is_numeric_dtype(df[c])
    numc = [c for c in cols if is_num(c)]
    catc = [c for c in cols if not is_num(c)]

    # ---- missing values
    if re.search(r"\b(missing|null|nulls|empty|blank|nan)\b", q):
        counts = df.isna().sum()
        if cols:
            lines = [f"- **{c}**: {int(counts[c])} missing ({df[c].isna().mean():.1%})" for c in cols]
            return reply("\n".join(lines))
        if counts.max() == 0:
            return reply("There are no missing values in your data. 🎉")
        table = pd.DataFrame({"column": counts.index, "missing": counts.values,
                              "missing %": (df.isna().mean().values * 100).round(1)})
        table = table[table["missing"] > 0].sort_values("missing", ascending=False)
        top = table.iloc[0]
        return reply(f"**{top['column']}** has the most missing values: {int(top['missing'])} "
                     f"({top['missing %']}%).", table)

    # ---- correlation
    if "correlat" in q or "related" in q or "relationship" in q:
        if len(numc) >= 2:
            r = df[numc[0]].corr(df[numc[1]])
            return reply(f"The correlation between **{numc[0]}** and **{numc[1]}** is **{r:.2f}**.")
        pairs = top_correlations(df, 5)
        if not pairs:
            return reply("I need at least two numeric columns to measure correlation.")
        table = pd.DataFrame(pairs, columns=["column A", "column B", "correlation"]).round(2)
        a, b, r = pairs[0]
        return reply(f"The strongest relationship is between **{a}** and **{b}** (r = {r:.2f}).", table)

    # ---- top / bottom N
    m = re.search(r"\b(top|bottom|best|worst)\b(?:\s+(\d+))?", q) or \
        re.search(r"\b(highest|lowest)\s+(\d+)\b", q)
    if m:
        n = int(m.group(2)) if m.group(2) else 5
        asc = m.group(1) in ("bottom", "worst", "lowest")
        word = "Bottom" if asc else "Top"
        if numc and catc:
            g = df.groupby(catc[0])[numc[0]].sum().sort_values(ascending=asc).head(n).reset_index()
            return reply(f"{word} {n} **{catc[0]}** by total **{numc[0]}**:", g)
        if numc:
            return reply(f"{word} {n} rows by **{numc[0]}**:",
                         df.sort_values(numc[0], ascending=asc).head(n))
        if catc:
            vc = df[catc[0]].value_counts(ascending=asc).head(n).reset_index()
            vc.columns = [catc[0], "count"]
            return reply(f"{word} {n} most frequent values of **{catc[0]}**:", vc)
        return reply("Which column should I rank? For example: *Show top 5 city by salary*.")

    # ---- duplicates
    if "duplicate" in q:
        return reply(f"There are **{int(df.duplicated().sum())}** duplicate rows.")

    # ---- outliers
    if "outlier" in q or "anomal" in q:
        summary = outlier_summary(df)
        if summary.empty:
            return reply("There are no numeric columns to check for outliers.")
        return reply("Outliers found with the IQR and Z-score methods:", summary)

    # ---- preprocessing advice
    if re.search(r"preprocess|suggest|recommend|clean|prepare|ready|next step", q):
        recs = get_recommendations(df)
        score, checks = ml_checks(df)
        lines = [f"- {r['title']}: {r['why']}" for r in recs]
        lines += [f"- {t}. {a}" for ok, t, a in checks if not ok]
        if not lines:
            return reply(f"Your data already looks ready. ML readiness score: **{score}/100**.")
        return reply(f"Here is what I would do (ML readiness score: **{score}/100**):\n\n" + "\n".join(lines))

    # ---- size of the data
    if re.search(r"how many (rows|records|entries|columns|features)|number of (rows|columns)|\bshape\b|\bsize\b", q):
        return reply(f"The data has **{len(df):,} rows** and **{df.shape[1]} columns**.")

    # ---- average / max / min / sum
    ops = {"average": "mean", "avg": "mean", "mean": "mean", "median": "median", "max": "max",
           "maximum": "max", "highest": "max", "largest": "max", "min": "min", "minimum": "min",
           "lowest": "min", "smallest": "min", "sum": "sum", "total": "sum", "std": "std"}
    m = re.search(r"\b(" + "|".join(ops) + r")\b", q)
    if m:
        if not numc:
            nums = ", ".join(df.select_dtypes("number").columns) or "none"
            return reply(f"Which numeric column? Available: {nums}.")
        op = ops[m.group(1)]
        lines = [f"- {op} of **{c}**: {getattr(df[c], op)():,.2f}" for c in numc]
        return reply("\n".join(lines))

    # ---- most common / unique values
    if re.search(r"most common|most frequent|unique|distinct|categories|value counts|how many different", q):
        col = (catc or cols or [None])[0]
        if col is None:
            return reply("Which column? For example: *What is the most common city?*")
        vc = df[col].value_counts().head(10).reset_index()
        vc.columns = [col, "count"]
        if vc.empty:
            return reply(f"**{col}** has no values.")
        return reply(f"**{col}** has {df[col].nunique()} unique values. The most common is "
                     f"**{vc.iloc[0][col]}** ({vc.iloc[0]['count']} rows).", vc)

    # ---- summary statistics
    if re.search(r"summary|describe|statistic|stats|overview", q):
        target = df[cols] if cols else df
        num = target.select_dtypes("number")
        table = (num if not num.empty else target).describe().T.round(2).reset_index()
        return reply("Summary statistics:", table.rename(columns={"index": "column"}))

    # ---- list the columns
    if re.search(r"(columns|features|fields)|column names", q):
        table = pd.DataFrame({"column": df.columns, "type": df.dtypes.astype(str).values})
        return reply(f"Your data has {df.shape[1]} columns:", table)

    return reply(
        "I'm not sure how to answer that yet. Try one of these:\n\n"
        "- Which column has the most missing values?\n"
        "- Which features are highly correlated?\n"
        "- What is the average of a numeric column?\n"
        "- Show top 5 (category column) by (numeric column)\n"
        "- Are there outliers? / How many duplicates?\n"
        "- Suggest preprocessing steps")


# ------------------------------------------------------------------ the chat window
def show_chatbot(df):
    st.header("💬 Ask Your Data")
    chat = st.session_state.setdefault("chat", [])

    num = df.select_dtypes("number").columns.tolist()
    cat = [c for c in df.columns if c not in num]
    examples = ["Which column has the most missing values?",
                "Which features are highly correlated?",
                "Suggest preprocessing steps"]
    if num and cat:
        examples.append(f"Show top 5 {cat[0]} by {num[0]}")

    st.caption("Try an example, or type your own question below:")
    clicked = None
    for i, (box, text) in enumerate(zip(st.columns(len(examples)), examples)):
        if box.button(text, key=f"chat_ex_{i}", width="stretch"):
            clicked = text

    messages = st.container()
    typed = st.chat_input("Ask a question about your data", key="chat_box")
    question = typed or clicked
    if question:
        chat.append({"role": "user", "text": question, "table": None})
        a = answer_question(df, question)
        chat.append({"role": "assistant", "text": a["text"], "table": a["table"]})

    with messages:
        for i, msg in enumerate(chat):
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
                if msg["table"] is not None:
                    st.dataframe(msg["table"], hide_index=True, key=f"chat_tbl_{i}")

    if chat and st.button("Clear chat", key="chat_clear"):
        chat.clear()
        st.rerun()