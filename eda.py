"""eda.py - all the automatic charts and insights."""
import plotly.express as px
import streamlit as st


def show_eda(df):
    st.header("📊 Automatic EDA")

    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Numbers", "Categories", "Correlation", "Insights"]
    )

    with tab1:
        if num_cols:
            col = st.selectbox("Choose a numeric column", num_cols)
            a, b = st.columns(2)
            a.plotly_chart(px.histogram(df, x=col, title=f"Distribution of {col}"))
            b.plotly_chart(px.box(df, y=col, title=f"Boxplot of {col}"))
            st.dataframe(df[col].describe())
        else:
            st.info("No numeric columns found.")

    with tab2:
        if cat_cols:
            cc = st.selectbox("Choose a text column", cat_cols)
            counts = df[cc].value_counts().head(15).reset_index()
            counts.columns = [cc, "count"]
            st.plotly_chart(px.bar(counts, x=cc, y="count",
                                   title=f"Most common values in {cc}"))
        else:
            st.info("No text columns found.")

    with tab3:
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=".2f",
                                      color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1,
                                      title="Which columns move together?"))
        else:
            st.info("Need at least 2 numeric columns.")

    with tab4:
        notes = []
        for c in df.columns:
            miss = df[c].isna().mean()
            if miss > 0.3:
                notes.append(f"⚠️ **{c}** is {miss:.0%} empty.")
        for c in num_cols:
            skew = df[c].skew()
            if abs(skew) > 1:
                notes.append(f"📈 **{c}** is lopsided (skew = {skew:.2f}).")
        for i, x in enumerate(num_cols):
            for y in num_cols[i + 1:]:
                r = df[[x, y]].corr().iloc[0, 1]
                if abs(r) > 0.7:
                    notes.append(f"🔗 **{x}** and **{y}** are strongly related (r = {r:.2f}).")
        if notes:
            for n in notes:
                st.write(n)
        else:
            st.success("No major problems found in this data.")