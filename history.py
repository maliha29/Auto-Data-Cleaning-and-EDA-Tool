"""history.py - remembers every cleaning step so you can undo or restore any version."""
import streamlit as st

MAX_VERSIONS = 20  # keeps memory small


def init_history(df):
    """Start fresh when a new file is loaded."""
    st.session_state.pop("pdf", None)
    st.session_state.pop("chat", None)
    st.session_state.original = df
    st.session_state.history = [{"label": "Original data", "df": df.copy()}]
    st.session_state.clean = df.copy()


def commit(new_df, label):
    """Save a new version after a cleaning step."""
    history = st.session_state.history
    history.append({"label": label, "df": new_df.copy()})
    if len(history) > MAX_VERSIONS:  # always keep the original + the newest versions
        st.session_state.history = [history[0]] + history[-(MAX_VERSIONS - 1):]
    st.session_state.clean = new_df


def can_undo():
    return len(st.session_state.history) > 1


def undo():
    """Go back one step."""
    if can_undo():
        st.session_state.history.pop()
        st.session_state.clean = st.session_state.history[-1]["df"].copy()


def restore(index):
    """Go back to an older version. This is saved as a new step, so nothing is ever lost."""
    old = st.session_state.history[index]
    commit(old["df"].copy(), f"Restored version {index}: {old['label']}")


def get_log():
    """List of step names, used by the report later."""
    return [v["label"] for v in st.session_state.history]


def show_history():
    st.subheader("🕘 Cleaning History")
    history = st.session_state.history
    last = len(history) - 1

    for i, version in enumerate(history):
        d = version["df"]
        left, mid, right = st.columns([5, 4, 1])
        marker = "▶ **" if i == last else ""
        end = "** (current)" if i == last else ""
        left.write(f"{marker}Version {i}: {version['label']}{end}")
        mid.caption(f"{d.shape[0]} rows × {d.shape[1]} columns · {int(d.isna().sum().sum())} empty cells")
        if i != last and right.button("Restore", key=f"restore_{i}"):
            restore(i)
            st.rerun()