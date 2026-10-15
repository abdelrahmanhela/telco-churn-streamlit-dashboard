"""Data preparation page: binary mapping, column overview, feature matrix."""
import streamlit as st

from utils.data import BINARY_COLS, SELECTED_FEATURES, build_features, prepare_data, require_data

st.title("Data preparation", icon=":material/filter_alt:")
st.caption(
    "Map Yes/No columns to booleans, review column types, and build the "
    "feature matrix used by the model."
)

raw = require_data()
prepared = prepare_data(raw)

# --- Step 1: binary mapping --------------------------------------------------
with st.container(border=True):
    st.subheader("1. Yes/No → True/False", icon=":material/toggle_on:")
    st.caption(f"Converted columns: {', '.join(BINARY_COLS)}")
    st.dataframe(
        prepared[BINARY_COLS].head(),
        width="stretch",
        hide_index=True,
        alt="Converted Yes/No columns, first 5 rows",
    )

# --- Step 2: column overview -------------------------------------------------
with st.container(border=True):
    st.subheader("2. Columns", icon=":material/list:")
    n_numeric = len(prepared.select_dtypes("number").columns)
    m1, m2, m3 = st.columns(3)
    m1.metric("Columns", prepared.shape[1])
    m2.metric("Numeric", n_numeric)
    m3.metric("Missing values", int(prepared.isna().sum().sum()))

    dtypes = (
        prepared.dtypes.astype(str)
        .rename("dtype")
        .rename_axis("column")
        .reset_index()
    )
    st.dataframe(
        dtypes,
        width="stretch",
        hide_index=True,
        alt="Data type of each column",
    )

# --- Step 3: feature matrix --------------------------------------------------
X, y = build_features(prepared)

with st.container(border=True):
    st.subheader("3. Feature matrix", icon=":material/grid_view:")
    st.caption(f"Selected features: {', '.join(SELECTED_FEATURES)}")
    m1, m2 = st.columns(2)
    m1.metric("Shape", f"{X.shape[0]} × {X.shape[1]}")
    m2.metric("Missing values in X", int(X.isna().sum().sum()))
    st.dataframe(
        X.head(),
        width="stretch",
        hide_index=True,
        alt="First rows of the model feature matrix",
    )
