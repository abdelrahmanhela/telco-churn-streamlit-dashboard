"""EDA page: churn distributions for categorical and numeric features."""
import streamlit as st

from utils.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_data, require_data
from utils.plots import categorical_by_churn, numeric_by_churn

st.title("Exploratory data analysis", icon=":material/query_stats:")
st.caption("Compare churn across customer segments and billing profiles.")

df = prepare_data(require_data())

# Lazy tabs: only the selected tab computes its charts.
tab_cat, tab_num = st.tabs(
    ["Categorical features", "Numeric features"],
    on_change="rerun",
    key="eda_tabs",
)

with tab_cat:
    if tab_cat.open:
        show_all = st.toggle("Show all categorical plots", value=False)
        if show_all:
            cols = st.columns(2)
            for i, col in enumerate(CATEGORICAL_FEATURES):
                with cols[i % 2]:
                    st.plotly_chart(
                        categorical_by_churn(df, col),
                        width="stretch",
                        alt=f"Distribution of {col} by churn status",
                    )
        else:
            feature = st.selectbox("Feature", CATEGORICAL_FEATURES)
            st.plotly_chart(
                categorical_by_churn(df, feature),
                width="stretch",
                alt=f"Distribution of {feature} by churn status",
            )

with tab_num:
    if tab_num.open:
        for col in NUMERIC_FEATURES:
            st.plotly_chart(
                numeric_by_churn(df, col),
                width="stretch",
                alt=f"Distribution of {col} by churn status",
            )
