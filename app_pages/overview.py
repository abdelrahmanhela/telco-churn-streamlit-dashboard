"""Overview page: data source, dataset summary, and preview."""
import streamlit as st

from utils.data import DEFAULT_CSV, get_raw_data, load_csv

st.title("Overview", icon=":material/home:")
st.caption(
    "Explore the Telco customer churn dataset and train a baseline model — "
    "then move through **Data prep → EDA → Model** in the sidebar."
)

# --- Data source -------------------------------------------------------------
with st.container(border=True):
    st.subheader("Data source", icon=":material/database:")
    uploaded = st.file_uploader("Telco churn CSV", type="csv")
    if uploaded is not None:
        st.session_state["raw_df"] = load_csv(uploaded)
    st.caption(
        f"Drop `{DEFAULT_CSV.name}` into the `data/` folder to load it automatically, "
        "or upload it above."
    )

df = get_raw_data()

if df is None:
    st.info(
        f"No dataset loaded yet. Upload `{DEFAULT_CSV.name}`, or place it at "
        f"`data/{DEFAULT_CSV.name}`."
    )
    st.stop()

# --- KPI row -----------------------------------------------------------------
churn_rate = (df["Churn"] == "Yes").mean()
avg_monthly = df["MonthlyCharges"].astype(float).mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(df):,}", border=True, height="stretch")
k2.metric("Columns", df.shape[1], border=True, height="stretch")
k3.metric("Churn rate", f"{churn_rate:.1%}", border=True, height="stretch")
k4.metric("Avg monthly charges", f"${avg_monthly:.2f}", border=True, height="stretch")

source = uploaded.name if uploaded is not None else f"data/{DEFAULT_CSV.name}"
st.caption(f"Loaded from `{source}`")

# --- Preview -----------------------------------------------------------------
st.subheader("Preview", icon=":material/table_chart:")
st.dataframe(
    df.head(20),
    width="stretch",
    alt="First 20 rows of the Telco churn dataset",
)
