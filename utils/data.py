"""Data loading and preparation (notebook cells 1-3 and the feature-matrix cell)."""
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Yes/No columns that become True/False
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

# Categorical columns plotted against Churn
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Columns used for the model
SELECTED_FEATURES = [
    "Partner", "Dependents", "PaperlessBilling", "InternetService", "OnlineSecurity",
    "MonthlyCharges", "tenure", "Contract", "TechSupport", "DeviceProtection",
]
CONTRACT_ORDER = {"Month-to-month": 0, "One year": 1, "Two year": 2}


@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    """`source` can be an uploaded file object or a path."""
    return pd.read_csv(source)


@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No columns to booleans and make TotalCharges numeric."""
    out = df.copy()
    for col in BINARY_COLS:
        if col in out.columns:
            out[col] = out[col].map({"Yes": True, "No": False})
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out


@st.cache_data(show_spinner=False)
def build_features(df_prepared: pd.DataFrame):
    """Build X (numeric feature matrix) and y (0/1 churn) from the prepared data."""
    X = df_prepared[SELECTED_FEATURES].copy()

    for col in ["Partner", "Dependents", "PaperlessBilling"]:
        X[col] = X[col].astype(int)

    # Yes/No -> 1/0, 'No internet service' -> NaN -> 0
    for col in ["OnlineSecurity", "TechSupport", "DeviceProtection"]:
        X[col] = X[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    X["Contract"] = X["Contract"].map(CONTRACT_ORDER).astype(int)
    X = pd.get_dummies(X, columns=["InternetService"], drop_first=True, dtype=int)

    y = df_prepared["Churn"].astype(int)
    return X, y


def get_raw_data():
    """Return the raw dataframe from the session (uploaded) or the default file, else None."""
    if "raw_df" in st.session_state:
        return st.session_state["raw_df"]
    if DEFAULT_CSV.exists():
        st.session_state["raw_df"] = load_csv(DEFAULT_CSV)
        return st.session_state["raw_df"]
    return None


def require_data():
    """Use at the top of every page: stop with a friendly message if no data is loaded."""
    df = get_raw_data()
    if df is None:
        st.warning("No dataset loaded yet. Go to the **Home** page and upload the Telco churn CSV.")
        st.stop()
    return df
