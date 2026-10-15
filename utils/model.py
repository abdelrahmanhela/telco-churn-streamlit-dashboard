"""Train/test split and model training (notebook modelling cells)."""
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_NAMES = ["Logistic Regression", "Random Forest"]


def make_model(name: str, random_state: int, max_iter: int = 1000, n_estimators: int = 200):
    if name == "Logistic Regression":
        return LogisticRegression(random_state=random_state, max_iter=max_iter)
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


@st.cache_data(show_spinner="Training model...")
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, model_name: str, test_size: float,
                       random_state: int, max_iter: int, n_estimators: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = make_model(model_name, random_state, max_iter, n_estimators)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "shapes": {"X_train": X_train.shape, "X_test": X_test.shape},
        "train_dist": y_train.value_counts(normalize=True).sort_index(),
        "test_dist": y_test.value_counts(normalize=True).sort_index(),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T,
        "feature_names": list(X.columns),
        "importance": (
            model.coef_[0] if model_name == "Logistic Regression" else model.feature_importances_
        ),
    }
