"""Model page: training settings, evaluation metrics, ROC, feature importance."""
import pandas as pd
import streamlit as st

from utils.data import build_features, prepare_data, require_data
from utils.model import MODEL_NAMES, train_and_evaluate
from utils.plots import confusion_matrix_fig, roc_curve_fig

st.title("Churn prediction model", icon=":material/model_training:")
st.caption("Train a baseline classifier and review its evaluation metrics.")

X, y = build_features(prepare_data(require_data()))

# --- Training settings (sidebar form: retrains only on submit) ---------------
with st.sidebar:
    st.header("Training settings", icon=":material/tune:")
    with st.form("training_settings"):
        model_name = st.segmented_control("Model", MODEL_NAMES, default=MODEL_NAMES[0])
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)
        if model_name == "Logistic Regression":
            max_iter = int(st.number_input("Max iterations", 100, 5000, 1000, 100))
            n_estimators = 200
        else:
            n_estimators = int(st.number_input("Trees (random forest)", 50, 1000, 200, 50))
            max_iter = 1000
        st.form_submit_button("Train model", type="primary", width="stretch")

model_name = model_name or MODEL_NAMES[0]

res = train_and_evaluate(
    X, y, model_name, test_size, int(random_state), max_iter, n_estimators
)

# --- Results -----------------------------------------------------------------
st.caption(
    f"{model_name} · test size {test_size:.0%} · random state {int(random_state)}"
)

k1, k2, k3 = st.columns(3)
k1.metric("Accuracy", f"{res['accuracy']:.4f}", border=True, height="stretch")
k2.metric("ROC AUC", f"{res['roc_auc']:.4f}", border=True, height="stretch")
k3.metric("Test rows", f"{res['shapes']['X_test'][0]:,}", border=True, height="stretch")

st.subheader("Evaluation", icon=":material/insights:")
c1, c2 = st.columns(2)
c1.plotly_chart(
    confusion_matrix_fig(res["y_test"], res["y_pred"]),
    width="stretch",
    alt="Confusion matrix of actual versus predicted churn",
)
c2.plotly_chart(
    roc_curve_fig(res["y_test"], res["y_proba"], res["roc_auc"]),
    width="stretch",
    alt="ROC curve of the trained model",
)

st.subheader("Classification report", icon=":material/receipt_long:")
st.dataframe(
    res["report"].round(3),
    width="stretch",
    alt="Precision, recall and F1 score per class",
)


def _split_caption(dist: pd.Series) -> str:
    """Format a 0/1 class distribution as 'Stay 73.2% · Churn 26.8%'."""
    labels = {0: "Stay", 1: "Churn"}
    return " · ".join(f"{labels.get(int(k), k)} {v:.1%}" for k, v in dist.items())


with st.container(border=True):
    st.subheader("Train / test split", icon=":material/call_split:")
    left, right = st.columns(2)
    with left:
        st.markdown("**Training set**")
        st.metric("Rows", f"{res['shapes']['X_train'][0]:,}")
        st.caption(_split_caption(res["train_dist"]))
    with right:
        st.markdown("**Testing set**")
        st.metric("Rows", f"{res['shapes']['X_test'][0]:,}")
        st.caption(_split_caption(res["test_dist"]))

st.subheader("Feature importance", icon=":material/bar_chart:")
imp = pd.Series(res["importance"], index=res["feature_names"]).sort_values()
st.bar_chart(imp, horizontal=True)
