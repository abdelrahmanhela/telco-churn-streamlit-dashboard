"""Plotly figure builders (replaces the many copy-pasted px.histogram cells)."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve


def categorical_by_churn(df: pd.DataFrame, col: str):
    return px.histogram(
        df, x=col, color="Churn", barmode="group",
        title=f"Distribution of {col} by Churn Status",
        labels={col: col, "count": "Number of Customers"},
    )


def numeric_by_churn(df: pd.DataFrame, col: str):
    return px.histogram(
        df.dropna(subset=[col]), x=col, color="Churn",
        marginal="box", barmode="overlay", opacity=0.7,
        title=f"Distribution of {col} by Churn Status",
        labels={col: col, "count": "Number of Customers"},
    )


def confusion_matrix_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted: Stay", "Predicted: Churn"],
        y=["Actual: Stay", "Actual: Churn"],
        title="Confusion Matrix",
    )
    return fig


def roc_curve_fig(y_true, y_proba, auc: float):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC = {auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    return fig
