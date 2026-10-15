"""Telco Churn Explorer — entry point (router).

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Telco Churn Explorer",
    page_icon=":material/analytics:",
    layout="wide",
)

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/home:"),
    st.Page("app_pages/data_prep.py", title="Data prep", icon=":material/filter_alt:"),
    st.Page("app_pages/eda.py", title="EDA", icon=":material/query_stats:"),
    st.Page("app_pages/model.py", title="Model", icon=":material/model_training:"),
]

pg = st.navigation(pages)
pg.run()
