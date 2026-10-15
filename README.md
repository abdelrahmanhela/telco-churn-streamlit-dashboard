# Telco Churn Explorer (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` on the Overview page, or drop it in `data/`
under that exact name and it loads automatically.

```
churn_app/
├── app.py              # entry point: st.navigation router
├── app_pages/
│   ├── overview.py     # data source, dataset summary, preview
│   ├── data_prep.py    # binary mapping, column types, feature matrix
│   ├── eda.py          # churn distributions (lazy tabs)
│   └── model.py        # sidebar training form, metrics, ROC, importance
├── .streamlit/
│   └── config.toml     # dual light/dark theme (theme tokens, no CSS)
└── utils/
    ├── data.py         # load / prepare / build features
    ├── plots.py        # plotly figures
    └── model.py        # training + evaluation
```
