# 🗽 NYC Airbnb Price Intelligence

A professional data-science web app built with **Streamlit** that analyses the 2019 New York City Airbnb dataset and predicts nightly prices using **Linear Regression** and **XGBoost**.

---

## ✨ Features

| Page | What it shows |
|---|---|
| **Dashboard** | KPI cards, borough price bar chart, room-type donut, price box-plots, and an interactive listing density map |
| **EDA & Insights** | Price distributions, outlier analysis, borough/neighbourhood deep-dives, room-type heatmap, correlation matrix |
| **Price Predictor** | Live price estimate — choose borough, neighbourhood, room type, and listing attributes; compare vs XGBoost & Linear Regression |
| **Model Metrics** | MAE, MSE, RMSE, R² side-by-side; actual-vs-predicted scatter; full methodology notes |

---

## 🚀 Running Locally

```bash
# 1 — Clone the repo
git clone https://github.com/<your-username>/nyc-airbnb-app.git
cd nyc-airbnb-app

# 2 — Install dependencies
pip install -r requirements.txt

# 3 — Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Choose your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy** — done! ✅

No secrets or API keys required.

---

## 📂 Project Structure

```
nyc_airbnb_app/
├── app.py                  # Entry point, sidebar, global CSS
├── utils.py                # Data loader, colour maps, shared Plotly layout
├── requirements.txt
├── README.md
├── data/
│   └── AB_NYC_2019_processed01.csv   # Pre-processed dataset (48,360 rows)
└── pages/
    ├── __init__.py
    ├── dashboard.py        # Overview dashboard
    ├── eda.py              # Exploratory analysis tabs
    ├── predictor.py        # Interactive price prediction
    └── metrics.py          # Model evaluation & methodology
```

---

## 🧪 Model Details

- **Target:** z-score normalised `price` → back-converted to USD for display (mean = $106, std = $64)
- **Features:** `room_type`, `neighbourhood_group`, `neighbourhood`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`
- **Outlier removal:** IQR method — listings above ≈ $334/night excluded
- **Train / Test split:** 80 / 20, `random_state=42`
- **XGBoost config:** `n_estimators=300`, `learning_rate=0.05`, `max_depth=8`

---

## 📊 Dataset

[New York City Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — Kaggle

---

## 🛠 Tech Stack

`Python` · `Streamlit` · `pandas` · `scikit-learn` · `XGBoost` · `Plotly`
