import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import copy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from utils import get_enriched_df, PLOT_LAYOUT
from pages.predictor import train_models

PRICE_MEAN = 106.0
PRICE_STD  = 64.0


@st.cache_data
def evaluate_models():
    df = get_enriched_df()
    lr, xgb, le_room, le_hood, le_boro, feature_cols = train_models()

    df2 = df.copy()
    df2["room_enc"] = le_room.transform(df2["room_type"])
    df2["hood_enc"] = le_hood.transform(df2["neighbourhood"])
    df2["boro_enc"] = le_boro.transform(df2["neighbourhood_group"])

    X = df2[feature_cols]
    y = df2["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    for name, model in [("Linear Regression", lr), ("XGBoost", xgb)]:
        y_pred = model.predict(X_test)
        # back to USD
        y_test_usd = y_test * PRICE_STD + PRICE_MEAN
        y_pred_usd = y_pred * PRICE_STD + PRICE_MEAN
        results[name] = {
            "mae":   mean_absolute_error(y_test_usd, y_pred_usd),
            "mse":   mean_squared_error(y_test_usd,  y_pred_usd),
            "rmse":  np.sqrt(mean_squared_error(y_test_usd, y_pred_usd)),
            "r2":    r2_score(y_test_usd, y_pred_usd),
            "y_test": y_test_usd.values[:300],
            "y_pred": y_pred_usd[:300],
        }
    return results


def render():
    from utils import inject_styles
    inject_styles()
    st.markdown("""
    <div style="padding:2rem 0 1rem">
        <p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
            MODEL EVALUATION
        </p>
        <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.4rem;
                   font-weight:800;margin:0;letter-spacing:-0.03em">
            Model <span style="color:#6060c8">Metrics</span>
        </h1>
        <p style="color:#5050a0;font-size:0.9rem;margin-top:0.8rem">
            Linear Regression vs XGBoost · 80/20 train-test split · 48,360 listings
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Evaluating models…"):
        results = evaluate_models()

    lr_r  = results["Linear Regression"]
    xgb_r = results["XGBoost"]

    # ── KPI comparison ─────────────────────────────────────────────────────────
    st.markdown("### Performance Summary")
    head = st.columns([2, 1, 1])
    head[0].markdown("<p style='color:#4040a0;font-size:0.78rem;letter-spacing:0.08em'>METRIC</p>", unsafe_allow_html=True)
    head[1].markdown("<p style='color:#7b7bff;font-size:0.78rem;letter-spacing:0.08em'>LINEAR REGRESSION</p>", unsafe_allow_html=True)
    head[2].markdown("<p style='color:#7bffcc;font-size:0.78rem;letter-spacing:0.08em'>XGBOOST</p>", unsafe_allow_html=True)

    rows = [
        ("MAE (USD)",  f"${lr_r['mae']:.2f}",  f"${xgb_r['mae']:.2f}"),
        ("MSE",        f"{lr_r['mse']:,.1f}",   f"{xgb_r['mse']:,.1f}"),
        ("RMSE (USD)", f"${lr_r['rmse']:.2f}",  f"${xgb_r['rmse']:.2f}"),
        ("R² Score",   f"{lr_r['r2']:.4f}",     f"{xgb_r['r2']:.4f}"),
    ]
    for label, lrv, xgbv in rows:
        rc = st.columns([2, 1, 1])
        rc[0].markdown(f"<p style='color:#9090c0;padding:0.4rem 0'>{label}</p>", unsafe_allow_html=True)
        rc[1].markdown(f"<p style='color:#c0c0ff;font-family:\"Syne\",sans-serif;font-weight:600;padding:0.4rem 0'>{lrv}</p>", unsafe_allow_html=True)
        rc[2].markdown(f"<p style='color:#7bffcc;font-family:\"Syne\",sans-serif;font-weight:600;padding:0.4rem 0'>{xgbv}</p>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:0;border-color:#1a1a2e'>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric bar chart ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">MAE & RMSE Comparison</p>', unsafe_allow_html=True)
        metrics_fig = go.Figure()
        metrics_fig.add_trace(go.Bar(
            name="Linear Regression",
            x=["MAE", "RMSE"],
            y=[lr_r["mae"], lr_r["rmse"]],
            marker_color="#7b7bff",
            text=[f"${lr_r['mae']:.1f}", f"${lr_r['rmse']:.1f}"],
            textposition="outside",
        ))
        metrics_fig.add_trace(go.Bar(
            name="XGBoost",
            x=["MAE", "RMSE"],
            y=[xgb_r["mae"], xgb_r["rmse"]],
            marker_color="#7bffcc",
            text=[f"${xgb_r['mae']:.1f}", f"${xgb_r['rmse']:.1f}"],
            textposition="outside",
        ))
        lay_m = copy.deepcopy(PLOT_LAYOUT)
        lay_m.update(dict(title="Error in USD (lower = better)", barmode="group", height=320))
        metrics_fig.update_layout(**lay_m)
        st.plotly_chart(metrics_fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">R² Score Comparison</p>', unsafe_allow_html=True)
        r2_fig = go.Figure(go.Bar(
            x=["Linear Regression", "XGBoost"],
            y=[lr_r["r2"], xgb_r["r2"]],
            marker_color=["#7b7bff", "#7bffcc"],
            text=[f"{lr_r['r2']:.4f}", f"{xgb_r['r2']:.4f}"],
            textposition="outside",
            textfont=dict(color="#c0c0ff"),
        ))
        lay_r2 = copy.deepcopy(PLOT_LAYOUT)
        lay_r2.update(dict(title="R² Score (higher = better)", height=320,
                           yaxis=dict(range=[0, 1.1], gridcolor="#1a1a2e")))
        r2_fig.update_layout(**lay_r2)
        st.plotly_chart(r2_fig, use_container_width=True)

    # ── Actual vs Predicted scatter ────────────────────────────────────────────
    st.markdown('<p class="section-header" style="margin-top:0.5rem">Actual vs Predicted — XGBoost</p>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=xgb_r["y_test"],
        y=xgb_r["y_pred"],
        mode="markers",
        marker=dict(color="#6060c8", opacity=0.45, size=5),
        name="Predictions",
    ))
    max_val = max(xgb_r["y_test"].max(), xgb_r["y_pred"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#ff7bbd", dash="dot", width=1.5),
        name="Perfect fit",
    ))
    lay_s = copy.deepcopy(PLOT_LAYOUT)
    lay_s.update(dict(
        title="First 300 test samples — XGBoost predictions vs actual prices",
        height=380,
        xaxis=dict(title="Actual Price ($)", gridcolor="#1a1a2e", tickfont=dict(color="#6060a0")),
        yaxis=dict(title="Predicted Price ($)", gridcolor="#1a1a2e", tickfont=dict(color="#6060a0")),
    ))
    fig_scatter.update_layout(**lay_s)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Methodology note ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Methodology</p>', unsafe_allow_html=True)
    notes = [
        ("Features used", "room_type, neighbourhood_group, neighbourhood, minimum_nights, number_of_reviews, reviews_per_month, host_listings_count, availability_365"),
        ("Target variable", "Normalised price (z-score). Converted back to USD for metric display using mean=$106, std=$64."),
        ("Outlier removal", "IQR method — prices above Q3 + 1.5×IQR (≈$334) were removed before modelling."),
        ("Train / Test split", "80 % train · 20 % test · random_state=42"),
        ("XGBoost config", "n_estimators=300 · learning_rate=0.05 · max_depth=8 · random_state=42"),
    ]
    for k, v in notes:
        st.markdown(f"""
        <div class="insight-card">
            <p><strong>{k}</strong> — {v}</p>
        </div>
        """, unsafe_allow_html=True)


render()






# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import copy
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from utils import get_enriched_df, PLOT_LAYOUT
# from pages.predictor import train_models

# PRICE_MEAN = 106.0
# PRICE_STD  = 64.0

# @st.cache_data
# def evaluate_models():
#     df = get_enriched_df()
#     lr, xgb, rf, le_room, le_hood, le_boro, feature_cols = train_models()
#     df2 = df.copy()
#     df2["room_enc"] = le_room.transform(df2["room_type"])
#     df2["hood_enc"] = le_hood.transform(df2["neighbourhood"])
#     df2["boro_enc"] = le_boro.transform(df2["neighbourhood_group"])

#     X = df2[feature_cols]
#     y = df2["price"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     results = {}
#     for name, model in [("Linear Regression", lr), ("Random Forest", rf), ("XGBoost", xgb)]:
#         y_pred     = model.predict(X_test)
#         y_test_usd = y_test * PRICE_STD + PRICE_MEAN
#         y_pred_usd = y_pred * PRICE_STD + PRICE_MEAN
#         results[name] = {
#             "mae":    mean_absolute_error(y_test_usd, y_pred_usd),
#             "mse":    mean_squared_error(y_test_usd,  y_pred_usd),
#             "rmse":   np.sqrt(mean_squared_error(y_test_usd, y_pred_usd)),
#             "r2":     r2_score(y_test_usd, y_pred_usd),
#             "y_test": y_test_usd.values[:300],
#             "y_pred": y_pred_usd[:300],
#         }
#     return results


# def render():
#     st.markdown("<style>[data-testid='stSidebarNav']{display:none!important}</style>", unsafe_allow_html=True)

#     st.markdown("""
#     <div style="padding:2rem 0 1rem">
#         <p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
#             MODEL EVALUATION
#         </p>
#         <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.4rem;
#                    font-weight:800;margin:0;letter-spacing:-0.03em">
#             Model <span style="color:#6060c8">Metrics</span>
#         </h1>
#         <p style="color:#5050a0;font-size:0.9rem;margin-top:0.8rem">
#             Linear Regression vs Random Forest vs XGBoost · 80/20 split · 48,360 listings
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("---")

#     with st.spinner("Evaluating models…"):
#         results = evaluate_models()

#     lr_r  = results["Linear Regression"]
#     rf_r  = results["Random Forest"]
#     xgb_r = results["XGBoost"]

#     st.markdown("### Performance Summary")
#     head = st.columns([2, 1, 1, 1])
#     head[0].markdown("<p style='color:#4040a0;font-size:0.78rem;letter-spacing:0.08em'>METRIC</p>", unsafe_allow_html=True)
#     head[1].markdown("<p style='color:#7b7bff;font-size:0.78rem;letter-spacing:0.08em'>LINEAR REG</p>", unsafe_allow_html=True)
#     head[2].markdown("<p style='color:#ffd07b;font-size:0.78rem;letter-spacing:0.08em'>RANDOM FOREST</p>", unsafe_allow_html=True)
#     head[3].markdown("<p style='color:#7bffcc;font-size:0.78rem;letter-spacing:0.08em'>XGBOOST</p>", unsafe_allow_html=True)

#     for label, lrv, rfv, xgbv in [
#         ("MAE (USD)",  f"${lr_r['mae']:.2f}",  f"${rf_r['mae']:.2f}",  f"${xgb_r['mae']:.2f}"),
#         ("MSE",        f"{lr_r['mse']:,.1f}",   f"{rf_r['mse']:,.1f}",  f"{xgb_r['mse']:,.1f}"),
#         ("RMSE (USD)", f"${lr_r['rmse']:.2f}",  f"${rf_r['rmse']:.2f}", f"${xgb_r['rmse']:.2f}"),
#         ("R² Score",   f"{lr_r['r2']:.4f}",     f"{rf_r['r2']:.4f}",    f"{xgb_r['r2']:.4f}"),
#     ]:
#         rc = st.columns([2, 1, 1, 1])
#         rc[0].markdown(f"<p style='color:#9090c0;padding:0.4rem 0'>{label}</p>", unsafe_allow_html=True)
#         rc[1].markdown(f"<p style='color:#c0c0ff;font-weight:600;padding:0.4rem 0'>{lrv}</p>", unsafe_allow_html=True)
#         rc[2].markdown(f"<p style='color:#ffd07b;font-weight:600;padding:0.4rem 0'>{rfv}</p>", unsafe_allow_html=True)
#         rc[3].markdown(f"<p style='color:#7bffcc;font-weight:600;padding:0.4rem 0'>{xgbv}</p>", unsafe_allow_html=True)
#         st.markdown("<hr style='margin:0;border-color:#1a1a2e'>", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown('<p class="section-header">MAE & RMSE Comparison</p>', unsafe_allow_html=True)
#         fig_m = go.Figure()
#         for name, r, color in [
#             ("Linear Regression", lr_r,  "#7b7bff"),
#             ("Random Forest",     rf_r,  "#ffd07b"),
#             ("XGBoost",           xgb_r, "#7bffcc"),
#         ]:
#             fig_m.add_trace(go.Bar(
#                 name=name, x=["MAE", "RMSE"],
#                 y=[r["mae"], r["rmse"]],
#                 marker_color=color,
#                 text=[f"${r['mae']:.1f}", f"${r['rmse']:.1f}"],
#                 textposition="outside",
#             ))
#         lay_m = copy.deepcopy(PLOT_LAYOUT)
#         lay_m.update(dict(title="Error in USD (lower = better)", barmode="group", height=320))
#         fig_m.update_layout(**lay_m)
#         st.plotly_chart(fig_m, use_container_width=True)

#     with col2:
#         st.markdown('<p class="section-header">R² Score Comparison</p>', unsafe_allow_html=True)
#         fig_r2 = go.Figure(go.Bar(
#             x=["Linear Regression", "Random Forest", "XGBoost"],
#             y=[lr_r["r2"], rf_r["r2"], xgb_r["r2"]],
#             marker_color=["#7b7bff", "#ffd07b", "#7bffcc"],
#             text=[f"{lr_r['r2']:.4f}", f"{rf_r['r2']:.4f}", f"{xgb_r['r2']:.4f}"],
#             textposition="outside",
#         ))
#         lay_r2 = copy.deepcopy(PLOT_LAYOUT)
#         lay_r2.update(dict(title="R² Score (higher = better)", height=320,
#                            yaxis=dict(range=[0, 1.1], gridcolor="#1a1a2e")))
#         fig_r2.update_layout(**lay_r2)
#         st.plotly_chart(fig_r2, use_container_width=True)

#     st.markdown('<p class="section-header" style="margin-top:0.5rem">Actual vs Predicted — XGBoost</p>', unsafe_allow_html=True)
#     fig_s = go.Figure()
#     fig_s.add_trace(go.Scatter(
#         x=xgb_r["y_test"], y=xgb_r["y_pred"], mode="markers",
#         marker=dict(color="#6060c8", opacity=0.45, size=5), name="Predictions",
#     ))
#     max_val = max(xgb_r["y_test"].max(), xgb_r["y_pred"].max())
#     fig_s.add_trace(go.Scatter(
#         x=[0, max_val], y=[0, max_val], mode="lines",
#         line=dict(color="#ff7bbd", dash="dot", width=1.5), name="Perfect fit",
#     ))
#     lay_s = copy.deepcopy(PLOT_LAYOUT)
#     lay_s.update(dict(
#         title="First 300 test samples — XGBoost predictions vs actual",
#         height=380,
#         xaxis=dict(title="Actual Price ($)", gridcolor="#1a1a2e", tickfont=dict(color="#6060a0")),
#         yaxis=dict(title="Predicted Price ($)", gridcolor="#1a1a2e", tickfont=dict(color="#6060a0")),
#     ))
#     fig_s.update_layout(**lay_s)
#     st.plotly_chart(fig_s, use_container_width=True)

#     st.markdown("---")
#     st.markdown('<p class="section-header">Methodology</p>', unsafe_allow_html=True)
#     for k, v in [
#         ("Features used", "room_type, neighbourhood_group, neighbourhood, minimum_nights, number_of_reviews, reviews_per_month, host_listings_count, availability_365"),
#         ("Target variable", "Normalised price (z-score) → converted back to USD (mean=$106, std=$64)"),
#         ("Outlier removal", "IQR method — prices above ~$334/night removed before modelling"),
#         ("Train / Test split", "80% train · 20% test · random_state=42"),
#         ("XGBoost config", "n_estimators=300 · learning_rate=0.05 · max_depth=8 · random_state=42"),
#         ("Random Forest config", "n_estimators=200 · max_depth=10 · random_state=42 · n_jobs=-1"),
#     ]:
#         st.markdown(f'<div class="insight-card"><p><strong>{k}</strong> — {v}</p></div>', unsafe_allow_html=True)

# render()
