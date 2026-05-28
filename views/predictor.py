import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from utils import get_enriched_df

PRICE_MEAN = 106.0
PRICE_STD  = 64.0

@st.cache_resource
def train_models():
    df = get_enriched_df()
    le_room  = LabelEncoder()
    le_hood  = LabelEncoder()
    le_boro  = LabelEncoder()
    df["room_enc"]  = le_room.fit_transform(df["room_type"])
    df["hood_enc"]  = le_hood.fit_transform(df["neighbourhood"])
    df["boro_enc"]  = le_boro.fit_transform(df["neighbourhood_group"])

    feature_cols = [
        "room_enc", "boro_enc", "hood_enc",
        "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
    ]
    X = df[feature_cols]
    y = df["price"]   # normalised

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=8, random_state=42,
                        verbosity=0)
    xgb.fit(X_train, y_train)

    return lr, xgb, le_room, le_hood, le_boro, feature_cols


def render():
    df = get_enriched_df()
    lr, xgb, le_room, le_hood, le_boro, feature_cols = train_models()

    st.markdown("""
    <div style="padding:2rem 0 1rem">
        <p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
            ML INFERENCE
        </p>
        <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.4rem;
                   font-weight:800;margin:0;letter-spacing:-0.03em">
            Price <span style="color:#6060c8">Predictor</span>
        </h1>
        <p style="color:#5050a0;font-size:0.9rem;margin-top:0.8rem">
            Configure your listing below and get an instant price estimate.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    col_form, col_result = st.columns([2, 1])

    with col_form:
        st.markdown("### Listing Details")

        room_type = st.selectbox(
            "Room Type",
            options=list(le_room.classes_),
        )

        borough = st.selectbox(
            "Borough (Neighbourhood Group)",
            options=list(le_boro.classes_),
        )

        neighbourhood_options = sorted(
            df[df["neighbourhood_group"] == borough]["neighbourhood"].unique()
        )
        neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

        c1, c2 = st.columns(2)
        with c1:
            min_nights = st.slider("Minimum Nights", 1, 30, 3)
            num_reviews = st.slider("Number of Reviews", 0, 200, 25)
        with c2:
            reviews_pm = st.slider("Reviews per Month", 0.0, 10.0, 1.5, step=0.1)
            host_listings = st.slider("Host Listings Count", 1, 50, 2)

        availability = st.slider("Availability (days/year)", 0, 365, 150)

        model_choice = st.radio("Model", ["XGBoost (recommended)", "Linear Regression"],
                                 horizontal=True)

        predict_clicked = st.button("🔮  Predict Price", use_container_width=True)

    with col_result:
        st.markdown("### Estimate")
        if predict_clicked:
            # Encode inputs
            try:
                room_enc = le_room.transform([room_type])[0]
            except Exception:
                room_enc = 0
            try:
                boro_enc = le_boro.transform([borough])[0]
            except Exception:
                boro_enc = 0
            try:
                hood_enc = le_hood.transform([neighbourhood])[0]
            except Exception:
                hood_enc = 0

            # Normalise scalar inputs using dataset stats
            df_stats = df[["minimum_nights","number_of_reviews",
                           "reviews_per_month","calculated_host_listings_count",
                           "availability_365"]].agg(["mean","std"])

            def norm(val, col):
                m = df_stats.loc["mean", col]
                s = df_stats.loc["std",  col]
                return (val - m) / s if s != 0 else 0.0

            row = pd.DataFrame([{
                "room_enc":  room_enc,
                "boro_enc":  boro_enc,
                "hood_enc":  hood_enc,
                "minimum_nights":                   norm(min_nights, "minimum_nights"),
                "number_of_reviews":                norm(num_reviews, "number_of_reviews"),
                "reviews_per_month":                norm(reviews_pm, "reviews_per_month"),
                "calculated_host_listings_count":   norm(host_listings, "calculated_host_listings_count"),
                "availability_365":                 norm(availability, "availability_365"),
            }])

            model = xgb if "XGBoost" in model_choice else lr
            pred_scaled = model.predict(row)[0]
            pred_price  = pred_scaled * PRICE_STD + PRICE_MEAN
            pred_price  = max(10.0, pred_price)

            weekly  = pred_price * 7
            monthly = pred_price * 30

            st.markdown(f"""
            <div class="pred-box">
                <h2>Estimated Nightly Rate</h2>
                <h1>${pred_price:.0f}</h1>
                <p>USD per night · {model_choice.split(" ")[0]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            s1.metric("Weekly Revenue",  f"${weekly:,.0f}")
            s2.metric("Monthly Revenue", f"${monthly:,.0f}")

            # How it compares
            boro_avg  = df[df["neighbourhood_group"] == borough]["price_orig"].mean()
            room_avg  = df[df["room_type"] == room_type]["price_orig"].mean()
            diff_boro = pred_price - boro_avg
            diff_room = pred_price - room_avg

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.08em;
                            text-transform:uppercase">Context</p>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="insight-card">
                <p>Borough avg ({borough}): <strong>${boro_avg:.0f}</strong> — 
                your estimate is <strong>{"+" if diff_boro>=0 else ""}{diff_boro:.0f}</strong> vs borough avg.</p>
            </div>
            <div class="insight-card">
                <p>Room-type avg ({room_type}): <strong>${room_avg:.0f}</strong> — 
                your estimate is <strong>{"+" if diff_room>=0 else ""}{diff_room:.0f}</strong> vs room avg.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#0f0f1a;border:1px dashed #2a2a40;border-radius:16px;
                        padding:2rem;text-align:center;margin-top:1rem">
                <div style="font-size:2.5rem">🔮</div>
                <p style="color:#4040a0;font-size:0.9rem;margin-top:0.8rem">
                    Fill in the listing details<br>and click <strong style="color:#6060c8">Predict Price</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)



render()

