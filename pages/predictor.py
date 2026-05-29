# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from utils import get_enriched_df

# PRICE_MEAN = 106.0
# PRICE_STD  = 64.0

# @st.cache_resource
# def train_models():
#     df = get_enriched_df()
#     le_room  = LabelEncoder()
#     le_hood  = LabelEncoder()
#     le_boro  = LabelEncoder()
#     df["room_enc"]  = le_room.fit_transform(df["room_type"])
#     df["hood_enc"]  = le_hood.fit_transform(df["neighbourhood"])
#     df["boro_enc"]  = le_boro.fit_transform(df["neighbourhood_group"])

#     feature_cols = [
#         "room_enc", "boro_enc", "hood_enc",
#         "minimum_nights", "number_of_reviews",
#         "reviews_per_month", "calculated_host_listings_count", "availability_365",
#     ]
#     X = df[feature_cols]
#     y = df["price"]   # normalised

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     lr = LinearRegression()
#     lr.fit(X_train, y_train)

#     xgb = XGBRegressor(n_estimators=300, learning_rate=0.05,
#                         max_depth=8, random_state=42,
#                         verbosity=0)
#     xgb.fit(X_train, y_train)

#     return lr, xgb, le_room, le_hood, le_boro, feature_cols


# def render():
#     df = get_enriched_df()
#     lr, xgb, le_room, le_hood, le_boro, feature_cols = train_models()

#     st.markdown("""
#     <div style="padding:2rem 0 1rem">
#         <p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
#             ML INFERENCE
#         </p>
#         <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.4rem;
#                    font-weight:800;margin:0;letter-spacing:-0.03em">
#             Price <span style="color:#6060c8">Predictor</span>
#         </h1>
#         <p style="color:#5050a0;font-size:0.9rem;margin-top:0.8rem">
#             Configure your listing below and get an instant price estimate.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("---")

#     col_form, col_result = st.columns([2, 1])

#     with col_form:
#         st.markdown("### Listing Details")

#         room_type = st.selectbox(
#             "Room Type",
#             options=list(le_room.classes_),
#         )

#         borough = st.selectbox(
#             "Borough (Neighbourhood Group)",
#             options=list(le_boro.classes_),
#         )

#         neighbourhood_options = sorted(
#             df[df["neighbourhood_group"] == borough]["neighbourhood"].unique()
#         )
#         neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

#         c1, c2 = st.columns(2)
#         with c1:
#             min_nights = st.slider("Minimum Nights", 1, 30, 3)
#             num_reviews = st.slider("Number of Reviews", 0, 200, 25)
#         with c2:
#             reviews_pm = st.slider("Reviews per Month", 0.0, 10.0, 1.5, step=0.1)
#             host_listings = st.slider("Host Listings Count", 1, 50, 2)

#         availability = st.slider("Availability (days/year)", 0, 365, 150)

#         model_choice = st.radio("Model", ["XGBoost (recommended)", "Linear Regression"],
#                                  horizontal=True)

#         predict_clicked = st.button("🔮  Predict Price", use_container_width=True)

#     with col_result:
#         st.markdown("### Estimate")
#         if predict_clicked:
#             # Encode inputs
#             try:
#                 room_enc = le_room.transform([room_type])[0]
#             except Exception:
#                 room_enc = 0
#             try:
#                 boro_enc = le_boro.transform([borough])[0]
#             except Exception:
#                 boro_enc = 0
#             try:
#                 hood_enc = le_hood.transform([neighbourhood])[0]
#             except Exception:
#                 hood_enc = 0

#             # Normalise scalar inputs using dataset stats
#             df_stats = df[["minimum_nights","number_of_reviews",
#                            "reviews_per_month","calculated_host_listings_count",
#                            "availability_365"]].agg(["mean","std"])

#             def norm(val, col):
#                 m = df_stats.loc["mean", col]
#                 s = df_stats.loc["std",  col]
#                 return (val - m) / s if s != 0 else 0.0

#             row = pd.DataFrame([{
#                 "room_enc":  room_enc,
#                 "boro_enc":  boro_enc,
#                 "hood_enc":  hood_enc,
#                 "minimum_nights":                   norm(min_nights, "minimum_nights"),
#                 "number_of_reviews":                norm(num_reviews, "number_of_reviews"),
#                 "reviews_per_month":                norm(reviews_pm, "reviews_per_month"),
#                 "calculated_host_listings_count":   norm(host_listings, "calculated_host_listings_count"),
#                 "availability_365":                 norm(availability, "availability_365"),
#             }])

#             model = xgb if "XGBoost" in model_choice else lr
#             pred_scaled = model.predict(row)[0]
#             pred_price  = pred_scaled * PRICE_STD + PRICE_MEAN
#             pred_price  = max(10.0, pred_price)

#             weekly  = pred_price * 7
#             monthly = pred_price * 30

#             st.markdown(f"""
#             <div class="pred-box">
#                 <h2>Estimated Nightly Rate</h2>
#                 <h1>${pred_price:.0f}</h1>
#                 <p>USD per night · {model_choice.split(" ")[0]}</p>
#             </div>
#             """, unsafe_allow_html=True)

#             st.markdown("<br>", unsafe_allow_html=True)

#             s1, s2 = st.columns(2)
#             s1.metric("Weekly Revenue",  f"${weekly:,.0f}")
#             s2.metric("Monthly Revenue", f"${monthly:,.0f}")

#             # How it compares
#             boro_avg  = df[df["neighbourhood_group"] == borough]["price_orig"].mean()
#             room_avg  = df[df["room_type"] == room_type]["price_orig"].mean()
#             diff_boro = pred_price - boro_avg
#             diff_room = pred_price - room_avg

#             st.markdown("<br>", unsafe_allow_html=True)
#             st.markdown("""<p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.08em;
#                             text-transform:uppercase">Context</p>""", unsafe_allow_html=True)
#             st.markdown(f"""
#             <div class="insight-card">
#                 <p>Borough avg ({borough}): <strong>${boro_avg:.0f}</strong> — 
#                 your estimate is <strong>{"+" if diff_boro>=0 else ""}{diff_boro:.0f}</strong> vs borough avg.</p>
#             </div>
#             <div class="insight-card">
#                 <p>Room-type avg ({room_type}): <strong>${room_avg:.0f}</strong> — 
#                 your estimate is <strong>{"+" if diff_room>=0 else ""}{diff_room:.0f}</strong> vs room avg.</p>
#             </div>
#             """, unsafe_allow_html=True)

#         else:
#             st.markdown("""
#             <div style="background:#0f0f1a;border:1px dashed #2a2a40;border-radius:16px;
#                         padding:2rem;text-align:center;margin-top:1rem">
#                 <div style="font-size:2.5rem">🔮</div>
#                 <p style="color:#4040a0;font-size:0.9rem;margin-top:0.8rem">
#                     Fill in the listing details<br>and click <strong style="color:#6060c8">Predict Price</strong>
#                 </p>
#             </div>
#             """, unsafe_allow_html=True)


# render()







import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from utils import get_enriched_df

PRICE_MEAN = 106.0
PRICE_STD  = 64.0

@st.cache_resource
def train_models():
    df = get_enriched_df()
    le_room = LabelEncoder()
    le_hood = LabelEncoder()
    le_boro = LabelEncoder()
    df["room_enc"] = le_room.fit_transform(df["room_type"])
    df["hood_enc"] = le_hood.fit_transform(df["neighbourhood"])
    df["boro_enc"] = le_boro.fit_transform(df["neighbourhood_group"])

    feature_cols = ["room_enc", "boro_enc", "hood_enc", "minimum_nights",
                    "number_of_reviews", "reviews_per_month",
                    "calculated_host_listings_count", "availability_365"]
    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return lr, xgb, rf, le_room, le_hood, le_boro, feature_cols


def render():
    st.markdown("<style>[data-testid='stSidebarNav']{display:none!important}</style>", unsafe_allow_html=True)

    df = get_enriched_df()
    lr, xgb, rf, le_room, le_hood, le_boro, feature_cols = train_models()

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
            Configure your listing and get an instant price estimate.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    col_form, col_result = st.columns([2, 1])

    with col_form:
        st.markdown("### Listing Details")
        room_type = st.selectbox("Room Type", options=list(le_room.classes_))
        borough   = st.selectbox("Borough", options=list(le_boro.classes_))
        neighbourhood_options = sorted(df[df["neighbourhood_group"] == borough]["neighbourhood"].unique())
        neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

        c1, c2 = st.columns(2)
        with c1:
            min_nights  = st.slider("Minimum Nights", 1, 30, 3)
            num_reviews = st.slider("Number of Reviews", 0, 200, 25)
        with c2:
            reviews_pm    = st.slider("Reviews per Month", 0.0, 10.0, 1.5, step=0.1)
            host_listings = st.slider("Host Listings Count", 1, 50, 2)

        availability = st.slider("Availability (days/year)", 0, 365, 150)

        model_choice = st.radio(
            "Model",
            ["XGBoost (recommended)", "Random Forest", "Linear Regression"],
            horizontal=True
        )

        predict_clicked = st.button("🔮  Predict Price", use_container_width=True)

    with col_result:
        st.markdown("### Estimate")
        if predict_clicked:
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

            df_stats = df[["minimum_nights", "number_of_reviews", "reviews_per_month",
                           "calculated_host_listings_count", "availability_365"]].agg(["mean", "std"])

            def norm(val, col):
                m = df_stats.loc["mean", col]
                s = df_stats.loc["std",  col]
                return (val - m) / s if s != 0 else 0.0

            row = pd.DataFrame([{
                "room_enc": room_enc,
                "boro_enc": boro_enc,
                "hood_enc": hood_enc,
                "minimum_nights":                 norm(min_nights,    "minimum_nights"),
                "number_of_reviews":              norm(num_reviews,   "number_of_reviews"),
                "reviews_per_month":              norm(reviews_pm,    "reviews_per_month"),
                "calculated_host_listings_count": norm(host_listings, "calculated_host_listings_count"),
                "availability_365":               norm(availability,  "availability_365"),
            }])

            if "XGBoost" in model_choice:
                model = xgb
            elif "Random Forest" in model_choice:
                model = rf
            else:
                model = lr

            pred_price = max(10.0, model.predict(row)[0] * PRICE_STD + PRICE_MEAN)

            st.markdown(f"""
            <div class="pred-box">
                <h2>Estimated Nightly Rate</h2>
                <h1>${pred_price:.0f}</h1>
                <p>USD per night · {model_choice.split(" ")[0]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.metric("Weekly Revenue",  f"${pred_price * 7:,.0f}")
            s2.metric("Monthly Revenue", f"${pred_price * 30:,.0f}")

            boro_avg = df[df["neighbourhood_group"] == borough]["price_orig"].mean()
            room_avg = df[df["room_type"] == room_type]["price_orig"].mean()
            diff_b   = pred_price - boro_avg
            diff_r   = pred_price - room_avg

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='color:#5050a0;font-size:0.8rem;letter-spacing:0.08em;text-transform:uppercase'>Context</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="insight-card">
                <p>Borough avg ({borough}): <strong>${boro_avg:.0f}</strong> —
                your estimate is <strong>{"+" if diff_b >= 0 else ""}{diff_b:.0f}</strong> vs borough avg.</p>
            </div>
            <div class="insight-card">
                <p>Room-type avg ({room_type}): <strong>${room_avg:.0f}</strong> —
                your estimate is <strong>{"+" if diff_r >= 0 else ""}{diff_r:.0f}</strong> vs room avg.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#0f0f1a;border:1px dashed #2a2a40;border-radius:16px;
                        padding:2rem;text-align:center;margin-top:1rem">
                <div style="font-size:2.5rem">🔮</div>
                <p style="color:#4040a0;font-size:0.9rem;margin-top:0.8rem">
                    Fill in the listing details<br>and click
                    <strong style="color:#6060c8">Predict Price</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

render()



# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from utils import get_enriched_df, BOROUGH_COLORS, ROOM_COLORS, PLOT_LAYOUT
# import copy

# def render():
#     st.markdown("<style>[data-testid='stSidebarNav']{display:none!important}</style>", unsafe_allow_html=True)

#     df = get_enriched_df()

#     st.markdown("""
#     <div style="padding:2rem 0 1rem">
#         <p style="color:#5050a0;font-family:'DM Sans',sans-serif;font-size:0.8rem;
#                   letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
#             NYC AIRBNB · 2019 DATASET
#         </p>
#         <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.6rem;
#                    font-weight:800;margin:0;letter-spacing:-0.03em;line-height:1.1">
#             Price Intelligence<br>
#             <span style="color:#6060c8">Dashboard</span>
#         </h1>
#         <p style="color:#5050a0;font-size:0.9rem;margin-top:0.8rem">
#             48,360 listings · 5 boroughs · 3 room types · Machine-learning price prediction
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("---")

#     k1, k2, k3, k4, k5 = st.columns(5)
#     k1.metric("Total Listings",    f"{len(df):,}")
#     k2.metric("Avg Price / Night", f"${df['price_orig'].mean():.0f}")
#     k3.metric("Median Price",      f"${df['price_orig'].median():.0f}")
#     k4.metric("Boroughs",          "5")
#     k5.metric("Neighbourhoods",    f"{df['neighbourhood'].nunique()}")

#     st.markdown("<br>", unsafe_allow_html=True)

#     col1, col2 = st.columns([3, 2])

#     with col1:
#         st.markdown('<p class="section-header">Avg Price by Borough</p>', unsafe_allow_html=True)
#         borough_avg = (
#             df.groupby("neighbourhood_group")["price_orig"]
#             .mean().sort_values(ascending=True).reset_index()
#         )
#         colors = [BOROUGH_COLORS.get(b, "#7b7bff") for b in borough_avg["neighbourhood_group"]]
#         fig_bar = go.Figure(go.Bar(
#             x=borough_avg["price_orig"],
#             y=borough_avg["neighbourhood_group"],
#             orientation="h",
#             marker=dict(color=colors, line=dict(width=0)),
#             text=borough_avg["price_orig"].apply(lambda x: f"${x:.0f}"),
#             textposition="outside",
#             textfont=dict(color="#c0c0ff", size=12),
#         ))
#         lay = copy.deepcopy(PLOT_LAYOUT)
#         lay.update(dict(
#             title="Average Nightly Price (USD)", height=280,
#             yaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#9090c0", size=12)),
#             xaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#6060a0")),
#         ))
#         fig_bar.update_layout(**lay)
#         st.plotly_chart(fig_bar, use_container_width=True)

#     with col2:
#         st.markdown('<p class="section-header">Room Type Share</p>', unsafe_allow_html=True)
#         room_counts = df["room_type"].value_counts().reset_index()
#         room_counts.columns = ["room_type", "count"]
#         fig_pie = go.Figure(go.Pie(
#             labels=room_counts["room_type"],
#             values=room_counts["count"],
#             hole=0.62,
#             marker=dict(colors=[ROOM_COLORS.get(r, "#7b7bff") for r in room_counts["room_type"]],
#                         line=dict(color="#0a0a12", width=2)),
#             textfont=dict(family="DM Sans", color="#ffffff", size=11),
#         ))
#         lay2 = copy.deepcopy(PLOT_LAYOUT)
#         lay2.update(dict(title="Listings by Room Type", height=280, showlegend=True))
#         fig_pie.update_layout(**lay2)
#         st.plotly_chart(fig_pie, use_container_width=True)

#     st.markdown('<p class="section-header" style="margin-top:0.5rem">Price Distribution by Borough</p>', unsafe_allow_html=True)
#     fig_box = go.Figure()
#     for borough, color in BOROUGH_COLORS.items():
#         sub = df[df["neighbourhood_group"] == borough]["price_orig"]
#         fig_box.add_trace(go.Box(
#             y=sub,
#             name=borough,
#             marker_color=color,
#             line_color=color,
#             boxpoints=False,
#         ))
#     lay3 = copy.deepcopy(PLOT_LAYOUT)
#     lay3.update(dict(title="Nightly Price Distribution (USD) — post outlier removal", height=340))
#     fig_box.update_layout(**lay3)
#     st.plotly_chart(fig_box, use_container_width=True)

#     st.markdown('<p class="section-header" style="margin-top:0.5rem">Listing Density Map</p>', unsafe_allow_html=True)
#     st.markdown('<p class="section-sub">Geographic spread of 48 k listings across the five boroughs</p>', unsafe_allow_html=True)

#     sample = df.sample(min(6000, len(df)), random_state=42)
#     fig_map = px.scatter_mapbox(
#         sample,
#         lat="latitude", lon="longitude",
#         color="neighbourhood_group",
#         color_discrete_map=BOROUGH_COLORS,
#         size="price_orig",
#         size_max=10,
#         opacity=0.55,
#         zoom=9.5,
#         center=dict(lat=40.7128, lon=-74.006),
#         mapbox_style="carto-darkmatter",
#         hover_data={"price_orig": True, "room_type": True,
#                     "neighbourhood": True, "latitude": False, "longitude": False},
#         labels={"neighbourhood_group": "Borough", "price_orig": "Price ($)"},
#     )
#     fig_map.update_layout(
#         paper_bgcolor="#0a0a12", plot_bgcolor="#0a0a12",
#         margin=dict(l=0, r=0, t=0, b=0), height=420,
#         legend=dict(bgcolor="#0a0a12", bordercolor="#1e1e2e",
#                     font=dict(color="#9090c0"), orientation="h", y=-0.05),
#     )
#     st.plotly_chart(fig_map, use_container_width=True)

# render()
