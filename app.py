import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import get_enriched_df, BOROUGH_COLORS, ROOM_COLORS, PLOT_LAYOUT
import copy

def render():
    df = get_enriched_df()

    st.markdown("""
    <div style="padding:2rem 0 1rem">
        <p style="color:#5050a0;font-family:'DM Sans',sans-serif;font-size:0.8rem;
                  letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
            NYC AIRBNB · 2019 DATASET
        </p>
        <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.6rem;
                   font-weight:800;margin:0;letter-spacing:-0.03em;line-height:1.1">
            Price Intelligence<br>
            <span style="color:#6060c8">Dashboard</span>
        </h1>
        <p style="color:#5050a0;font-size:0.9rem;margin-top:0.8rem">
            48,360 listings · 5 boroughs · 3 room types · Machine-learning price prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Listings",    f"{len(df):,}")
    k2.metric("Avg Price / Night", f"${df['price_orig'].mean():.0f}")
    k3.metric("Median Price",      f"${df['price_orig'].median():.0f}")
    k4.metric("Boroughs",          "5")
    k5.metric("Neighbourhoods",    f"{df['neighbourhood'].nunique()}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<p class="section-header">Avg Price by Borough</p>', unsafe_allow_html=True)
        borough_avg = (
            df.groupby("neighbourhood_group")["price_orig"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        colors = [BOROUGH_COLORS.get(b, "#7b7bff") for b in borough_avg["neighbourhood_group"]]
        fig_bar = go.Figure(go.Bar(
            x=borough_avg["price_orig"],
            y=borough_avg["neighbourhood_group"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=borough_avg["price_orig"].apply(lambda x: f"${x:.0f}"),
            textposition="outside",
            textfont=dict(color="#c0c0ff", size=12),
        ))
        lay = copy.deepcopy(PLOT_LAYOUT)
        lay.update(dict(
            title="Average Nightly Price (USD)",
            height=280,
            yaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e",
                       tickfont=dict(color="#9090c0", size=12)),
            xaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e",
                       tickfont=dict(color="#6060a0")),
        ))
        fig_bar.update_layout(**lay)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Room Type Share</p>', unsafe_allow_html=True)
        room_counts = df["room_type"].value_counts().reset_index()
        room_counts.columns = ["room_type", "count"]
        fig_pie = go.Figure(go.Pie(
            labels=room_counts["room_type"],
            values=room_counts["count"],
            hole=0.62,
            marker=dict(colors=[ROOM_COLORS.get(r, "#7b7bff") for r in room_counts["room_type"]],
                        line=dict(color="#0a0a12", width=2)),
            textfont=dict(family="DM Sans", color="#ffffff", size=11),
        ))
        lay2 = copy.deepcopy(PLOT_LAYOUT)
        lay2.update(dict(
            title="Listings by Room Type",
            height=280,
            showlegend=True,
        ))
        fig_pie.update_layout(**lay2)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<p class="section-header" style="margin-top:0.5rem">Price Distribution by Borough</p>', unsafe_allow_html=True)
    fig_box = go.Figure()
    for borough, color in BOROUGH_COLORS.items():
        sub = df[df["neighbourhood_group"] == borough]["price_orig"]
        fig_box.add_trace(go.Box(
            y=sub,
            name=borough,
            marker_color=color,
            line_color=color,
            boxpoints=False,
        ))
    lay3 = copy.deepcopy(PLOT_LAYOUT)
    lay3.update(dict(title="Nightly Price Distribution (USD) — post outlier removal", height=340))
    fig_box.update_layout(**lay3)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<p class="section-header" style="margin-top:0.5rem">Listing Density Map</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Geographic spread of 48 k listings across the five boroughs</p>', unsafe_allow_html=True)

    sample = df.sample(min(6000, len(df)), random_state=42)

    fig_map = px.scatter_mapbox(
        sample,
        lat="latitude", lon="longitude",
        color="neighbourhood_group",
        color_discrete_map=BOROUGH_COLORS,
        size="price_orig",
        size_max=10,
        opacity=0.55,
        zoom=9.5,
        center=dict(lat=40.7128, lon=-74.006),
        mapbox_style="carto-darkmatter",
        hover_data={"price_orig": True, "room_type": True,
                    "neighbourhood": True, "latitude": False, "longitude": False},
        labels={"neighbourhood_group": "Borough", "price_orig": "Price ($)"},
    )
    fig_map.update_layout(
        paper_bgcolor="#0a0a12",
        plot_bgcolor="#0a0a12",
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        legend=dict(bgcolor="#0a0a12", bordercolor="#1e1e2e",
                    font=dict(color="#9090c0"), orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_map, use_container_width=True)
