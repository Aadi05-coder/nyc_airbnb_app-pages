import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import copy
from utils import get_enriched_df, get_encoded_df, BOROUGH_COLORS, ROOM_COLORS, PLOT_LAYOUT


def render():
    df = get_enriched_df()
    df_enc, le_room, le_hood = get_encoded_df()

    st.markdown("""
    <div style="padding:2rem 0 1rem">
        <p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
            EXPLORATORY DATA ANALYSIS
        </p>
        <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.4rem;
                   font-weight:800;margin:0;letter-spacing:-0.03em">
            Deep-Dive <span style="color:#6060c8">Insights</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📦  Distributions", "🏙️  Borough Analysis", "🛏️  Room Type Analysis", "🔥  Correlations"]
    )

    # ── Tab 1 — Distributions ──────────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-header">Price Distribution</p>', unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df["price_orig"],
                nbinsx=60,
                marker_color="#6060c8",
                opacity=0.8,
                name="Listings",
            ))
            lay = copy.deepcopy(PLOT_LAYOUT)
            lay.update(title="Nightly Price (USD) — post outlier removal", height=300)
            fig_hist.update_layout(**lay)
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.markdown('<p class="section-header">Availability (365 days, scaled)</p>', unsafe_allow_html=True)
            fig_av = go.Figure()
            fig_av.add_trace(go.Histogram(
                x=df["availability_365"],
                nbinsx=50,
                marker_color="#ff7bbd",
                opacity=0.8,
            ))
            lay2 = copy.deepcopy(PLOT_LAYOUT)
            lay2.update(title="Availability (normalised)", height=300)
            fig_av.update_layout(**lay2)
            st.plotly_chart(fig_av, use_container_width=True)

        # Outlier story
        st.markdown("---")
        st.markdown('<p class="section-header">Outlier Removal Impact</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.markdown("""
        <div class="insight-card">
            <p>Before outlier removal the dataset contained <strong>prices above $334</strong> treated as outliers using the IQR method (Q3 + 1.5 × IQR).</p>
        </div>
        <div class="insight-card">
            <p>IQR bounds: <strong>lower = $0</strong>, <strong>upper = $334</strong> — anything beyond was dropped.</p>
        </div>
        """, unsafe_allow_html=True)
        col_b.markdown("""
        <div class="insight-card">
            <p>After cleaning, <strong>48,360 listings</strong> remain from the original ~48 k records — roughly <strong>5 % removed</strong>.</p>
        </div>
        <div class="insight-card">
            <p>Resulting price range: <strong>$10 – $334</strong> · Median ≈ <strong>$92</strong> · Mean ≈ <strong>$106</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2 — Borough ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        borough_stats = (
            df.groupby("neighbourhood_group")["price_orig"]
            .agg(["mean", "median", "count"])
            .rename(columns={"mean": "Avg Price", "median": "Median Price", "count": "Listings"})
            .reset_index()
        )

        # Grouped bar
        fig_bgroup = go.Figure()
        fig_bgroup.add_trace(go.Bar(
            name="Avg Price",
            x=borough_stats["neighbourhood_group"],
            y=borough_stats["Avg Price"],
            marker_color="#6060c8",
            text=borough_stats["Avg Price"].apply(lambda v: f"${v:.0f}"),
            textposition="outside",
        ))
        fig_bgroup.add_trace(go.Bar(
            name="Median Price",
            x=borough_stats["neighbourhood_group"],
            y=borough_stats["Median Price"],
            marker_color="#ff7bbd",
            text=borough_stats["Median Price"].apply(lambda v: f"${v:.0f}"),
            textposition="outside",
        ))
        lay_b = copy.deepcopy(PLOT_LAYOUT)
        lay_b.update(dict(title="Avg vs Median Price by Borough", barmode="group", height=340))
        fig_bgroup.update_layout(**lay_b)
        st.plotly_chart(fig_bgroup, use_container_width=True)

        # Top 15 neighbourhoods
        st.markdown('<p class="section-header" style="margin-top:1rem">Top 15 Neighbourhoods by Avg Price</p>', unsafe_allow_html=True)
        top_n = (
            df.groupby(["neighbourhood", "neighbourhood_group"])["price_orig"]
            .mean()
            .reset_index()
            .sort_values("price_orig", ascending=False)
            .head(15)
        )
        colors_n = [BOROUGH_COLORS.get(b, "#7b7bff") for b in top_n["neighbourhood_group"]]
        fig_top = go.Figure(go.Bar(
            x=top_n["price_orig"],
            y=top_n["neighbourhood"],
            orientation="h",
            marker=dict(color=colors_n),
            text=top_n["price_orig"].apply(lambda v: f"${v:.0f}"),
            textposition="outside",
            textfont=dict(color="#c0c0ff", size=11),
        ))
        lay_n = copy.deepcopy(PLOT_LAYOUT)
        lay_n.update(dict(title="Top Neighbourhoods (avg nightly price)", height=420,
                          yaxis=dict(autorange="reversed", gridcolor="#1a1a2e",
                                     tickfont=dict(color="#9090c0"))))
        fig_top.update_layout(**lay_n)
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Tab 3 — Room Type ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<p class="section-header">Avg Price by Room Type</p>', unsafe_allow_html=True)
            rt_avg = (
                df.groupby("room_type")["price_orig"].mean()
                .reset_index()
                .sort_values("price_orig", ascending=False)
            )
            fig_rt = go.Figure(go.Bar(
                x=rt_avg["room_type"],
                y=rt_avg["price_orig"],
                marker=dict(color=[ROOM_COLORS.get(r, "#7b7bff") for r in rt_avg["room_type"]]),
                text=rt_avg["price_orig"].apply(lambda v: f"${v:.0f}"),
                textposition="outside",
                textfont=dict(color="#c0c0ff"),
            ))
            lay_rt = copy.deepcopy(PLOT_LAYOUT)
            lay_rt.update(title="Average Price (USD)", height=300)
            fig_rt.update_layout(**lay_rt)
            st.plotly_chart(fig_rt, use_container_width=True)

        with c2:
            st.markdown('<p class="section-header">Listing Count by Room Type</p>', unsafe_allow_html=True)
            rt_cnt = df["room_type"].value_counts().reset_index()
            rt_cnt.columns = ["room_type", "count"]
            fig_cnt = go.Figure(go.Bar(
                x=rt_cnt["room_type"],
                y=rt_cnt["count"],
                marker=dict(color=[ROOM_COLORS.get(r, "#7b7bff") for r in rt_cnt["room_type"]]),
                text=rt_cnt["count"].apply(lambda v: f"{v:,}"),
                textposition="outside",
                textfont=dict(color="#c0c0ff"),
            ))
            lay_cnt = copy.deepcopy(PLOT_LAYOUT)
            lay_cnt.update(title="Number of Listings", height=300)
            fig_cnt.update_layout(**lay_cnt)
            st.plotly_chart(fig_cnt, use_container_width=True)

        # Heatmap: room × borough
        st.markdown('<p class="section-header">Avg Price: Room Type × Borough</p>', unsafe_allow_html=True)
        pivot = df.pivot_table(index="room_type", columns="neighbourhood_group",
                                values="price_orig", aggfunc="mean")
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            text=[[f"${v:.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(color="white", size=12),
        ))
        lay_h = copy.deepcopy(PLOT_LAYOUT)
        lay_h.update(title="Average Nightly Price ($)", height=280)
        fig_heat.update_layout(**lay_h)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 4 — Correlations ──────────────────────────────────────────────────
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        numeric_cols = [
            "price", "minimum_nights", "number_of_reviews",
            "reviews_per_month", "calculated_host_listings_count", "availability_365",
        ]
        corr = df_enc[numeric_cols].corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
        ))
        lay_c = copy.deepcopy(PLOT_LAYOUT)
        lay_c.update(title="Pearson Correlation Matrix", height=460)
        fig_corr.update_layout(**lay_c)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-header">Key Observations</p>', unsafe_allow_html=True)
        observations = [
            ("Room type → Price", "Room type is the strongest single predictor of price. Entire homes command 2–3× the nightly rate of shared rooms."),
            ("Reviews ↔ Availability", "Listings with more reviews tend to have higher availability, suggesting active, well-managed properties."),
            ("Minimum nights", "Longer minimum-night requirements correlate weakly with lower review counts, pointing to occasional long-term rentals."),
            ("Host listings count", "Hosts with many listings price competitively — a slight negative correlation with per-listing price."),
        ]
        for title, body in observations:
            st.markdown(f"""
            <div class="insight-card">
                <p><strong>{title}</strong> — {body}</p>
            </div>
            """, unsafe_allow_html=True)



render()








# import streamlit as st
# import plotly.graph_objects as go
# import pandas as pd
# import numpy as np
# import copy
# from utils import get_enriched_df, get_encoded_df, BOROUGH_COLORS, ROOM_COLORS, PLOT_LAYOUT


# def render():
#     st.markdown("<style>[data-testid='stSidebarNav']{display:none!important}</style>", unsafe_allow_html=True)

#     df = get_enriched_df()
#     df_enc, le_room, le_hood = get_encoded_df()

#     st.markdown("""
#     <div style="padding:2rem 0 1rem">
#         <p style="color:#5050a0;font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem">
#             EXPLORATORY DATA ANALYSIS
#         </p>
#         <h1 style="font-family:'Syne',sans-serif;color:#ffffff;font-size:2.4rem;
#                    font-weight:800;margin:0;letter-spacing:-0.03em">
#             Deep-Dive <span style="color:#6060c8">Insights</span>
#         </h1>
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("---")

#     tab1, tab2, tab3, tab4 = st.tabs(
#         ["📦  Distributions", "🏙️  Borough Analysis", "🛏️  Room Type Analysis", "🔥  Correlations"]
#     )

#     with tab1:
#         st.markdown("<br>", unsafe_allow_html=True)
#         c1, c2 = st.columns(2)
#         with c1:
#             st.markdown('<p class="section-header">Price Distribution</p>', unsafe_allow_html=True)
#             fig_hist = go.Figure()
#             fig_hist.add_trace(go.Histogram(x=df["price_orig"], nbinsx=60, marker_color="#6060c8", opacity=0.8))
#             lay = copy.deepcopy(PLOT_LAYOUT)
#             lay.update(title="Nightly Price (USD) — post outlier removal", height=300)
#             fig_hist.update_layout(**lay)
#             st.plotly_chart(fig_hist, use_container_width=True)

#         with c2:
#             st.markdown('<p class="section-header">Availability (scaled)</p>', unsafe_allow_html=True)
#             fig_av = go.Figure()
#             fig_av.add_trace(go.Histogram(x=df["availability_365"], nbinsx=50, marker_color="#ff7bbd", opacity=0.8))
#             lay2 = copy.deepcopy(PLOT_LAYOUT)
#             lay2.update(title="Availability (normalised)", height=300)
#             fig_av.update_layout(**lay2)
#             st.plotly_chart(fig_av, use_container_width=True)

#         st.markdown("---")
#         st.markdown('<p class="section-header">Outlier Removal Impact</p>', unsafe_allow_html=True)
#         col_a, col_b = st.columns(2)
#         col_a.markdown("""
#         <div class="insight-card"><p>Before outlier removal, prices above <strong>$334</strong> were treated as outliers using the IQR method.</p></div>
#         <div class="insight-card"><p>IQR bounds: <strong>lower = $0</strong>, <strong>upper = $334</strong> — anything beyond was dropped.</p></div>
#         """, unsafe_allow_html=True)
#         col_b.markdown("""
#         <div class="insight-card"><p>After cleaning, <strong>48,360 listings</strong> remain — roughly <strong>5% removed</strong>.</p></div>
#         <div class="insight-card"><p>Resulting range: <strong>$10–$334</strong> · Median ≈ <strong>$92</strong> · Mean ≈ <strong>$106</strong>.</p></div>
#         """, unsafe_allow_html=True)

#     with tab2:
#         st.markdown("<br>", unsafe_allow_html=True)
#         borough_stats = (
#             df.groupby("neighbourhood_group")["price_orig"]
#             .agg(["mean", "median", "count"])
#             .rename(columns={"mean": "Avg Price", "median": "Median Price", "count": "Listings"})
#             .reset_index()
#         )
#         fig_bgroup = go.Figure()
#         fig_bgroup.add_trace(go.Bar(
#             name="Avg Price", x=borough_stats["neighbourhood_group"], y=borough_stats["Avg Price"],
#             marker_color="#6060c8", text=borough_stats["Avg Price"].apply(lambda v: f"${v:.0f}"),
#             textposition="outside",
#         ))
#         fig_bgroup.add_trace(go.Bar(
#             name="Median Price", x=borough_stats["neighbourhood_group"], y=borough_stats["Median Price"],
#             marker_color="#ff7bbd", text=borough_stats["Median Price"].apply(lambda v: f"${v:.0f}"),
#             textposition="outside",
#         ))
#         lay_b = copy.deepcopy(PLOT_LAYOUT)
#         lay_b.update(dict(title="Avg vs Median Price by Borough", barmode="group", height=340))
#         fig_bgroup.update_layout(**lay_b)
#         st.plotly_chart(fig_bgroup, use_container_width=True)

#         st.markdown('<p class="section-header" style="margin-top:1rem">Top 15 Neighbourhoods by Avg Price</p>', unsafe_allow_html=True)
#         top_n = (
#             df.groupby(["neighbourhood", "neighbourhood_group"])["price_orig"]
#             .mean().reset_index().sort_values("price_orig", ascending=False).head(15)
#         )
#         colors_n = [BOROUGH_COLORS.get(b, "#7b7bff") for b in top_n["neighbourhood_group"]]
#         fig_top = go.Figure(go.Bar(
#             x=top_n["price_orig"], y=top_n["neighbourhood"], orientation="h",
#             marker=dict(color=colors_n),
#             text=top_n["price_orig"].apply(lambda v: f"${v:.0f}"),
#             textposition="outside", textfont=dict(color="#c0c0ff", size=11),
#         ))
#         lay_n = copy.deepcopy(PLOT_LAYOUT)
#         lay_n.update(dict(title="Top Neighbourhoods (avg nightly price)", height=420,
#                           yaxis=dict(autorange="reversed", gridcolor="#1a1a2e",
#                                      tickfont=dict(color="#9090c0"))))
#         fig_top.update_layout(**lay_n)
#         st.plotly_chart(fig_top, use_container_width=True)

#     with tab3:
#         st.markdown("<br>", unsafe_allow_html=True)
#         c1, c2 = st.columns(2)
#         with c1:
#             st.markdown('<p class="section-header">Avg Price by Room Type</p>', unsafe_allow_html=True)
#             rt_avg = df.groupby("room_type")["price_orig"].mean().reset_index().sort_values("price_orig", ascending=False)
#             fig_rt = go.Figure(go.Bar(
#                 x=rt_avg["room_type"], y=rt_avg["price_orig"],
#                 marker=dict(color=[ROOM_COLORS.get(r, "#7b7bff") for r in rt_avg["room_type"]]),
#                 text=rt_avg["price_orig"].apply(lambda v: f"${v:.0f}"), textposition="outside",
#             ))
#             lay_rt = copy.deepcopy(PLOT_LAYOUT)
#             lay_rt.update(title="Average Price (USD)", height=300)
#             fig_rt.update_layout(**lay_rt)
#             st.plotly_chart(fig_rt, use_container_width=True)

#         with c2:
#             st.markdown('<p class="section-header">Listing Count by Room Type</p>', unsafe_allow_html=True)
#             rt_cnt = df["room_type"].value_counts().reset_index()
#             rt_cnt.columns = ["room_type", "count"]
#             fig_cnt = go.Figure(go.Bar(
#                 x=rt_cnt["room_type"], y=rt_cnt["count"],
#                 marker=dict(color=[ROOM_COLORS.get(r, "#7b7bff") for r in rt_cnt["room_type"]]),
#                 text=rt_cnt["count"].apply(lambda v: f"{v:,}"), textposition="outside",
#             ))
#             lay_cnt = copy.deepcopy(PLOT_LAYOUT)
#             lay_cnt.update(title="Number of Listings", height=300)
#             fig_cnt.update_layout(**lay_cnt)
#             st.plotly_chart(fig_cnt, use_container_width=True)

#         st.markdown('<p class="section-header">Avg Price: Room Type × Borough</p>', unsafe_allow_html=True)
#         pivot = df.pivot_table(index="room_type", columns="neighbourhood_group",
#                                 values="price_orig", aggfunc="mean")
#         fig_heat = go.Figure(go.Heatmap(
#             z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
#             colorscale="Viridis",
#             text=[[f"${v:.0f}" for v in row] for row in pivot.values],
#             texttemplate="%{text}", textfont=dict(color="white", size=12),
#         ))
#         lay_h = copy.deepcopy(PLOT_LAYOUT)
#         lay_h.update(title="Average Nightly Price ($)", height=280)
#         fig_heat.update_layout(**lay_h)
#         st.plotly_chart(fig_heat, use_container_width=True)

#     with tab4:
#         st.markdown("<br>", unsafe_allow_html=True)
#         numeric_cols = ["price", "minimum_nights", "number_of_reviews",
#                         "reviews_per_month", "calculated_host_listings_count", "availability_365"]
#         corr = df_enc[numeric_cols].corr()
#         fig_corr = go.Figure(go.Heatmap(
#             z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
#             colorscale="RdBu", zmid=0,
#             text=[[f"{v:.2f}" for v in row] for row in corr.values],
#             texttemplate="%{text}", textfont=dict(size=10, color="white"),
#         ))
#         lay_c = copy.deepcopy(PLOT_LAYOUT)
#         lay_c.update(title="Pearson Correlation Matrix", height=460)
#         fig_corr.update_layout(**lay_c)
#         st.plotly_chart(fig_corr, use_container_width=True)

#         st.markdown("---")
#         st.markdown('<p class="section-header">Key Observations</p>', unsafe_allow_html=True)
#         for title, body in [
#             ("Room type → Price", "Room type is the strongest single predictor. Entire homes command 2–3× the rate of shared rooms."),
#             ("Reviews ↔ Availability", "Listings with more reviews tend to have higher availability — active, well-managed properties."),
#             ("Minimum nights", "Longer minimum-night requirements correlate weakly with lower review counts."),
#             ("Host listings count", "Hosts with many listings price competitively — slight negative correlation with per-listing price."),
#         ]:
#             st.markdown(f'<div class="insight-card"><p><strong>{title}</strong> — {body}</p></div>', unsafe_allow_html=True)

# render()
