import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/AB_NYC_2019_processed01.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def get_enriched_df() -> pd.DataFrame:
    """Return df with original price scale approximated for display."""
    df = load_data()
    # The processed CSV has z-score normalised price.
    # We reconstruct approximate original price (pre-outlier-removal dataset
    # had mean ~152, std ~240 for full range; post-outlier ~106, std ~64).
    # Use the cleaned dataset's stats to invert.
    PRICE_MEAN = 106.0
    PRICE_STD  = 64.0
    df["price_orig"] = df["price"] * PRICE_STD + PRICE_MEAN
    df["price_orig"] = df["price_orig"].clip(lower=10)
    return df

@st.cache_data
def get_encoded_df():
    df = get_enriched_df()
    le_room = LabelEncoder()
    le_hood = LabelEncoder()
    df["room_type_encoded"]       = le_room.fit_transform(df["room_type"])
    df["neighbourhood_grp_enc"]   = le_hood.fit_transform(df["neighbourhood_group"])
    return df, le_room, le_hood

BOROUGH_COLORS = {
    "Manhattan":    "#7b7bff",
    "Brooklyn":     "#ff7bbd",
    "Queens":       "#7bffcc",
    "Bronx":        "#ffd07b",
    "Staten Island":"#ff907b",
}

ROOM_COLORS = {
    "Entire home/apt": "#7b7bff",
    "Private room":    "#ff7bbd",
    "Shared room":     "#7bffcc",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0a12",
    plot_bgcolor="#0d0d1a",
    font=dict(family="DM Sans, sans-serif", color="#9090c0", size=12),
    title_font=dict(family="Syne, sans-serif", color="#e0e0ff", size=16),
    xaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#6060a0")),
    yaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#6060a0")),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="#0a0a12", bordercolor="#1e1e2e", borderwidth=1,
                font=dict(color="#9090c0")),
)
