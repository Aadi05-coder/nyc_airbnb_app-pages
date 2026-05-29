# # import pandas as pd
# # import numpy as np
# # import streamlit as st
# # from sklearn.preprocessing import LabelEncoder

# # DATA_PATH = "data/AB_NYC_2019_processed01.csv"

# # @st.cache_data
# # def load_data() -> pd.DataFrame:
# #     df = pd.read_csv(DATA_PATH)
# #     return df

# # @st.cache_data
# # def get_enriched_df() -> pd.DataFrame:
# #     """Return df with original price scale approximated for display."""
# #     df = load_data()
# #     # The processed CSV has z-score normalised price.
# #     # We reconstruct approximate original price (pre-outlier-removal dataset
# #     # had mean ~152, std ~240 for full range; post-outlier ~106, std ~64).
# #     # Use the cleaned dataset's stats to invert.
# #     PRICE_MEAN = 106.0
# #     PRICE_STD  = 64.0
# #     df["price_orig"] = df["price"] * PRICE_STD + PRICE_MEAN
# #     df["price_orig"] = df["price_orig"].clip(lower=10)
# #     return df

# # @st.cache_data
# # def get_encoded_df():
# #     df = get_enriched_df()
# #     le_room = LabelEncoder()
# #     le_hood = LabelEncoder()
# #     df["room_type_encoded"]       = le_room.fit_transform(df["room_type"])
# #     df["neighbourhood_grp_enc"]   = le_hood.fit_transform(df["neighbourhood_group"])
# #     return df, le_room, le_hood

# # BOROUGH_COLORS = {
# #     "Manhattan":    "#7b7bff",
# #     "Brooklyn":     "#ff7bbd",
# #     "Queens":       "#7bffcc",
# #     "Bronx":        "#ffd07b",
# #     "Staten Island":"#ff907b",
# # }

# # ROOM_COLORS = {
# #     "Entire home/apt": "#7b7bff",
# #     "Private room":    "#ff7bbd",
# #     "Shared room":     "#7bffcc",
# # }

# # PLOT_LAYOUT = dict(
# #     paper_bgcolor="#0a0a12",
# #     plot_bgcolor="#0d0d1a",
# #     font=dict(family="DM Sans, sans-serif", color="#9090c0", size=12),
# #     title_font=dict(family="Syne, sans-serif", color="#e0e0ff", size=16),
# #     xaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#6060a0")),
# #     yaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#6060a0")),
# #     margin=dict(l=40, r=20, t=50, b=40),
# #     legend=dict(bgcolor="#0a0a12", bordercolor="#1e1e2e", borderwidth=1,
# #                 font=dict(color="#9090c0")),
# # )







# import pandas as pd
# import numpy as np
# import streamlit as st
# from sklearn.preprocessing import LabelEncoder

# DATA_PATH = "data/AB_NYC_2019_processed01.csv"

# def setup_page(title="NYC Airbnb Price Intelligence"):
#     st.set_page_config(
#         page_title=title,
#         page_icon="🗽",
#         layout="wide",
#         initial_sidebar_state="expanded",
#     )
#     st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
#     html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#     [data-testid="stSidebar"] { background: #0a0a0f; border-right: 1px solid #1e1e2e; }
#     [data-testid="stSidebar"] * { color: #e0e0f0 !important; }
#     .stApp { background: #07070f; }
#     .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
#     h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }
#     [data-testid="metric-container"] {
#         background: #0f0f1a; border: 1px solid #1e1e35; border-radius: 12px; padding: 1rem 1.2rem;
#     }
#     [data-testid="metric-container"] > div:first-child {
#         color: #7878a8 !important; font-size: 0.78rem !important;
#         letter-spacing: 0.08em; text-transform: uppercase;
#     }
#     [data-testid="metric-container"] [data-testid="stMetricValue"] {
#         color: #ffffff !important; font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important;
#     }
#     .stTabs [data-baseweb="tab-list"] { background: #0f0f1a; border-radius: 10px; padding: 4px; gap: 2px; }
#     .stTabs [data-baseweb="tab"] { color: #6060a0 !important; font-family: 'DM Sans', sans-serif; font-size: 0.85rem; border-radius: 8px; padding: 8px 20px; }
#     .stTabs [aria-selected="true"] { background: #1a1a30 !important; color: #c0c0ff !important; }
#     .stSelectbox label, .stSlider label { color: #9090c0 !important; font-size: 0.82rem; letter-spacing: 0.04em; }
#     .stButton > button {
#         background: linear-gradient(135deg, #6060c8 0%, #9040c8 100%);
#         color: white; border: none; border-radius: 10px;
#         font-family: 'Syne', sans-serif; font-weight: 600;
#         letter-spacing: 0.04em; padding: 0.6rem 2rem; transition: opacity 0.2s;
#     }
#     .stButton > button:hover { opacity: 0.85; }
#     hr { border-color: #1e1e35 !important; }
#     .pred-box {
#         background: linear-gradient(135deg, #0e0e2a 0%, #1a0e2e 100%);
#         border: 1px solid #6060c8; border-radius: 16px;
#         padding: 2rem; text-align: center; margin-top: 1.5rem;
#     }
#     .pred-box h2 { color: #b0a0ff !important; font-size: 1rem; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem; }
#     .pred-box h1 { color: #ffffff !important; font-size: 3rem !important; margin: 0; }
#     .pred-box p { color: #6060a0; font-size: 0.8rem; margin-top: 0.6rem; }
#     .section-header { color: #ffffff; font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
#     .section-sub { color: #6060a0; font-size: 0.85rem; letter-spacing: 0.04em; margin-bottom: 1.5rem; }
#     .insight-card {
#         background: #0f0f1a; border: 1px solid #1e1e35;
#         border-left: 3px solid #6060c8; border-radius: 0 10px 10px 0;
#         padding: 1rem 1.2rem; margin-bottom: 0.8rem;
#     }
#     .insight-card p { color: #a0a0c0; font-size: 0.85rem; margin: 0; }
#     .insight-card strong { color: #c0c0ff; }
#     </style>
#     """, unsafe_allow_html=True)

# @st.cache_data
# def load_data() -> pd.DataFrame:
#     df = pd.read_csv(DATA_PATH)
#     return df

# @st.cache_data
# def get_enriched_df() -> pd.DataFrame:
#     df = load_data()
#     PRICE_MEAN = 106.0
#     PRICE_STD  = 64.0
#     df["price_orig"] = df["price"] * PRICE_STD + PRICE_MEAN
#     df["price_orig"] = df["price_orig"].clip(lower=10)
#     return df

# @st.cache_data
# def get_encoded_df():
#     df = get_enriched_df()
#     le_room = LabelEncoder()
#     le_hood = LabelEncoder()
#     df["room_type_encoded"]     = le_room.fit_transform(df["room_type"])
#     df["neighbourhood_grp_enc"] = le_hood.fit_transform(df["neighbourhood_group"])
#     return df, le_room, le_hood

# BOROUGH_COLORS = {
#     "Manhattan":    "#7b7bff",
#     "Brooklyn":     "#ff7bbd",
#     "Queens":       "#7bffcc",
#     "Bronx":        "#ffd07b",
#     "Staten Island":"#ff907b",
# }

# ROOM_COLORS = {
#     "Entire home/apt": "#7b7bff",
#     "Private room":    "#ff7bbd",
#     "Shared room":     "#7bffcc",
# }

# PLOT_LAYOUT = dict(
#     paper_bgcolor="#0a0a12",
#     plot_bgcolor="#0d0d1a",
#     font=dict(family="DM Sans, sans-serif", color="#9090c0", size=12),
#     title_font=dict(family="Syne, sans-serif", color="#e0e0ff", size=16),
#     xaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#6060a0")),
#     yaxis=dict(gridcolor="#1a1a2e", linecolor="#1a1a2e", tickfont=dict(color="#6060a0")),
#     margin=dict(l=40, r=20, t=50, b=40),
#     legend=dict(bgcolor="#0a0a12", bordercolor="#1e1e2e", borderwidth=1, font=dict(color="#9090c0")),
# )








import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/AB_NYC_2019_processed01.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_enriched_df() -> pd.DataFrame:
    df = load_data()
    PRICE_MEAN = 106.0
    PRICE_STD  = 64.0
    df["price_orig"] = (df["price"] * PRICE_STD + PRICE_MEAN).clip(lower=10)
    return df

@st.cache_data
def get_encoded_df():
    df = get_enriched_df()
    le_room = LabelEncoder()
    le_hood = LabelEncoder()
    df["room_type_encoded"]     = le_room.fit_transform(df["room_type"])
    df["neighbourhood_grp_enc"] = le_hood.fit_transform(df["neighbourhood_group"])
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

def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    [data-testid="stSidebarNav"] { display: none !important; }
    html, body, [class*="css"]   { font-family: 'DM Sans', sans-serif; }
    .stApp                       { background: #07070f; }
    .main .block-container       { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
    h1, h2, h3, h4               { font-family: 'Syne', sans-serif !important; }

    [data-testid="stSidebar"] {
        background: #0a0a0f !important;
        border-right: 1px solid #1e1e2e;
    }
    [data-testid="stSidebar"] * { color: #e0e0f0 !important; }

    [data-testid="metric-container"] {
        background: #0f0f1a; border: 1px solid #1e1e35; border-radius: 12px; padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] > div:first-child {
        color: #7878a8 !important; font-size: 0.78rem !important;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #0f0f1a; border-radius: 10px; padding: 4px; gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6060a0 !important; font-size: 0.85rem; border-radius: 8px; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] { background: #1a1a30 !important; color: #c0c0ff !important; }
    .stSelectbox label, .stSlider label { color: #9090c0 !important; font-size: 0.82rem; }
    .stButton > button {
        background: linear-gradient(135deg, #6060c8 0%, #9040c8 100%);
        color: white; border: none; border-radius: 10px;
        font-family: 'Syne', sans-serif; font-weight: 600;
        letter-spacing: 0.04em; padding: 0.6rem 2rem;
    }
    hr { border-color: #1e1e35 !important; }
    .pred-box {
        background: linear-gradient(135deg, #0e0e2a 0%, #1a0e2e 100%);
        border: 1px solid #6060c8; border-radius: 16px;
        padding: 2rem; text-align: center; margin-top: 1.5rem;
    }
    .pred-box h2 { color: #b0a0ff !important; font-size: 1rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .pred-box h1 { color: #ffffff !important; font-size: 3rem !important; margin: 0; }
    .pred-box p  { color: #6060a0; font-size: 0.8rem; margin-top: 0.6rem; }
    .section-header {
        color: #ffffff; font-family: 'Syne', sans-serif;
        font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem;
    }
    .section-sub { color: #6060a0; font-size: 0.85rem; letter-spacing: 0.04em; margin-bottom: 1.5rem; }
    .insight-card {
        background: #0f0f1a; border: 1px solid #1e1e35;
        border-left: 3px solid #6060c8; border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .insight-card p      { color: #a0a0c0; font-size: 0.85rem; margin: 0; }
    .insight-card strong { color: #c0c0ff; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;border-bottom:1px solid #1e1e2e;margin-bottom:1.5rem">
            <div style="font-size:2.5rem">🗽</div>
            <h1 style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                       color:#ffffff;letter-spacing:-0.02em;margin:0.4rem 0 0.1rem">
                NYC Airbnb<br>Intelligence
            </h1>
            <p style="color:#5050a0;font-size:0.75rem;margin:0">2019 · Price Analytics</p>
        </div>
        """, unsafe_allow_html=True)

        st.page_link("app.py",            label="📊  Dashboard")
        st.page_link("pages/eda.py",       label="🔍  EDA & Insights")
        st.page_link("pages/predictor.py", label="🤖  Price Predictor")
        st.page_link("pages/metrics.py",   label="📈  Model Metrics")

        st.markdown("---")
        st.markdown(
            "<p style='color:#3a3a6a;font-size:0.73rem;text-align:center'>"
            "Streamlit · scikit-learn · XGBoost<br>Airbnb NYC 2019</p>",
            unsafe_allow_html=True,
        )
