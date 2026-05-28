import streamlit as st

setup_page("NYC Airbnb · Dashboard")
render()

st.set_page_config(
    page_title="NYC Airbnb Price Intelligence",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
[data-testid="stSidebar"] .stRadio > label { 
    font-family: 'DM Sans', sans-serif; 
    font-size: 0.85rem;
    letter-spacing: 0.04em;
}

/* ── Main background ── */
.stApp { background: #07070f; }
.main .block-container { 
    padding-top: 2rem; 
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] > div:first-child {
    color: #7878a8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
}

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #6060a0 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    border-radius: 8px;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1a1a30 !important;
    color: #c0c0ff !important;
}

/* ── Selectbox & Slider labels ── */
.stSelectbox label, .stSlider label { 
    color: #9090c0 !important; 
    font-size: 0.82rem;
    letter-spacing: 0.04em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6060c8 0%, #9040c8 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.6rem 2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Dividers ── */
hr { border-color: #1e1e35 !important; }

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #0e0e2a 0%, #1a0e2e 100%);
    border: 1px solid #6060c8;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.pred-box h2 {
    color: #b0a0ff !important;
    font-size: 1rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.pred-box h1 {
    color: #ffffff !important;
    font-size: 3rem !important;
    margin: 0;
}
.pred-box p {
    color: #6060a0;
    font-size: 0.8rem;
    margin-top: 0.6rem;
}

/* ── Section headers ── */
.section-header {
    color: #ffffff;
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.section-sub {
    color: #6060a0;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    margin-bottom: 1.5rem;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1.5rem;
}
.sidebar-brand h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    margin: 0.4rem 0 0.1rem;
}
.sidebar-brand p { color: #5050a0 !important; font-size: 0.75rem; margin: 0; }

/* ── Insight cards ── */
.insight-card {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-left: 3px solid #6060c8;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.insight-card p { color: #a0a0c0; font-size: 0.85rem; margin: 0; }
.insight-card strong { color: #c0c0ff; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2rem">🗽</div>
        <h1>NYC Airbnb<br>Intelligence</h1>
        <p>2019 · Price Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["📊  Dashboard", "🔍  EDA & Insights", "🤖  Price Predictor", "📈  Model Metrics"],
        label_visibility="visible",
    )
    
    st.markdown("---")
    st.markdown(
        "<p style='color:#3a3a6a;font-size:0.73rem;text-align:center;padding:0.5rem 0'>"
        "Built with Streamlit · scikit-learn · XGBoost<br>"
        "Data: Airbnb NYC 2019</p>",
        unsafe_allow_html=True,
    )

# ── Route pages ───────────────────────────────────────────────────────────────
pg = page.split("  ")[1].strip()

if pg == "Dashboard":
    from pages.dashboard import render
elif pg == "EDA & Insights":
    from pages.eda import render
elif pg == "Price Predictor":
    from pages.predictor import render
else:
    from pages.metrics import render

render()
