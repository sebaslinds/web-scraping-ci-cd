import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Book Analytics", layout="wide")

# =========================
# UNICORN UI STYLE
# =========================
st.markdown("""
<style>

/* BACKGROUND */
html, body, [class*="css"] {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: #e2e8f0;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* HERO CARD */
.hero {
    background: linear-gradient(135deg, rgba(14,165,233,0.3), rgba(34,197,94,0.2));
    backdrop-filter: blur(16px);
    padding: 35px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* KPI CARDS */
.kpi-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 25px 60px rgba(0,0,0,0.6);
}

/* TEXT */
.kpi-label {
    color: rgba(255,255,255,0.85);
    font-size: 14px;
}

.kpi-value {
    font-size: 34px;
    font-weight: 700;
    color: white;
}

/* SECTION */
.section {
    margin-top: 40px;
}

/* AI INSIGHTS */
.insight-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* BUTTON */
button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #22c55e);
    border: none;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/silver/books_clean.csv")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

min_price = float(df["price"].min())
max_price = float(df["price"].max())

price_range = st.sidebar.slider(
    "💰 Price Range",
    min_price,
    max_price,
    (min_price, max_price)
)

df_filtered = df[
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1])
]

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <h1>📚 Book Analytics Dashboard</h1>
    <p>End-to-End Data Product · Data Engineering · ML Predictions · SaaS UI</p>
</div>
""", unsafe_allow_html=True)

# =========================
# REFRESH
# =========================
colA, colB = st.columns([6,1])

with colB:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if st.button("🔄"):
        load_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(st.session_state.last_refresh.strftime("%H:%M:%S"))

# =========================
# KPIs
# =========================
st.markdown('<div class="section"><h2>📊 Overview</h2></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

def kpi(col, title, value):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1, "📦 Books", len(df_filtered))
kpi(c2, "💰 Avg Price", f"{df_filtered['price'].mean():.2f} £")
kpi(c3, "📈 Max Price", f"{df_filtered['price'].max():.2f} £")
kpi(c4, "📉 Min Price", f"{df_filtered['price'].min():.2f} £")

# =========================
# CHARTS
# =========================
st.markdown('<div class="section"><h2>📊 Analytics</h2></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df_filtered, x="price", nbins=30)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.box(df_filtered, y="price")
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# AI INSIGHTS
# =========================
st.markdown('<div class="section"><h2>🧠 AI Insights</h2></div>', unsafe_allow_html=True)

if not df_filtered.empty:
    avg = df_filtered["price"].mean()
    max_p = df_filtered["price"].max()
    min_p = df_filtered["price"].min()

    st.markdown(f"""
    <div class="insight-card">
        📊 Average price is <b>{avg:.2f} £</b><br><br>
        📈 Highest price: <b>{max_p:.2f} £</b><br>
        📉 Lowest price: <b>{min_p:.2f} £</b><br><br>
        💡 Insight: Price distribution shows a spread typical of retail books,
        with potential clustering in mid-range values.
    </div>
    """, unsafe_allow_html=True)

# =========================
# TABLE
# =========================
st.markdown('<div class="section"><h2>📄 Data</h2></div>', unsafe_allow_html=True)

st.dataframe(df_filtered)