import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="📊 Book Analytics",
    layout="wide"
)

# =========================
# STYLE (CLEAN SAAS UI)
# =========================
st.markdown("""
<style>

/* GLOBAL */
.main {
    background-color: #0b1220;
    color: #e5e7eb;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* HEADER */
.hero {
    padding: 6px 0 10px 0;
}

/* KPI CARDS */
.kpi-card {
    background: linear-gradient(135deg, #0b1220, #121a2b);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px;
}

.kpi-label {
    color: #9fb3c8;
    font-size: 0.9rem;
}

.kpi-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
}

/* TABLE */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* BUTTON */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
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
    df = df.dropna(subset=["price"])
    return df

df = load_data()

# =========================
# HEADER + REFRESH
# =========================
col_title, col_refresh = st.columns([6,1])

with col_title:
    st.markdown("""
    <div class="hero">
        <h1>📚 Book Analytics Dashboard</h1>
        <p style="margin:0;color:#cbd5f5;">
            Web Scraping · Medallion Architecture · Analytics · Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_refresh:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if st.button("🔄 Refresh"):
        load_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(st.session_state.last_refresh.strftime("%H:%M:%S"))

# =========================
# CONTEXT
# =========================
st.markdown("""
### 📊 Context
This project demonstrates how scraped book data can be transformed into an end-to-end analytics platform with interactive exploration, actionable KPIs, and a machine learning component for price prediction.

### ⚙️ Pipeline
- Automated scraping
- Bronze → Silver → Gold transformation
- CI/CD with GitHub Actions

### 🎯 Objective
Analyze pricing patterns and predict book prices.
""")

# =========================
# RAW DATA
# =========================
with st.expander("📊 Raw Data Preview"):
    st.dataframe(df.head(20), use_container_width=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

min_price = float(df["price"].min())
max_price = float(df["price"].max())

price_range = st.sidebar.slider(
    "💰 Price Range (£)",
    min_price,
    max_price,
    (min_price, max_price)
)

# =========================
# FILTER DATA
# =========================
df_filtered = df[
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1])
]

if df_filtered.empty:
    st.warning("No data matches your filters.")
    st.stop()

# =========================
# KPI FUNCTION
# =========================
def kpi(col, label, value):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# KPIs
# =========================
k1, k2, k3, k4 = st.columns(4)

kpi(k1, "📦 Books", len(df_filtered))
kpi(k2, "💰 Avg Price", f"{df_filtered['price'].mean():.2f} £")
kpi(k3, "📈 Max Price", f"{df_filtered['price'].max():.2f} £")
kpi(k4, "📉 Min Price", f"{df_filtered['price'].min():.2f} £")

st.markdown("---")

# =========================
# CHARTS
# =========================
def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price Distribution")
    fig1 = px.histogram(df_filtered, x="price", nbins=30, color_discrete_sequence=["#6366f1"])
    st.plotly_chart(style_fig(fig1), use_container_width=True)

with col2:
    st.subheader("📈 Price Boxplot")
    fig2 = px.box(df_filtered, y="price", color_discrete_sequence=["#10b981"])
    st.plotly_chart(style_fig(fig2), use_container_width=True)

# =========================
# MACHINE LEARNING
# =========================
st.markdown("## 🤖 Machine Learning")

input_page = st.number_input("Page", 1, 1000, 50)
input_title = st.number_input("Title Length", 1, 200, 20)

if st.button("Predict Price"):
    prediction = predict_price(input_page, input_title)
    st.success(f"Predicted price: {prediction:.2f} £")

# =========================
# TABLE + EXPORT
# =========================
st.markdown("## 📄 Data")

st.dataframe(df_filtered, use_container_width=True)

csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "books_filtered.csv")

# =========================
# INSIGHTS
# =========================
st.markdown("## 🧠 Insights")

st.write(f"""
- 📊 Average price: **{df_filtered['price'].mean():.2f} £**
- 📈 Highest price: **{df_filtered['price'].max():.2f} £**
- 📉 Lowest price: **{df_filtered['price'].min():.2f} £**
- 📦 Total books: **{len(df_filtered)}**
""")