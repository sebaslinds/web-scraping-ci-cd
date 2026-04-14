import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from ml.predict import predict_price

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="📊 Book Analytics",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main {
    background-color: #0b1220;
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.hero {
    padding: 6px 0 10px 0;
}

.metric-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px;
}

.metric-label {
    color: #9fb3c8;
    font-size: 0.9rem;
}

.metric-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
}

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
    df["title_length"] = df["title"].astype(str).str.len()
    return df

df = load_data()

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
    (min_price, max_price),
    key="price_slider"
)

df_filtered = df[
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1])
]

# =========================
# HEADER + REFRESH
# =========================
col_title, col_refresh = st.columns([6, 1])

with col_title:
    st.markdown("""
    <div class="hero">
        <h1>📚 Book Analytics Dashboard</h1>
        <p style="margin:0;color:#cbd5f5;">
            Data Pipeline · Analytics · Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_refresh:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if st.button("🔄", key="refresh_btn"):
        load_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(st.session_state.last_refresh.strftime("%H:%M:%S"))

# =========================
# CHECK DATA
# =========================
if df_filtered.empty:
    st.warning("No data matches your filters.")
    st.stop()

# =========================
# KPI
# =========================
def metric_card(col, label, value):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

metric_card(k1, "📦 Books", len(df_filtered))
metric_card(k2, "💰 Avg Price", f"{df_filtered['price'].mean():.2f} £")
metric_card(k3, "📈 Max Price", f"{df_filtered['price'].max():.2f} £")
metric_card(k4, "📉 Min Price", f"{df_filtered['price'].min():.2f} £")

st.markdown("---")

# =========================
# CHARTS
# =========================
def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb"
    )
    return fig

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price Distribution")
    fig1 = px.histogram(df_filtered, x="price", nbins=30)
    st.plotly_chart(style_fig(fig1), use_container_width=True)

with col2:
    st.subheader("📈 Price Spread")
    fig2 = px.box(df_filtered, y="price")
    st.plotly_chart(style_fig(fig2), use_container_width=True)

# =========================
# MACHINE LEARNING (FIXED)
# =========================
st.markdown("## 🤖 Machine Learning")

input_page = st.number_input(
    "Page",
    min_value=1,
    max_value=1000,
    value=50,
    key="ml_page_input"
)

input_title = st.number_input(
    "Title Length",
    min_value=1,
    max_value=200,
    value=20,
    key="ml_title_input"
)

if st.button("Predict Price", key="predict_btn"):
    prediction = predict_price(input_page, input_title)
    st.success(f"Predicted price: {prediction:.2f} £")

# =========================
# DATA TABLE
# =========================
st.markdown("## 📄 Data")

st.dataframe(df_filtered)

csv = df_filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download CSV",
    csv,
    "books_filtered.csv",
    key="download_btn"
)

# =========================
# INSIGHTS
# =========================
st.markdown("## 🧠 Insights")

st.write(f"""
- Average price: **{df_filtered['price'].mean():.2f} £**
- Highest price: **{df_filtered['price'].max():.2f} £**
- Lowest price: **{df_filtered['price'].min():.2f} £**
- Total books: **{len(df_filtered)}**
""")