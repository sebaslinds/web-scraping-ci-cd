import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Book Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# ULTIMATE UI (SaaS)
# =========================
st.markdown("""
<style>

/* ========= GLOBAL ========= */
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.15), transparent),
                radial-gradient(900px 500px at 110% 10%, rgba(16,185,129,0.12), transparent),
                linear-gradient(180deg, #0b1020 0%, #0f172a 100%) !important;
    color: #e5e7eb;
}
.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Remove Streamlit white header/strips */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: transparent !important;
    display: none !important;
}

/* ========= TYPO ========= */
h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    letter-spacing: -0.03em;
}
h2, h3 {
    color: #c7d2fe !important;
    font-weight: 700 !important;
}
.muted { color: #94a3b8; }

/* ========= SIDEBAR ========= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ========= HERO ========= */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.20), rgba(16,185,129,0.14));
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px 24px;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}
.hero h3 { margin: 0 0 .4rem 0; color:#fff !important; }
.hero p { margin: 0; color:#cbd5e1 !important; }

/* ========= CARDS ========= */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    backdrop-filter: blur(6px);
}
.card:hover {
    transform: translateY(-2px);
    transition: .2s ease;
}

/* ========= KPI ========= */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #1f2937) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 22px !important;
    box-shadow: 0 10px 26px rgba(0,0,0,0.28) !important;
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: .9rem !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: .02em;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    transition: .2s ease;
}

/* ========= BUTTONS ========= */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: .65rem 1rem !important;
    border: none !important;
    box-shadow: 0 10px 22px rgba(79,70,229,0.28);
}
.stButton>button:hover, .stDownloadButton>button:hover {
    filter: brightness(1.08);
}

/* ========= EXPANDER ========= */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    font-weight: 700;
}

/* ========= DATAFRAME ========= */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    overflow: hidden;
}

/* ========= DIVIDER ========= */
hr { border-color: rgba(255,255,255,0.08); }

</style>
""", unsafe_allow_html=True)

# =========================
# DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/silver/books_clean.csv")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()
    df["title_length"] = df["title"].astype(str).str.len()
    return df

df = load_data()

# =========================
# HEADER
# =========================
st.title("📚 Book Analytics")

st.markdown("""
<div class="hero">
    <h3>End-to-End Data Product</h3>
    <p>Automated scraping → Bronze → Silver → Gold modeling, CI/CD, interactive analytics, and a machine learning layer for price prediction.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Context
This dashboard analyzes book prices collected through automated web scraping.

### Pipeline
- Automated scraping
- Bronze → Silver → Gold transformation
- CI/CD with GitHub Actions

### Objective
Identify pricing patterns and predict book prices.
""")

st.markdown("---")

# =========================
# PREVIEW
# =========================
with st.expander("📊 Raw Data Preview"):
    st.dataframe(df.head(20), use_container_width=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

min_price = float(df["price"].min())
max_price = float(df["price"].max())

if min_price == max_price:
    st.sidebar.warning("All prices are identical.")
    price_range = (min_price, max_price)
else:
    price_range = st.sidebar.slider(
        "💰 Price Range (£)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        format="£%.2f"
    )

categories = None
if "category" in df.columns:
    categories = st.sidebar.multiselect(
        "📚 Category",
        options=sorted(df["category"].dropna().unique()),
        default=sorted(df["category"].dropna().unique())
    )

# =========================
# FILTER
# =========================
df_filtered = df[
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1])
].copy()

if categories is not None:
    df_filtered = df_filtered[df_filtered["category"].isin(categories)]

# =========================
# REFRESH
# =========================
left, right = st.columns([5, 1])

with right:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if st.button("🔄 Refresh"):
        load_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

if df_filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# =========================
# KPIs
# =========================
st.markdown("## 📊 Overview")
k1, k2, k3, k4 = st.columns(4)

k1.metric("📦 Books", f"{len(df_filtered):,}")
k2.metric("💰 Avg Price", f"{df_filtered['price'].mean():.2f} £")
k3.metric("📈 Max Price", f"{df_filtered['price'].max():.2f} £")
k4.metric("📉 Min Price", f"{df_filtered['price'].min():.2f} £")

st.markdown("---")

# =========================
# ANALYTICS
# =========================
st.markdown("## 📈 Analytics")
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig_hist = px.histogram(df_filtered, x="price", nbins=30, title="Price Distribution",
                            color_discrete_sequence=["#6366f1"])
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        title_font_color="#ffffff"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig_box = px.box(df_filtered, y="price", title="Price Spread",
                     color_discrete_sequence=["#10b981"])
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        title_font_color="#ffffff"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if "category" in df_filtered.columns:
    st.markdown("### 📚 Category Analysis")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df_cat = (
        df_filtered.groupby("category", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
    )
    fig_cat = px.bar(df_cat, x="category", y="price",
                     title="Average Price by Category",
                     color="price", color_continuous_scale="viridis")
    fig_cat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        title_font_color="#ffffff",
        xaxis_tickangle=-35
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# =========================
# ML
# =========================
st.markdown("## 🤖 Machine Learning")

features = [c for c in ["page", "title_length"] if c in df.columns]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

m1, m2 = st.columns(2)
m1.metric("📉 MAE", f"{mean_absolute_error(y_test, y_pred):.2f} £")
m2.metric("📊 R² Score", f"{r2_score(y_test, y_pred):.2f}")

st.caption("Model: Random Forest Regressor")

st.markdown("### 🔮 Predict Price")

p1, p2 = st.columns(2)
with p1:
    input_page = st.number_input("Page", 1, 1000, 50)
with p2:
    input_title = st.number_input("Title Length", 1, 200, 20)

if st.button("Predict Price"):
    pred_df = pd.DataFrame([{"page": input_page, "title_length": input_title}])
    prediction = model.predict(pred_df)[0]
    st.success(f"Predicted price: {prediction:.2f} £")

st.markdown("---")

# =========================
# DATA
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
- Average price: **{df_filtered['price'].mean():.2f} £**
- Highest price: **{df_filtered['price'].max():.2f} £**
- Lowest price: **{df_filtered['price'].min():.2f} £**
- Total books: **{len(df_filtered)}**
""")