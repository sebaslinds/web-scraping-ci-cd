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
# STYLES
# =========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: #e2e8f0;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"],
div[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

.hero {
    background: linear-gradient(135deg, rgba(14,165,233,0.22), rgba(34,197,94,0.16));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 28px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}

.metric-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.25);
}

.metric-label {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 8px;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}

.insight-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
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
    all_categories = sorted(df["category"].dropna().unique())
    categories = st.sidebar.multiselect(
        "📚 Category",
        options=all_categories,
        default=all_categories
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
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <h1>📚 Book Analytics</h1>
    <p style="margin:0; color:#dbeafe;">
        Web Scraping · Medallion Architecture · Analytics · ML
    </p>
</div>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([5, 1])

with top_right:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if st.button("🔄 Refresh"):
        load_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(f"Updated {st.session_state.last_refresh.strftime('%H:%M:%S')}")

st.markdown("### Product Overview")
st.write(
    "This project demonstrates how scraped book data can be transformed into an end-to-end analytics platform with interactive exploration, actionable KPIs, and a machine learning component for price prediction."
)

if df_filtered.empty:
    st.warning("No results match the selected filters.")
    st.stop()

# =========================
# KPI ROW
# =========================
st.markdown("## Overview")

c1, c2, c3, c4 = st.columns(4)

def metric_card(col, label, value):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

metric_card(c1, "📦 Books", f"{len(df_filtered):,}")
metric_card(c2, "💰 Avg Price", f"{df_filtered['price'].mean():.2f} £")
metric_card(c3, "📈 Max Price", f"{df_filtered['price'].max():.2f} £")
metric_card(c4, "📉 Min Price", f"{df_filtered['price'].min():.2f} £")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Analytics",
    "🤖 ML",
    "📄 Data",
    "🧠 Insights"
])

# =========================
# ANALYTICS TAB
# =========================
with tab1:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df_filtered,
            x="price",
            nbins=30,
            title="Price Distribution",
            color_discrete_sequence=["#6366f1"]
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            title_font_color="#ffffff"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig_box = px.box(
            df_filtered,
            y="price",
            title="Price Spread",
            color_discrete_sequence=["#10b981"]
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            title_font_color="#ffffff"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if "category" in df_filtered.columns:
        st.markdown("### Category Analysis")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df_cat = (
            df_filtered.groupby("category", as_index=False)["price"]
            .mean()
            .sort_values("price", ascending=False)
        )
        fig_cat = px.bar(
            df_cat,
            x="category",
            y="price",
            title="Average Price by Category",
            color="price",
            color_continuous_scale="viridis"
        )
        fig_cat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            title_font_color="#ffffff",
            xaxis_tickangle=-35
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ML TAB
# =========================
with tab2:
    st.markdown("### Price Prediction")

    features = [c for c in ["page", "title_length"] if c in df.columns]
    X = df[features]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m1, m2 = st.columns(2)
    metric_card(m1, "📉 MAE", f"{mean_absolute_error(y_test, y_pred):.2f} £")
    metric_card(m2, "📊 R² Score", f"{r2_score(y_test, y_pred):.2f}")

    st.caption("Model: Random Forest Regressor")

    p1, p2 = st.columns(2)
    with p1:
        input_page = st.number_input("Page", min_value=1, max_value=1000, value=50)
    with p2:
        input_title = st.number_input("Title Length", min_value=1, max_value=200, value=20)

    if st.button("Predict Price"):
        pred_df = pd.DataFrame([{
            "page": input_page,
            "title_length": input_title
        }])
        prediction = model.predict(pred_df)[0]
        st.success(f"Predicted price: {prediction:.2f} £")

# =========================
# DATA TAB
# =========================
with tab3:
    st.markdown("### Filtered Dataset")
    st.dataframe(df_filtered, use_container_width=True)

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        csv,
        "books_filtered.csv",
        "text/csv"
    )

    with st.expander("Raw Data Preview"):
        st.dataframe(df.head(20), use_container_width=True)

# =========================
# INSIGHTS TAB
# =========================
with tab4:
    avg_price = df_filtered["price"].mean()
    max_price_f = df_filtered["price"].max()
    min_price_f = df_filtered["price"].min()

    st.markdown(
        f"""
        <div class="insight-card">
            <h3 style="margin-top:0;">🧠 Key Insights</h3>
            <p>Average price: <b>{avg_price:.2f} £</b></p>
            <p>Highest price: <b>{max_price_f:.2f} £</b></p>
            <p>Lowest price: <b>{min_price_f:.2f} £</b></p>
            <p>Total books in selection: <b>{len(df_filtered)}</b></p>
            <p style="margin-top:1rem;">
                This dataset shows a retail-style pricing range with visible spread between lower-cost and premium books.
                The dashboard helps surface pricing structure and supports quick predictive exploration through ML.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )