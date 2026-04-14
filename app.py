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
    page_title="📊 Book Analytics Dashboard",
    layout="wide"
)

# =========================
# STYLE (SaaS UI)
# =========================
st.markdown("""
<style>

/* GLOBAL */
.main {
    background: #0b0f19;
    color: #E6EAF0;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* TITLES */
h1 {
    font-size: 42px !important;
    font-weight: 700;
    color: #F9FAFB;
}

h2, h3 {
    color: #A5B4FC;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* KPI CARDS */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 14px;
}

/* KPI TEXT */
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 30px !important;
    font-weight: 700;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #6366F1, #4F46E5);
    color: white;
    border-radius: 10px;
    padding: 8px 16px;
    border: none;
    font-weight: 600;
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
# FEATURE ENGINEERING
# =========================
df["title_length"] = df["title"].astype(str).apply(len)

# =========================
# HEADER
# =========================
st.title("📚 Book Analytics Dashboard")

st.markdown("""
### 🚀 End-to-End Data Platform

Build with:
- Data Engineering (ETL pipeline)
- CI/CD automation
- Machine Learning (price prediction)
- Interactive analytics dashboard
""")

st.markdown(
    "<p style='color:#94A3B8'>Real-time insights from scraped book data</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# DATA PREVIEW
# =========================
with st.expander("📊 Raw Data Preview"):
    st.dataframe(df.head(20))

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

min_price = float(df["price"].min())
max_price = float(df["price"].max())

if min_price == max_price:
    st.sidebar.warning("⚠️ All prices are identical")
    price_range = (min_price, max_price)
else:
    price_range = st.sidebar.slider(
        "💰 Price Range (£)",
        min_price,
        max_price,
        (min_price, max_price)
    )

# =========================
# FILTER
# =========================
df_filtered = df[
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1])
]

# =========================
# REFRESH
# =========================
colA, colB = st.columns([4,1])

with colB:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if st.button("🔄 Refresh"):
        load_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(f"⏱ Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# =========================
# EMPTY STATE
# =========================
if df_filtered.empty:
    st.warning("⚠️ No data for selected filters")
    st.stop()

# =========================
# OVERVIEW
# =========================
st.markdown("## 📊 Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("📦 Books", len(df_filtered))
k2.metric("💰 Avg Price", f"{df_filtered['price'].mean():.2f} £")
k3.metric("📈 Max Price", f"{df_filtered['price'].max():.2f} £")
k4.metric("📉 Min Price", f"{df_filtered['price'].min():.2f} £")

st.markdown("---")

# =========================
# ANALYTICS
# =========================
st.markdown("## 📈 Analytics")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df_filtered, x="price", nbins=30, title="Price Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(df_filtered, y="price", title="Price Spread")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# MACHINE LEARNING
# =========================
st.markdown("## 🤖 Machine Learning")

features = ["page", "title_length"]
features = [f for f in features if f in df.columns]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

m1, m2 = st.columns(2)
m1.metric("📉 MAE", f"{mae:.2f} £")
m2.metric("📊 R² Score", f"{r2:.2f}")

st.caption("Model: Random Forest Regressor")

# =========================
# PREDICTION
# =========================
st.markdown("### 🔮 Predict Price")

input_page = st.number_input("Page", 1, 1000, 50)
input_title = st.number_input("Title Length", 1, 200, 20)

if st.button("💡 Predict"):
    pred_df = pd.DataFrame([{
        "page": input_page,
        "title_length": input_title
    }])

    prediction = model.predict(pred_df)[0]
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
    "books_filtered.csv"
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