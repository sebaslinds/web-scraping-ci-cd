import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="📊 Book Analytics PRO",
    layout="wide"
)

# =========================
# STYLE PRO
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

/* TITLES */
h1, h2, h3 {
    color: #00C9A7;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111827;
}
section[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

/* KPI CARDS */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

/* KPI LABEL */
[data-testid="stMetricLabel"] {
    color: #9CA3AF !important;
}

/* KPI VALUE */
[data-testid="stMetricValue"] {
    color: #F9FAFB !important;
    font-size: 28px !important;
    font-weight: 700 !important;
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
st.title("📚 Book Analytics Dashboard PRO")
st.markdown("🚀 Pipeline CI/CD + Data Engineering + Machine Learning")

# =========================
# CONTEXTE
# =========================
st.markdown("""
### 📊 Contexte
Analyse des prix de livres scrapés automatiquement.

### ⚙️ Pipeline
- Scraping automatisé
- Transformation Bronze → Silver → Gold
- CI/CD GitHub Actions

### 🎯 Objectif
Analyser les prix + prédire les tendances.
""")

st.markdown("---")

# =========================
# RAW DATA
# =========================
with st.expander("📊 Aperçu des données"):
    st.dataframe(df.head(20))

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

min_price = float(df["price"].min())
max_price = float(df["price"].max())

if min_price == max_price:
    st.sidebar.warning("⚠️ Tous les prix sont identiques")
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

    st.caption(f"⏱ {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# =========================
# EMPTY CHECK
# =========================
if df_filtered.empty:
    st.warning("⚠️ Aucun résultat")
    st.stop()

# =========================
# KPIs
# =========================
k1, k2, k3, k4 = st.columns(4)

k1.metric("📦 Books", len(df_filtered))
k2.metric("💰 Avg Price", f"{df_filtered['price'].mean():.2f} £")
k3.metric("📈 Max", f"{df_filtered['price'].max():.2f} £")
k4.metric("📉 Min", f"{df_filtered['price'].min():.2f} £")

st.markdown("---")

# =========================
# CHARTS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribution")
    fig1 = px.histogram(df_filtered, x="price", nbins=30)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📈 Boxplot")
    fig2 = px.box(df_filtered, y="price")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# MACHINE LEARNING
# =========================
st.markdown("---")
st.markdown("## 🤖 Price Prediction Model")

features = ["page", "title_length"]
features = [f for f in features if f in df.columns]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

c1, c2 = st.columns(2)
c1.metric("📉 MAE", f"{mae:.2f} £")
c2.metric("📊 R²", f"{r2:.2f}")

st.caption("Modèle basé sur RandomForest")

# =========================
# PREDICTION UI
# =========================
st.markdown("### 🔮 Predict a Book Price")

input_page = st.number_input("Page", min_value=1, max_value=1000, value=50)
input_title = st.number_input("Title Length", min_value=1, max_value=200, value=20)

if st.button("💡 Predict Price"):
    input_df = pd.DataFrame([{
        "page": input_page,
        "title_length": input_title
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted price: {prediction:.2f} £")

# =========================
# TABLE
# =========================
st.markdown("---")
st.subheader("📄 Data")

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
- 📊 Average price: **{df_filtered['price'].mean():.2f} £**
- 📈 Highest price: **{df_filtered['price'].max():.2f} £**
- 📉 Lowest price: **{df_filtered['price'].min():.2f} £**
- 📦 Total books: **{len(df_filtered)}**
""")