import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="📊 Book Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLE (léger)
# =========================
st.markdown("""
<style>
.kpi-card {
    padding: 16px;
    border-radius: 12px;
    background: #0f172a;
    color: white;
}
.small-muted { color: #94a3b8; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA (cache)
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # sécuriser les types
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # si tu as rating/category, on garde tel quel
    return df

DATA_PATH = "data/silver/books_clean.csv"  # 🔁 IMPORTANT
df = load_data(DATA_PATH)

# =========================
# HEADER
# =========================
st.title("📚 Book Analytics Dashboard")
st.caption("Pipeline CI/CD • Bronze → Silver → Gold • Analyse des prix des livres")

# =========================
# RAW DATA (expander)
# =========================
with st.expander("📊 Aperçu des données (raw)"):
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df.head(20), use_container_width=True)

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔍 Filters")

# garde-fou si dataset vide
if df.empty or "price" not in df.columns:
    st.error("Dataset vide ou colonne 'price' manquante.")
    st.stop()

price_min_global = float(df["price"].min())
price_max_global = float(df["price"].max())

# slider range (robuste)
price_range = st.sidebar.slider(
    "💰 Plage de prix (£)",
    min_value=price_min_global,
    max_value=price_max_global,
    value=(price_min_global, price_max_global),
    step=0.5
)

# filtres optionnels (si colonnes présentes)
category = None
if "category" in df.columns:
    cats = sorted([c for c in df["category"].dropna().unique()])
    category = st.sidebar.multiselect("📚 Catégories", cats, default=cats)

rating = None
if "rating" in df.columns:
    ratings = sorted([r for r in df["rating"].dropna().unique()])
    rating = st.sidebar.multiselect("⭐ Ratings", ratings, default=ratings)

# =========================
# APPLY FILTERS
# =========================
mask = (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])

if category is not None:
    mask &= df["category"].isin(category)

if rating is not None:
    mask &= df["rating"].isin(rating)

df_filtered = df.loc[mask].copy()

# =========================
# ALERTS / QUALITY
# =========================
alerts_col, actions_col = st.columns([3, 1])
with alerts_col:
    if df_filtered.empty:
        st.warning("Aucune donnée avec ces filtres.")
    elif df_filtered["price"].nunique() == 1:
        st.info("Tous les prix sont identiques dans la sélection.")
with actions_col:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if st.button("🔄 Refresh data"):
        load_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# =========================
# KPIs
# =========================
def safe_mean(s):
    return float(s.mean()) if len(s) else 0.0

def safe_max(s):
    return float(s.max()) if len(s) else 0.0

def safe_min(s):
    return float(s.min()) if len(s) else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("📦 Nb livres", f"{len(df_filtered):,}")
k2.metric("💰 Prix moyen", f"{safe_mean(df_filtered['price']):.2f} £")
k3.metric("📈 Prix max", f"{safe_max(df_filtered['price']):.2f} £")
k4.metric("📉 Prix min", f"{safe_min(df_filtered['price']):.2f} £")

st.markdown("---")

# =========================
# CHARTS (Plotly)
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Distribution des prix")
    if not df_filtered.empty:
        fig_hist = px.histogram(
            df_filtered,
            x="price",
            nbins=30,
            title="Distribution des prix",
        )
        fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.empty()

with c2:
    st.subheader("📦 Prix par catégorie")
    if "category" in df_filtered.columns and not df_filtered.empty:
        df_cat = (
            df_filtered.groupby("category", as_index=False)["price"]
            .mean()
            .sort_values("price", ascending=False)
        )
        fig_bar = px.bar(
            df_cat,
            x="category",
            y="price",
            title="Prix moyen par catégorie",
        )
        fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Colonne 'category' non disponible.")

# =========================
# TABLE + EXPORT
# =========================
st.subheader("📄 Données filtrées")

col_t1, col_t2 = st.columns([4, 1])
with col_t1:
    st.dataframe(df_filtered, use_container_width=True)

with col_t2:
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export CSV",
        data=csv,
        file_name=f"books_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# =========================
# INSIGHTS (storytelling)
# =========================
st.markdown("## 🧠 Insights")
if not df_filtered.empty:
    p_mean = safe_mean(df_filtered["price"])
    p_max = safe_max(df_filtered["price"])
    p_min = safe_min(df_filtered["price"])
    st.write(
        f"- Le prix moyen est **{p_mean:.2f} £**, avec un max à **{p_max:.2f} £** et un min à **{p_min:.2f} £**."
    )
    if "category" in df_filtered.columns:
        top_cat = (
            df_filtered.groupby("category")["price"].mean().sort_values(ascending=False)
        )
        if not top_cat.empty:
            st.write(f"- Catégorie la plus chère en moyenne : **{top_cat.index[0]}**.")
else:
    st.write("- Ajuste les filtres pour voir les insights.")