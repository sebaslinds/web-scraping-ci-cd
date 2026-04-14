import pandas as pd
from google.cloud import bigquery

def build_gold():
    df = pd.read_csv("data/silver/books_clean.csv")

    # =========================
    # FEATURE ENGINEERING
    # =========================

    # title_length
    if "title" in df.columns:
        df["title_length"] = df["title"].astype(str).str.len()

    # fallback page si absent
    if "page" not in df.columns:
        df["page"] = 100

    # =========================
    # DATA CLEANING
    # =========================

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # =========================
    # SELECT COLUMNS FOR GOLD
    # =========================

    df_gold = df[["price", "page", "title_length"]]

    # =========================
    # LOAD TO BIGQUERY
    # =========================

    client = bigquery.Client(project="domainecareycabaneasucre")

    table_id = "domainecareycabaneasucre.books.books_agg"

    job = client.load_table_from_dataframe(df_gold, table_id)
    job.result()

    print("✅ Data uploaded to BigQuery")

if __name__ == "__main__":
    build_gold()