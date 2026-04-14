import pandas as pd
from google.cloud import bigquery

def build_gold():
    df = pd.read_csv("data/silver/books_clean.csv")

    if "category" in df.columns:
        df_agg = df.groupby("category", as_index=False)["price"].mean()
    else:
        df_agg = df.groupby("page", as_index=False)["price"].mean()

    client = bigquery.Client(project="domainecareycabaneasucre")

    table_id = "domainecareycabaneasucre.books.books_agg"

    job = client.load_table_from_dataframe(df_agg, table_id)
    job.result()

    print("✅ Data uploaded to BigQuery")

if __name__ == "__main__":
    build_gold()