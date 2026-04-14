import pandas as pd

def build_gold():
    df = pd.read_csv("data/silver/books_clean.csv")

    if "category" in df.columns:
        df_agg = df.groupby("category", as_index=False)["price"].mean()
    else:
        df_agg = df.groupby("page", as_index=False)["price"].mean()

    df_agg.to_csv("data/gold/books_agg.csv", index=False)
    print("Gold layer created: data/gold/books_agg.csv")

if __name__ == "__main__":
    build_gold()