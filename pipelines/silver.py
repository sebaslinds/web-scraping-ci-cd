import pandas as pd

def clean_data():
    df = pd.read_csv("data/bronze/books_raw.csv")

    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("£", "", regex=False)
            .str.replace("Â", "", regex=False)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "title" in df.columns:
        df["title_length"] = df["title"].astype(str).str.len()

    df = df.dropna(subset=["price"])

    df.to_csv("data/silver/books_clean.csv", index=False)
    print("Silver layer created: data/silver/books_clean.csv")

if __name__ == "__main__":
    clean_data()