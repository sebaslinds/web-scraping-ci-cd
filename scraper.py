import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt


def scrape_books():
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"

    data = []

    for page in range(1, 6):
        print(f"Scraping page {page}...")
        url = base_url.format(page)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.find_all("article", class_="product_pod")

        for book in books:
            title = book.h3.a["title"]

            price_text = book.find("p", class_="price_color").text
            price = float(re.sub(r"[^\d.]", "", price_text))

            availability = book.find("p", class_="instock availability").text.strip()

            data.append({
                "title": title,
                "price": price,
                "availability": availability,
                "page": page
            })

    return pd.DataFrame(data)


def main():
    print("Pipeline en cours...")

    df = scrape_books()

    # BRONZE
    os.makedirs("data/bronze", exist_ok=True)
    df.to_csv("data/bronze/books_raw.csv", index=False)

    # SILVER
    df_clean = df.copy()
    df_clean["price"] = df_clean["price"].astype(float)

    os.makedirs("data/silver", exist_ok=True)
    df_clean.to_csv("data/silver/books_clean.csv", index=False)

    # GOLD
    df_gold = df_clean.groupby("availability")["price"].mean().reset_index()

    os.makedirs("data/gold", exist_ok=True)
    df_gold.to_csv("data/gold/books_agg.csv", index=False)

    # DASHBOARD
    df_gold.plot(kind="bar", x="availability", y="price")
    plt.title("Average Price by Availability")
    plt.tight_layout()
    plt.savefig("data/gold/dashboard.png")

    print("Pipeline terminé avec succès")


if __name__ == "__main__":
    main()