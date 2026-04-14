import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime

def scrape_books():
    url = "http://books.toscrape.com/"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Erreur lors du scraping")

    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    data = []

    for book in books:
        title = book.h3.a["title"]

        price_text = book.find("p", class_="price_color").text
        price = float(re.sub(r"[^\d.]", "", price_text))

        availability = book.find("p", class_="instock availability").text.strip()

        data.append({
            "title": title,
            "price": price,
            "availability": availability
        })

    return pd.DataFrame(data)


def main():
    try:
        print("Scraping en cours...")

        df = scrape_books()

        # Création du dossier data (important pour CI/CD)
        os.makedirs("data", exist_ok=True)

        # Nom de fichier avec timestamp (pro)
        filename = f"data/books_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        df.to_csv(filename, index=False)

        print(f"Fichier sauvegardé: {filename}")

    except Exception as e:
        print("Erreur dans le pipeline:", e)


if __name__ == "__main__":
    main()

try:
    df = scrape_books()
    df.to_csv("data/books.csv", index=False)
    print("Scraping réussi")
except Exception as e:
    print("Erreur dans le pipeline :", e)