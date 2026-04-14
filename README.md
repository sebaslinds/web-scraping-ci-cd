\# 📊 Web Scraping CI/CD Pipeline



\## 🚀 Description



Ce projet implémente un pipeline automatisé de web scraping en Python.



Il extrait des données de livres depuis le site :

http://books.toscrape.com/



Les données sont nettoyées et sauvegardées sous forme de fichiers CSV versionnés avec timestamp.



Le pipeline est exécuté automatiquement grâce à GitHub Actions (CI/CD).



\---



\## 🧱 Stack Technique



\- Python 3.10

\- requests

\- BeautifulSoup4

\- pandas

\- GitHub Actions (CI/CD)



\---



\## ⚙️ Fonctionnalités



\- ✅ Scraping web automatisé

\- ✅ Nettoyage des données (regex)

\- ✅ Gestion des erreurs

\- ✅ Création automatique du dossier `data/`

\- ✅ Sauvegarde des données avec timestamp

\- ✅ Pipeline CI/CD automatisé



\---



\## 📁 Structure du projet





web-scraping-ci-cd/

│

├── scraper.py

├── requirements.txt

├── README.md

├── data/

│

└── .github/

└── workflows/

└── pipeline.yml





\---



\## ▶️ Exécution locale



\### 1. Installer les dépendances





pip install -r requirements.txt





\### 2. Lancer le script





python scraper.py





\### 3. Résultat



Un fichier CSV est généré dans le dossier `data/` :





data/books\_YYYYMMDD\_HHMMSS.csv





\---



\## ⚙️ CI/CD avec GitHub Actions



Le pipeline s’exécute automatiquement :



\- à chaque `git push`

\- tous les jours à midi (UTC)



\### Étapes du pipeline :



1\. Installation de Python

2\. Installation des dépendances

3\. Exécution du script

4\. Sauvegarde des données

5\. Commit automatique des nouveaux fichiers



\---



\## 📊 Exemple de données



| title | price | availability |

|------|------|-------------|

| Book Title | 51.77 | In stock |



\---



\## 💡 Améliorations futures



\- Pagination (scraping multi-pages)

\- Logging avancé

\- Tests unitaires (pytest)

\- Dockerisation

\- Data pipeline (Bronze / Silver / Gold)

\- Intégration avec une base de données



\---



\## 🧠 Objectif du projet



Ce projet démontre :



\- Automatisation de pipeline data

\- Web scraping robuste

\- Bonnes pratiques Python

\- Mise en place d’un CI/CD

\- Approche Data Engineering



\---



\## 👤 Auteur



Sebastien Lindsay

"# web-scraping-ci-cd" 
