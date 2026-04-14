# 📚 Book Analytics Dashboard

🚀 **End-to-End Data Engineering + Machine Learning Project**

---

## 🧠 Overview

This project delivers a full **data pipeline + analytics dashboard** to analyze book pricing trends from web-scraped data.

It demonstrates production-ready skills in:

* Data Engineering (ETL pipeline)
* Data Modeling (Bronze → Silver → Gold)
* CI/CD automation
* Machine Learning (price prediction)
* Interactive Dashboard (Streamlit)

---

## 🏗️ Architecture

```
Web Scraping → Bronze → Silver → Gold → Dashboard + ML
```

### Pipeline Stages

| Layer  | Description                  |
| ------ | ---------------------------- |
| Bronze | Raw scraped data             |
| Silver | Cleaned & structured data    |
| Gold   | Aggregated analytics dataset |

---

## ⚙️ Tech Stack

* **Python**
* **Pandas**
* **Plotly**
* **Streamlit**
* **Scikit-learn**
* **GitHub Actions (CI/CD)**

---

## 📊 Dashboard Features

### 🔍 Data Exploration

* Interactive filters (price range, category)
* Raw data preview
* Clean UI (SaaS-style)

### 📈 Analytics

* Price distribution (Histogram)
* Price spread (Boxplot)
* Category analysis

### 📦 KPIs

* Total books
* Average price
* Min / Max prices

### 🤖 Machine Learning

* Random Forest model
* Price prediction feature
* Evaluation metrics:

  * MAE
  * R² score

### 📥 Export

* Download filtered dataset as CSV

---

## 🧪 Machine Learning

The project includes a simple regression model:

* Features:

  * Page number
  * Title length
* Model:

  * Random Forest Regressor
* Metrics:

  * Mean Absolute Error (MAE)
  * R² Score

---

## 🖥️ Local Setup

```bash
git clone https://github.com/sebaslinds/web-scraping-ci-cd.git
cd web-scraping-ci-cd

pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment

The app is deployed using **Streamlit Cloud**.

👉 Add `.streamlit/config.toml`:

```toml
[theme]
base="dark"
primaryColor="#6366F1"
backgroundColor="#0b1020"
secondaryBackgroundColor="#111827"
textColor="#E5E7EB"
```

---

## 🔄 CI/CD

This project includes GitHub Actions for:

* Automated pipeline execution
* Data updates
* Continuous integration

---

## 🎯 Business Value

This dashboard enables:

* Monitoring pricing trends
* Detecting outliers
* Understanding market distribution
* Predicting book prices using ML

---

## 👨‍💻 Author

**Sébastien Lindsay**

* GitHub: https://github.com/sebaslinds
* LinkedIn: https://www.linkedin.com/in/s%C3%A9bastien-lindsay-3a66b4b5/

---

## ⭐ Project Highlights

* Full data pipeline (not just a dashboard)
* Production-ready UI (SaaS style)
* Machine learning integration
* Cloud deployment ready

---

## 🚀 Next Improvements

* Add real-time data ingestion
* Deploy API for predictions
* Improve ML model (feature engineering)
* Add user authentication (multi-user dashboard)

---


