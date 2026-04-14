📊 Book Analytics Dashboard
🚀 Overview

This project demonstrates an end-to-end data product built from web-scraped book data.
It combines data engineering, analytics, and machine learning into a fully interactive dashboard.

The application enables users to explore pricing patterns, monitor key metrics, and generate predictive insights through a streamlined interface.

🎯 Objective

The goal of this project is to:

Transform raw scraped data into a structured analytics layer
Provide interactive data exploration and business-ready KPIs
Apply machine learning to predict book prices
Showcase a production-style data pipeline with CI/CD
🧱 Architecture (Medallion Design)

The data pipeline follows a modern Medallion Architecture:

Bronze → Silver → Gold
🔹 Bronze Layer
Raw scraped data
Minimal processing
Stored as-is for traceability
🔹 Silver Layer
Data cleaning and normalization
Type casting (price, availability, etc.)
Removal of inconsistencies
🔹 Gold Layer
Aggregated datasets
Analytics-ready structure
Optimized for dashboard consumption
⚙️ Pipeline

The pipeline is fully automated and reproducible:

🌐 Web scraping (books dataset)
🧹 Data cleaning & transformation
📦 Structured storage (CSV layers)
🔁 CI/CD with GitHub Actions
🚀 Deployment-ready Streamlit app
📈 Features
🔍 Interactive Analytics
Dynamic filtering (price range)
Real-time updates
Raw data inspection
📊 Business KPIs
Total number of books
Average price
Maximum price
Minimum price
📉 Data Visualization
Price distribution histogram
Boxplot for price variability
🤖 Machine Learning
Linear regression model
Predicts price from available features
Evaluation metrics:
MAE (Mean Absolute Error)
R² Score
🖥️ Dashboard

The dashboard is built with Streamlit and designed with a clean SaaS-style UI.

Key components:
KPI cards (fully responsive)
Interactive charts (Plotly)
Data table with export option
Refresh mechanism with caching
ML metrics section
🛠️ Tech Stack
Layer	Technology
Scraping	Python (requests / BeautifulSoup)
Data Processing	Pandas
Visualization	Plotly
App Framework	Streamlit
Machine Learning	Scikit-learn
CI/CD	GitHub Actions
🚀 Deployment

The application can be deployed on:

Streamlit Cloud

📦 Installation
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt

streamlit run app.py
📊 Example Use Cases
Analyze pricing trends in scraped datasets
Build end-to-end data pipelines for portfolios
Demonstrate data engineering + ML integration
Create interactive dashboards for stakeholders
🧠 Key Learnings
Designing scalable data pipelines (Bronze → Silver → Gold)
Building interactive dashboards with Streamlit
Integrating machine learning into analytics workflows
Structuring a project for production readiness
📌 Future Improvements
Add more predictive features (category, rating, etc.)
Deploy model as an API
Add real-time data ingestion
Enhance ML model (Random Forest)
👨‍💻 Author

Sébastien Lindsay
