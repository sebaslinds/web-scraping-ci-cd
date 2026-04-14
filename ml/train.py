import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_model():
    df = pd.read_csv("data/silver/books_clean.csv")

    features = [col for col in ["page", "title_length"] if col in df.columns]

    if len(features) == 0:
        raise ValueError("No valid features found for training.")

    X = df[features]
    y = df["price"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "ml/model.pkl")
    print("Model saved to ml/model.pkl")

if __name__ == "__main__":
    train_model()