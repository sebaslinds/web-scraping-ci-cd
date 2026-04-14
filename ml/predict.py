import joblib
import pandas as pd

model = joblib.load("ml/model.pkl")

def predict_price(page, title_length):
    df = pd.DataFrame([{
        "page": page,
        "title_length": title_length
    }])
    return model.predict(df)[0]