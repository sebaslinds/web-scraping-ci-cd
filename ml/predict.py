import joblib
import pandas as pd

model = joblib.load("ml/model.pkl")

def predict_price(page, title_length):
    input_df = pd.DataFrame([{
        "page": page,
        "title_length": title_length
    }])
    prediction = model.predict(input_df)[0]
    return prediction