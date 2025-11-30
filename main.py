from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Modeli yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "DÖF Prediction API çalışıyor!"}

@app.post("/predict")
def predict(features: list):
    """
    Örnek:
    POST /predict
    {
        "features": [5, 2, 1, 0]
    }
    """
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
