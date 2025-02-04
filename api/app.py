from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = "../models/llm_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.post("/predict")
def predict(input_data):
    try:
        input_df = pd.DataFrame([input_data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
def home():
    return {"status": "OK"}