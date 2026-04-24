"""NexusGuard – Detetor de Anomalias e Fraude em Tempo Real."""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="NexusGuard API", version="0.1.0", root_path="/nexusguard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/app/models/isolation_forest.pkl"
SCALER_PATH = "/app/models/scaler.pkl"

model: IsolationForest | None = None
scaler: StandardScaler | None = None


class DataPoint(BaseModel):
    features: list[float]


class BatchData(BaseModel):
    data: list[list[float]]


@app.on_event("startup")
def load_or_create_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        # Create a default model (will be retrained with real data)
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        scaler = StandardScaler()
        # Fit on dummy data so the model is ready
        dummy = np.random.randn(200, 5)
        scaler.fit(dummy)
        model.fit(scaler.transform(dummy))
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(point: DataPoint):
    """Predict if a single data point is an anomaly (-1) or normal (1)."""
    arr = np.array(point.features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = int(model.predict(arr_scaled)[0])
    score = float(model.decision_function(arr_scaled)[0])
    return {
        "prediction": prediction,
        "is_anomaly": prediction == -1,
        "anomaly_score": score,
    }


@app.post("/predict/batch")
def predict_batch(batch: BatchData):
    """Predict anomalies on a batch of data points."""
    arr = np.array(batch.data)
    arr_scaled = scaler.transform(arr)
    predictions = model.predict(arr_scaled).tolist()
    scores = model.decision_function(arr_scaled).tolist()
    return {
        "predictions": predictions,
        "anomaly_count": predictions.count(-1),
        "total": len(predictions),
        "scores": scores,
    }


@app.post("/train")
async def train(file: UploadFile = File(...)):
    """Retrain the model from a CSV file upload."""
    global model, scaler
    try:
        df = pd.read_csv(file.file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].dropna().values

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(data_scaled)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        return {
            "status": "trained",
            "samples": len(data),
            "features": len(numeric_cols),
        }
    finally:
        await file.close()
