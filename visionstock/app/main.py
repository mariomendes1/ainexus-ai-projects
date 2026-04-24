"""VisionStock – Previsão de Inventário Inteligente."""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

app = FastAPI(title="VisionStock API", version="0.1.0", root_path="/visionstock")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "/app/models"
rf_model = None
xgb_model = None
scaler = None


class PredictionRequest(BaseModel):
    product_id: str
    historical_sales: list[float]
    days_ahead: int = 30


class ProductData(BaseModel):
    product_id: str
    current_stock: int
    avg_daily_sales: float
    lead_time_days: int = 7


def create_features(series: list[float]) -> np.ndarray:
    """Create time series features from a sales history."""
    arr = np.array(series)
    features = []
    for i in range(7, len(arr)):
        window = arr[i - 7 : i]
        features.append(
            [
                window.mean(),
                window.std(),
                window.min(),
                window.max(),
                arr[i - 1],
                arr[i - 3 : i].mean(),
                arr[i - 7 : i].mean(),
            ]
        )
    return np.array(features) if features else np.empty((0, 7))


@app.on_event("startup")
def load_models():
    global rf_model, xgb_model, scaler
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        xgb_model = joblib.load(xgb_path)
        scaler = joblib.load(scaler_path)
    else:
        # Create default models with synthetic data
        np.random.seed(42)
        sales = np.abs(
            np.random.normal(50, 15, 365) + 10 * np.sin(np.linspace(0, 4 * np.pi, 365))
        )
        X = create_features(sales.tolist())
        y = sales[7:]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_scaled, y)

        xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_model.fit(X_scaled, y)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(rf_model, rf_path)
        joblib.dump(xgb_model, xgb_path)
        joblib.dump(scaler, scaler_path)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": rf_model is not None}


@app.post("/predict")
def predict(req: PredictionRequest):
    """Predict future sales for a product."""
    if len(req.historical_sales) < 8:
        return {"error": "Necessário pelo menos 8 pontos de dados históricos."}

    predictions_rf = []
    predictions_xgb = []
    current = list(req.historical_sales)

    for _ in range(req.days_ahead):
        features = create_features(current)
        if len(features) == 0:
            break
        last_features = scaler.transform(features[-1:])
        pred_rf = max(0, float(rf_model.predict(last_features)[0]))
        pred_xgb = max(0, float(xgb_model.predict(last_features)[0]))
        predictions_rf.append(round(pred_rf, 2))
        predictions_xgb.append(round(pred_xgb, 2))
        # Use ensemble average for next step
        current.append((pred_rf + pred_xgb) / 2)

    today = datetime.now()
    dates = [
        (today + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(len(predictions_rf))
    ]

    return {
        "product_id": req.product_id,
        "predictions": {
            "random_forest": predictions_rf,
            "xgboost": predictions_xgb,
            "ensemble": [
                round((a + b) / 2, 2) for a, b in zip(predictions_rf, predictions_xgb)
            ],
        },
        "dates": dates,
        "total_predicted_demand": round(
            sum((a + b) / 2 for a, b in zip(predictions_rf, predictions_xgb)), 2
        ),
    }


@app.post("/urgency")
def stock_urgency(product: ProductData):
    """Calculate stock urgency level (green/yellow/red)."""
    days_of_stock = product.current_stock / max(product.avg_daily_sales, 0.01)
    if days_of_stock > product.lead_time_days * 2:
        urgency = "green"
        message = "Stock suficiente"
    elif days_of_stock > product.lead_time_days:
        urgency = "yellow"
        message = "Encomendar em breve"
    else:
        urgency = "red"
        message = "Stock crítico – encomendar já!"

    return {
        "product_id": product.product_id,
        "urgency": urgency,
        "days_of_stock": round(days_of_stock, 1),
        "message": message,
    }


@app.post("/train")
async def train(file: UploadFile = File(...)):
    """Retrain models from a CSV with sales history."""
    global rf_model, xgb_model, scaler
    try:
        df = pd.read_csv(file.file)
        sales_col = [c for c in df.columns if "sale" in c.lower() or "vend" in c.lower()]
        if not sales_col:
            sales_col = df.select_dtypes(include=[np.number]).columns[:1]
        sales = df[sales_col[0]].dropna().values

        X = create_features(sales.tolist())
        y = sales[7:]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_scaled, y)

        xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_model.fit(X_scaled, y)

        joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
        joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

        return {"status": "trained", "samples": len(sales), "features": X.shape[1]}
    finally:
        await file.close()
