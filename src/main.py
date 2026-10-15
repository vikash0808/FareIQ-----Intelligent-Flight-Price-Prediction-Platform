from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Flight Price Prediction API", version="1.0")

# Global variables to hold our model artifacts
model = None
feature_columns = None

@app.on_event("startup")
def load_artifacts():
    global model, feature_columns
    model_path = "artifacts/flight_model.pkl"
    features_path = "artifacts/feature_columns.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise RuntimeError("❌ Model artifacts missing! Run train.py first.")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_columns = pickle.load(f)
    print("✅ Model artifacts loaded successfully into memory!")

# Define the exact data structure expected from the user UI
class FlightInput(BaseModel):
    airline: str
    source_city: str
    departure_time: str
    stops: str
    arrival_time: str
    destination_city: str
    class_type: str  # 'Economy' or 'Business'
    duration: float
    days_left: int

@app.post("/predict")
def predict_fare(data: FlightInput):
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")
        
    try:
        # 1. Create a raw dataframe matching our original format structure
        raw_input = {
            'airline': data.airline,
            'source_city': data.source_city,
            'departure_time': data.departure_time,
            'stops': data.stops,
            'arrival_time': data.arrival_time,
            'destination_city': data.destination_city,
            'class': data.class_type,
            'duration': data.duration,
            'days_left': data.days_left
        }
        input_df = pd.DataFrame([raw_input])
        
        # 2. Process categoricals via One-Hot encoding layout
        categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
        encoded_df = pd.get_dummies(input_df, columns=categorical_cols)
        
        # 3. Align features perfectly with what the XGBoost model expects
        final_features = pd.DataFrame(0, index=[0], columns=feature_columns)
        for col in encoded_df.columns:
            if col in final_features.columns:
                final_features[col] = encoded_df[col].values
                
        # 4. Generate the prediction fare calculation
        predicted_price = model.predict(final_features)[0]
        
        return {
            "status": "success",
            "predicted_price": float(np.round(predicted_price, 2))
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "API backend is responding normally."}