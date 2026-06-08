"""
api/main.py
-----------
FastAPI REST API for Healthcare Disease Prediction Model Deployment.

Endpoints:
    GET  /              -> API info
    GET  /health        -> Health check
    POST /predict       -> Single patient prediction
    POST /predict/batch -> Batch predictions
    GET  /model/info    -> Model metadata
    GET  /metrics       -> Model performance metrics

Run:
    uvicorn api.main:app --reload --port 8000
    Visit: http://127.0.0.1:8000/docs
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI(
    title="Healthcare Disease Prediction API",
    description="Predictive Analytics API for Diabetes/Disease Detection using ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------------------------------
# Load artifacts at startup
# -------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model():
    scaler_path  = os.path.join(BASE, 'models', 'scaler.pkl')
    best_path    = os.path.join(BASE, 'models', 'best_model.pkl')
    rf_path      = os.path.join(BASE, 'models', 'random_forest.pkl')
    feat_path    = os.path.join(BASE, 'models', 'feature_names.pkl')
    metrics_path = os.path.join(BASE, 'reports', 'model_comparison.csv')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feat_path, 'rb') as f:
        feature_names = pickle.load(f)

    model_name = 'Random Forest'
    if os.path.exists(best_path):
        with open(best_path, 'rb') as f:
            best = pickle.load(f)
        model      = best['model']
        model_name = best['name']
    else:
        with open(rf_path, 'rb') as f:
            model = pickle.load(f)

    metrics = {}
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path, index_col=0)
        if model_name in df.index:
            metrics = df.loc[model_name].to_dict()

    return model, scaler, feature_names, model_name, metrics

try:
    MODEL, SCALER, FEATURE_NAMES, MODEL_NAME, MODEL_METRICS = load_model()
    MODEL_LOADED = True
    print(f"[INFO] Model loaded: {MODEL_NAME}")
except Exception as e:
    MODEL_LOADED = False
    print(f"[ERROR] Could not load model: {e}")


# -------------------------------------------------------
# Request / Response Schemas
# -------------------------------------------------------
class PatientInput(BaseModel):
    age              : int   = Field(..., ge=1, le=120,  description="Patient age in years")
    gender           : str   = Field(...,                description="Gender: Male or Female")
    bmi              : float = Field(..., ge=10, le=60,  description="Body Mass Index")
    glucose_level    : float = Field(..., ge=50, le=400, description="Fasting glucose level (mg/dL)")
    blood_pressure   : float = Field(..., ge=40, le=150, description="Diastolic blood pressure (mmHg)")
    insulin          : float = Field(..., ge=0,  le=600, description="Serum insulin (mu U/ml)")
    skin_thickness   : float = Field(..., ge=0,  le=100, description="Triceps skin fold thickness (mm)")
    cholesterol      : float = Field(..., ge=50, le=500, description="Total cholesterol (mg/dL)")
    hba1c            : float = Field(..., ge=3,  le=15,  description="HbA1c percentage")
    family_history   : int   = Field(..., ge=0,  le=1,   description="Family history of diabetes (0/1)")
    smoking_status   : str   = Field(...,                description="Never / Former / Current")
    physical_activity: str   = Field(...,                description="Low / Moderate / High")
    diet_quality     : str   = Field(...,                description="Poor / Average / Good")
    stress_level     : int   = Field(..., ge=1,  le=10,  description="Stress level (1-10)")
    pregnancies      : int   = Field(0,   ge=0,  le=20,  description="Number of pregnancies (females only)")

    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @validator('smoking_status')
    def validate_smoking(cls, v):
        if v not in ['Never', 'Former', 'Current']:
            raise ValueError("smoking_status must be 'Never', 'Former', or 'Current'")
        return v

    @validator('physical_activity')
    def validate_activity(cls, v):
        if v not in ['Low', 'Moderate', 'High']:
            raise ValueError("physical_activity must be 'Low', 'Moderate', or 'High'")
        return v

    @validator('diet_quality')
    def validate_diet(cls, v):
        if v not in ['Poor', 'Average', 'Good']:
            raise ValueError("diet_quality must be 'Poor', 'Average', or 'Good'")
        return v


class PredictionResponse(BaseModel):
    prediction       : int
    prediction_label : str
    confidence       : float
    risk_level       : str
    model_used       : str
    timestamp        : str
    recommendations  : List[str]


class BatchInput(BaseModel):
    patients: List[PatientInput]


# -------------------------------------------------------
# Feature Engineering for Inference
# -------------------------------------------------------
def engineer_features(p: PatientInput) -> np.ndarray:
    gender_enc   = 1 if p.gender == 'Male' else 0
    smoking_map  = {'Never': 0, 'Former': 1, 'Current': 2}
    activity_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    diet_map     = {'Poor': 0, 'Average': 1, 'Good': 2}

    smoking_enc  = smoking_map[p.smoking_status]
    activity_enc = activity_map[p.physical_activity]
    diet_enc     = diet_map[p.diet_quality]

    bmi_glucose_ratio     = p.bmi / (p.glucose_level + 1)
    age_bmi_interaction   = p.age * p.bmi
    glucose_insulin_ratio = p.glucose_level / (p.insulin + 1)
    risk_score = (
        int(p.glucose_level > 140) * 3 +
        int(p.bmi > 30) * 2 +
        int(p.hba1c > 6.5) * 3 +
        p.family_history * 2 +
        int(p.age > 45)
    )
    is_obese     = int(p.bmi >= 30)
    high_glucose = int(p.glucose_level >= 140)
    diabetic_hba1c = int(p.hba1c >= 6.5)

    features = [
        p.age, gender_enc, p.bmi, p.glucose_level, p.blood_pressure,
        p.insulin, p.skin_thickness, p.cholesterol, p.hba1c,
        p.family_history, smoking_enc, activity_enc, diet_enc,
        p.stress_level, p.pregnancies,
        bmi_glucose_ratio, age_bmi_interaction, glucose_insulin_ratio,
        risk_score, is_obese, high_glucose, diabetic_hba1c
    ]
    return np.array(features).reshape(1, -1)


def get_recommendations(p: PatientInput, prediction: int, confidence: float) -> List[str]:
    recs = []
    if p.bmi > 30:
        recs.append("Maintain a healthy weight through diet and regular exercise.")
    if p.glucose_level > 140:
        recs.append("Monitor blood glucose levels regularly and consult a physician.")
    if p.hba1c > 6.5:
        recs.append("HbA1c is elevated. Seek medical evaluation for diabetes management.")
    if p.physical_activity == 'Low':
        recs.append("Increase physical activity to at least 150 minutes per week.")
    if p.smoking_status == 'Current':
        recs.append("Quitting smoking significantly reduces disease risk.")
    if p.diet_quality == 'Poor':
        recs.append("Adopt a balanced diet rich in vegetables, fiber, and low in sugar.")
    if p.stress_level > 7:
        recs.append("Practice stress management techniques such as meditation or yoga.")
    if p.family_history == 1:
        recs.append("With family history of diabetes, regular screening is essential.")
    if not recs:
        recs.append("Maintain your healthy lifestyle with regular checkups.")
    return recs


# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "name"       : "Healthcare Disease Prediction API",
        "version"    : "1.0.0",
        "model"      : MODEL_NAME if MODEL_LOADED else "Not loaded",
        "status"     : "running",
        "docs"       : "/docs",
        "endpoints"  : ["/predict", "/predict/batch", "/health", "/model/info"]
    }


@app.get("/health")
def health_check():
    return {
        "status"       : "healthy" if MODEL_LOADED else "model_not_loaded",
        "model_loaded" : MODEL_LOADED,
        "model_name"   : MODEL_NAME if MODEL_LOADED else None,
        "timestamp"    : datetime.now().isoformat()
    }


@app.get("/model/info")
def model_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name"    : MODEL_NAME,
        "feature_count" : len(FEATURE_NAMES),
        "features"      : FEATURE_NAMES,
        "metrics"       : MODEL_METRICS,
        "task"          : "Binary Classification (Disease Prediction)"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X = engineer_features(patient)
        X_scaled = SCALER.transform(X)
        prediction = int(MODEL.predict(X_scaled)[0])
        confidence = float(MODEL.predict_proba(X_scaled)[0][prediction]) * 100

        if confidence >= 80:
            risk_level = "High Risk" if prediction == 1 else "Low Risk"
        elif confidence >= 60:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Uncertain"

        return PredictionResponse(
            prediction       = prediction,
            prediction_label = "Disease Positive" if prediction == 1 else "Disease Negative",
            confidence       = round(confidence, 2),
            risk_level       = risk_level,
            model_used       = MODEL_NAME,
            timestamp        = datetime.now().isoformat(),
            recommendations  = get_recommendations(patient, prediction, confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for i, patient in enumerate(batch.patients):
        try:
            X = engineer_features(patient)
            X_scaled = SCALER.transform(X)
            pred = int(MODEL.predict(X_scaled)[0])
            conf = float(MODEL.predict_proba(X_scaled)[0][pred]) * 100
            results.append({
                "patient_index"   : i + 1,
                "prediction"      : pred,
                "prediction_label": "Disease Positive" if pred == 1 else "Disease Negative",
                "confidence"      : round(conf, 2)
            })
        except Exception as e:
            results.append({"patient_index": i + 1, "error": str(e)})
    return {"total": len(batch.patients), "predictions": results}


# -------------------------------------------------------
# Run
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
