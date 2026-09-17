from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "penguins_model.joblib"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]


app = FastAPI(
    title="Palmer Penguins API",
    version="1.0",
)


class PenguinInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: str
    sex: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(data: PenguinInput):
    features = pd.DataFrame([data.model_dump()])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    classes = model.named_steps["classifier"].classes_

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(class_name): float(probability)
            for class_name, probability in zip(classes, probabilities)
        },
    }