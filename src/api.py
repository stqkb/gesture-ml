"""FastAPI server for gesture recognition."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

from src.predict import GesturePredictor

app = FastAPI(
    title="Gesture Digit Recognition API",
    description="Recognize hand gesture digits (0-9) from MediaPipe landmarks",
    version="1.0.0",
)

# Global predictor (loaded once at startup)
predictor: Optional[GesturePredictor] = None


class PredictRequest(BaseModel):
    features: list[float]  # 63-dim landmark vector

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5] * 63
            }
        }


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: dict[str, float]


class BatchPredictRequest(BaseModel):
    samples: list[list[float]]  # list of 63-dim vectors


@app.on_event("startup")
def load_model():
    global predictor
    predictor = GesturePredictor()
    print("✅ Model loaded at startup")


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 63:
        raise HTTPException(400, f"Expected 63 features, got {len(req.features)}")

    proba = predictor.predict_proba(np.array(req.features))
    digit = max(proba, key=proba.get)

    return PredictResponse(
        digit=int(digit),
        confidence=proba[digit],
        probabilities=proba,
    )


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    results = []
    for sample in req.samples:
        if len(sample) != 63:
            raise HTTPException(400, f"Expected 63 features, got {len(sample)}")
        proba = predictor.predict_proba(np.array(sample))
        digit = max(proba, key=proba.get)
        results.append({"digit": int(digit), "confidence": proba[digit]})
    return {"predictions": results}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "val_acc": predictor.val_acc if predictor else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
