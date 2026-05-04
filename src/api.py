"""FastAPI server for gesture recognition."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from src.predict import GesturePredictor
from src.utils import resolve_path

logger = logging.getLogger(__name__)
predictor = None


@asynccontextmanager
async def lifespan(app):
    global predictor
    mp = resolve_path("models/best_model.pt")
    xp = mp.with_suffix('.xgb.pkl')
    if not mp.exists() and not xp.exists():
        raise RuntimeError(f"Model not found: {mp}. Run 'python -m src.train' first.")
    predictor = GesturePredictor()
    logger.info("Model loaded at startup")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Gesture Digit Recognition API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    features: list[float]
    model_config = {"json_schema_extra": {"example": {"features": [0.5]*63}}}

class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: dict[str, float]

class BatchPredictRequest(BaseModel):
    samples: list[list[float]]


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 63:
        raise HTTPException(400, f"Expected 63 features, got {len(req.features)}")
    f = np.array(req.features, dtype=np.float32)
    f = np.nan_to_num(f, nan=0.0, posinf=10.0, neginf=-10.0)
    f = np.clip(f, -10, 10)
    proba = predictor.predict_proba(f)
    digit = max(proba, key=proba.get)
    return PredictResponse(digit=int(digit), confidence=proba[digit], probabilities=proba)


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    if not req.samples:
        raise HTTPException(400, "Empty samples list")
    for i, s in enumerate(req.samples):
        if len(s) != 63:
            raise HTTPException(400, f"Sample {i}: expected 63 features, got {len(s)}")
    af = np.array(req.samples, dtype=np.float32)
    af = np.nan_to_num(af, nan=0.0, posinf=10.0, neginf=-10.0)
    af = np.clip(af, -10, 10)
    digits = predictor.predict_batch(af)
    results = [{"digit": d, "confidence": predictor.predict_proba(af[i])[str(d)]} for i, d in enumerate(digits)]
    return {"predictions": results}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None, "val_acc": predictor.val_acc if predictor else None}


def main():
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()