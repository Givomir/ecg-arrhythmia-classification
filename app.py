"""
REST API for ECG classification (FastAPI)
=======================================
Serves any of the models (cascade / RF / MLP / CNN) through predict.py.
The model is selected with the ARTIFACT_DIR variable (default: artifacts_cascade).

EDUCATIONAL PROTOTYPE - not a medical device.

Starting:
    ARTIFACT_DIR=artifacts_mlp uvicorn app:app --reload          (bash)
    $env:ARTIFACT_DIR="artifacts_mlp"; uvicorn app:app --reload  (PowerShell)

Test:
    python test_api.py --data_dir /path/to/mit-bih --record 200 --beat 10
"""
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import predict

app = FastAPI(title="ECG Arrhythmia Classifier",
              description="Educational prototype — not a medical device.")


class Beat(BaseModel):
    features: List[float]   # the full feature vector for ONE beat


@app.get("/health")
def health():
    return {'status': 'ok', 'model_type': predict.MODEL_TYPE,
            'artifact_dir': predict.ARTIFACT_DIR,
            'n_features': int(predict._scaler.n_features_in_)}


@app.post("/predict")
def predict_beat(beat: Beat):
    expected = int(predict._scaler.n_features_in_)
    if len(beat.features) != expected:
        raise HTTPException(status_code=422,
                            detail=f"Expected {expected} features, got {len(beat.features)}")
    result = predict.make_prediction(beat.features)
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])
    return result
