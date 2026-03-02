from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.models.inference import predict_faces

app = FastAPI(title="Face Recognition API", version="1.0.0")
MODEL_PATH = Path("models/face_classifier.pt")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "task": "face detection + identity classification",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = Path(tmp.name)

    try:
        result = predict_faces(temp_path, MODEL_PATH)
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
