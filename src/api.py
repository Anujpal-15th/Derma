"""FastAPI inference service. Run: uvicorn src.api:app --port 8000  (docs at /docs)"""
import io
import json
import logging
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data import CLASSES, ROOT, load_config, to_features
from src.train import apply_temperature

log = logging.getLogger("uvicorn.error")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
DISCLAIMER = "Research prototype for education only - not a diagnostic device."
state = {}


@asynccontextmanager
async def lifespan(app):
    cfg = load_config()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow_uri"]))
    info_path = ROOT / cfg["out_dir"] / "best_model.json"
    try:
        info = json.loads(info_path.read_text())
        uri = os.getenv("MODEL_URI", info["uri"])
        if info.get("flavor") == "pytorch":
            from src.cnn import build_model  # plain state_dict: loads without MLflow's pt2 export
            state["model"] = build_model(ROOT / info["weights"])
        else:
            state["model"] = mlflow.sklearn.load_model(uri)
        state["info"] = info
    except Exception:  # service still starts; /health reports degraded, details only in server log
        log.exception("model load failed")
        state["error"] = "model not loaded. Run `python -m src.train` first."
    yield


app = FastAPI(title="DermaMNIST classifier API", version="1.0", lifespan=lifespan,
              description=f"Skin-lesion class probabilities from dermatoscopic RGB images. {DISCLAIMER}")
(ROOT / "results").mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=ROOT / "results"), name="results")


@app.get("/", include_in_schema=False)
def dashboard():
    """Results dashboard: metrics, plots, and live JSON from every endpoint."""
    return FileResponse(ROOT / "src" / "static" / "index.html")


class PixelInput(BaseModel):
    pixels: list[list[list[int]]] = Field(..., description="28x28x3 nested list of uint8 RGB values",
                                          max_length=28)


class Prediction(BaseModel):
    label: str
    class_index: int
    probabilities: dict[str, float]
    model: str
    disclaimer: str = DISCLAIMER


def _predict(img: np.ndarray) -> Prediction:
    """img: uint8 RGB array (H, W, 3). The CNN resizes it itself; sklearn models need 28x28."""
    if "model" not in state:
        raise HTTPException(503, state.get("error", "model not loaded"))
    if state["info"].get("flavor") == "pytorch":
        from src.cnn import predict_proba
        proba = predict_proba(state["model"], img[None])
    else:
        if img.shape[:2] != (28, 28):
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((28, 28), Image.Resampling.BICUBIC))
        proba = state["model"].predict_proba(to_features(img))
    proba = apply_temperature(proba, state["info"].get("temperature", 1.0))[0]  # val-fitted calibration
    i = int(proba.argmax())
    return Prediction(label=CLASSES[i], class_index=i, model=state["info"]["model"],
                      probabilities={c: round(float(p), 4) for c, p in zip(CLASSES, proba)})


@app.get("/health")
def health():
    ok = "model" in state
    return {"status": "ok" if ok else "degraded", "model_loaded": ok,
            "model": state.get("info"), "error": state.get("error")}


@app.post("/predict", response_model=Prediction)
def predict(body: PixelInput):
    try:
        img = np.array(body.pixels, dtype=np.int64)
    except ValueError:  # ragged nested lists
        raise HTTPException(422, "pixels must be a rectangular 28x28x3 array")
    if img.shape != (28, 28, 3):
        raise HTTPException(422, f"expected shape (28, 28, 3), got {img.shape}")
    if img.min() < 0 or img.max() > 255:
        raise HTTPException(422, "pixel values must be in 0..255")
    return _predict(img.astype(np.uint8))


@app.post("/predict/image", response_model=Prediction)
async def predict_image(file: UploadFile = File(...)):
    """Any image file (<=5 MB). Centre-cropped to a square, resized to 128px (the model's own preprocessing does the rest)."""
    from PIL import Image
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file larger than 5 MB")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise HTTPException(413, "image dimensions too large")
        img = img.convert("RGB")
        side = min(img.size)  # centre square crop keeps lesion proportions
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((128, 128), Image.Resampling.BICUBIC)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "could not decode image")
    return _predict(np.asarray(img))
