"""Smoke test for the API. Requires `python -m src.train` first. Run: python tests/test_api.py"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api import app
from src.data import load_splits

X, y = load_splits()["test"]
with TestClient(app) as c:
    assert c.get("/health").json()["model_loaded"]
    r = c.post("/predict", json={"pixels": X[0].tolist()})
    assert r.status_code == 200 and abs(sum(r.json()["probabilities"].values()) - 1) < 1e-2
    assert c.post("/predict", json={"pixels": [[[0, 0, 0]]]}).status_code == 422  # wrong shape
    assert c.post("/predict", json={"pixels": [[[1, 2, 3]], [[1]]]}).status_code == 422  # ragged
    assert c.post("/predict", json={"pixels": "nope"}).status_code == 422  # wrong type
    assert c.post("/predict", json={"pixels": (X[0].astype(int) + 300).tolist()}).status_code == 422  # range
    buf = io.BytesIO()
    Image.fromarray(X[1]).resize((200, 200)).save(buf, "PNG")
    assert c.post("/predict/image", files={"file": ("x.png", buf.getvalue(), "image/png")}).status_code == 200
    assert c.post("/predict/image", files={"file": ("x.png", b"junk", "image/png")}).status_code == 400
    assert c.post("/predict/image", files={"file": ("x.png", b"0" * (5 * 1024 * 1024 + 1), "image/png")}).status_code == 413
print("API tests passed")
