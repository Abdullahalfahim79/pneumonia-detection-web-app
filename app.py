import threading
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
import models

app = FastAPI(title="Pneumonia X-ray Analyzer")
lock = threading.Lock()


@app.post("/api/predict")
def predict(file: UploadFile = File(...), method: str = Form("gradcam")):
    if method not in ("gradcam", "scorecam"):
        raise HTTPException(400, "method must be gradcam or scorecam")
    data = file.file.read()
    try:
        with lock:
            return models.analyze(data, method)
    except Exception as e:
        raise HTTPException(400, f"Could not analyze this image: {e}")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
