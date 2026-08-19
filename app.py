import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.onnx"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


# =========================
# Load Model
# =========================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# =========================
# Load Classes
# =========================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

class_names = data["class_names"]
class_labels = data["class_labels"]


# =========================
# FastAPI
# =========================

app = FastAPI(
    title="Dr. Hakeem API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Preprocessing
# =========================

def preprocess_image(image: Image.Image):

    image = image.convert("RGB")
    image = image.resize((300, 300))

    image = np.array(image).astype(np.float32)
    image = image / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


# =========================
# Prediction
# =========================

def predict(image: Image.Image):

    input_tensor = preprocess_image(image)

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    logits = outputs[0]

    exp_x = np.exp(
        logits[0] - np.max(logits[0])
    )

    probabilities = exp_x / np.sum(exp_x)

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_code = class_names[predicted_index]

    predicted_class = class_labels[predicted_code]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return {
        "prediction": predicted_class,
        "code": predicted_code,
        "confidence": round(confidence, 2)
    }


# =========================
# Routes
# =========================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Dr. Hakeem API"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    return predict(image)
