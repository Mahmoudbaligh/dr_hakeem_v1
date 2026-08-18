import json
import numpy as np
import onnxruntime as ort
import gradio as gr

from PIL import Image


# =========================
# Load Model
# =========================

MODEL_PATH = "model.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# =========================
# Load Classes
# =========================

with open("class_names.json", "r", encoding="utf-8") as f:
    data = json.load(f)

class_names = data["class_names"]
class_labels = data["class_labels"]


# =========================
# Preprocessing
# =========================

def preprocess_image(image):

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

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # Add batch
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


# =========================
# Softmax
# =========================

def softmax(x):

    exp_x = np.exp(x - np.max(x))

    return exp_x / np.sum(exp_x)


# =========================
# Prediction
# =========================

def predict(image):

    if image is None:
        return "Please upload an image."

    input_tensor = preprocess_image(image)

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    logits = outputs[0]

    probabilities = softmax(logits[0])

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_code = class_names[predicted_index]

    predicted_class = class_labels[predicted_code]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return {
        "Prediction": predicted_class,
        "Code": predicted_code,
        "Confidence": f"{confidence:.2f}%"
    }


# =========================
# Gradio Interface
# =========================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload Skin Image"
    ),
    outputs=gr.JSON(
        label="Prediction"
    ),
    title="Skin Disease Classification",
    description="Upload a skin lesion image to classify it."
)


demo.launch()