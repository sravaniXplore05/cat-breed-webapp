import numpy as np
import json
from PIL import Image
import tflite_runtime.interpreter as tflite
import streamlit as st
import os

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource(show_spinner="Loading breed classifier...")
def load_tflite():
    model_path = os.path.join("models", "cat_breed_model_v3.tflite")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- LOAD CLASS NAMES ----------------
@st.cache_resource
def load_class_names():
    class_path = os.path.join("models", "class_indices.json")
    with open(class_path, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

class_names = load_class_names()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img = img.resize((256, 256))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- BREED PREDICTION ----------------
def predict_breed(img):
    img = preprocess_image(img)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    max_prob = float(np.max(probs))
    predicted_index = int(np.argmax(probs))
    nearest_breed = class_names.get(predicted_index, "Unknown")

    # 🔴 UNKNOWN BREED
    if max_prob < 0.25:
        return {
            "status": "unknown",
            "closest_breed": nearest_breed
        }

    # 🟢 KNOWN BREED
    if max_prob >= 0.80:
        level = "High"
    elif max_prob >= 0.65:
        level = "Medium"
    else:
        level = "Low"

    return {
        "status": "known",
        "breed": nearest_breed,
        "level": level
    }
