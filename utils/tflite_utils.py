import numpy as np
import json
from PIL import Image
import tensorflow as tf

# ---------------- LOAD MODEL ----------------
interpreter = tf.lite.Interpreter(model_path="models/cat_breed_model_v3.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- LOAD CLASS NAMES ----------------
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

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

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    max_prob = float(np.max(probs))
    predicted_index = int(np.argmax(probs))
    nearest_breed = class_names[predicted_index]

    # 🔴 UNKNOWN BREED CASE
    if max_prob < 0.55:
        return {
            "status": "unknown",
            "closest_breed": nearest_breed
        }

    # 🟢 KNOWN BREED CASE
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
