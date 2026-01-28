import streamlit as st
import cv2
import numpy as np
from PIL import Image

from utils.yolo_utils import detect_cats
from utils.tflite_utils import predict_breed

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cat Breed Prediction",
    page_icon="🐱",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align: center;'>🐱 Cat Breed Prediction App</h1>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUT METHOD INFO ----------------
st.markdown(
    """
    <h2 style='text-align: center;'>
        Select Input Method
    </h2>
    <p style='text-align: center; font-size: 18px;'>
        Choose how you want to provide the cat image
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- IMAGE INPUT ----------------
option = st.radio(
    "",
    ["📂 Upload Image", "📷 Use Camera"],
    horizontal=True
)

image = None

if option == "📂 Upload Image":
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

else:
    cam = st.camera_input("Capture Image")
    if cam:
        image = Image.open(cam).convert("RGB")

# ---------------- PROCESS ----------------
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    img_np = np.array(image)
    is_grayscale = False

    # Detect grayscale
    if len(img_np.shape) == 2:
        is_grayscale = True
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ---------------- YOLO ----------------
    boxes = detect_cats(img_cv)

    # ❗ FALLBACK FOR GRAYSCALE
    if len(boxes) == 0 and is_grayscale:
        st.warning(
            "⚠️ Grayscale image detected. "
            "YOLO failed. Using full image for prediction."
        )

        result = predict_breed(img_np)

        st.subheader("🐾 Prediction Result")

        if result["status"] == "unknown":
            st.warning("Breed not in training dataset")
            st.info(f"Closest match: {result['closest_breed']}")
        else:
            st.success(f"Breed: {result['breed']}")
            st.info(f"Confidence: {result['level']}")

        st.stop()

    # ❌ No cat detected
    if len(boxes) == 0:
        st.error("❌ No cat detected. Please upload a clear color image.")
        st.stop()

    # ---------------- CROP ----------------
    x1, y1, x2, y2 = boxes[0]
    cat_crop = img_cv[y1:y2, x1:x2]

    st.image(
        cv2.cvtColor(cat_crop, cv2.COLOR_BGR2RGB),
        caption="Detected Cat",
        use_container_width=True
    )

    # ---------------- PREDICT ----------------
    if st.button("🔮 Predict Cat Breed"):
        result = predict_breed(cat_crop)

        st.subheader("🐾 Prediction Result")

        if result["status"] == "unknown":
            st.warning("Breed not in training dataset")
            st.info(f"Closest match: {result['closest_breed']}")
        else:
            st.success(f"Breed: {result['breed']}")
            st.info(f"Confidence: {result['level']}")