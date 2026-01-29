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

st.markdown(
    """
    <h2 style='text-align: center;'>📸 Select Input Method</h2>
    <p style='text-align: center; font-size:18px;'>
        Upload or capture a cat image. The system will detect the cat and predict its breed.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- LARGE BUTTON INPUT ----------------
col1, col2 = st.columns(2)

image = None
is_grayscale = False

with col1:
    upload_clicked = st.button("📂 Upload Image", use_container_width=True)

with col2:
    camera_clicked = st.button("📷 Use Camera", use_container_width=True)

# ---------------- HANDLE UPLOAD ----------------
if upload_clicked:
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image = Image.open(uploaded)

# ---------------- HANDLE CAMERA ----------------
if camera_clicked:
    cam = st.camera_input("Capture a photo")
    if cam:
        image = Image.open(cam)

# ---------------- PROCESS ----------------
if image is not None:

    # ✅ Detect grayscale BEFORE conversion
    if image.mode == "L":
        is_grayscale = True

    # Convert to RGB
    image = image.convert("RGB")

    st.image(image, caption="Input Image", use_container_width=True)

    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ---------------- GRAYSCALE FLOW ----------------
    if is_grayscale:
        st.warning(
            "⚠️ Grayscale image detected. "
            "Converted to RGB and directly predicting breed."
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

    # ---------------- COLOR IMAGE → YOLO ----------------
    boxes = detect_cats(img_cv)

    if len(boxes) == 0:
        st.error("❌ No cat detected. Please upload a clear color image.")
        st.stop()

    # ---------------- CROP CAT ----------------
    x1, y1, x2, y2 = boxes[0]
    cat_crop = img_cv[y1:y2, x1:x2]

    st.image(
        cv2.cvtColor(cat_crop, cv2.COLOR_BGR2RGB),
        caption="Detected Cat",
        use_container_width=True
    )

    # ---------------- PREDICT ----------------
    if st.button("🔮 Predict Cat Breed", use_container_width=True):
        result = predict_breed(cat_crop)

        st.subheader("🐾 Prediction Result")

        if result["status"] == "unknown":
            st.warning("⚠️ Breed not in training dataset")
            st.info(f"🐱 Closest match: {result['closest_breed']}")
        else:
            st.success(f"🐱 Breed: {result['breed']}")
            st.info(f"📊 Confidence: {result['level']}")