import streamlit as st
import cv2                       # ✅ FIXED
import numpy as np
from PIL import Image

# Safe imports
try:
    from utils.yolo_utils import detect_cats
    from utils.tflite_utils import predict_breed
except Exception as e:
    st.error("❌ Failed to load model utilities.")
    st.code(str(e))
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cat Breed Prediction",
    page_icon="🐱",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align: center; font-size: 45px;'>🐱 Cat Breed Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 20px;'>Upload or capture a cat image. "
    "The system will <b>detect the cat</b> and <b>predict its breed</b>.</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- IMAGE INPUT ----------------
st.markdown(
    "<h2 style='font-size: 30px;'>📸 Select Image Input Method</h2>",
    unsafe_allow_html=True
)

option = st.radio(
    label="",
    options=["📂 Upload Image", "📷 Use Camera"],
    horizontal=True
)

image = None

if option == "📂 Upload Image":
    uploaded = st.file_uploader(
        "Choose a cat image (JPG / PNG / JPEG)",
        type=["jpg", "png", "jpeg"]
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
else:
    cam = st.camera_input("Capture a clear cat image")
    if cam:
        image = Image.open(cam).convert("RGB")

# ---------------- IMAGE PROCESSING ----------------
if image is not None:
    st.divider()
    st.image(image, caption="Uploaded / Captured Image", use_container_width=True)

    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ---------------- YOLO DETECTION ----------------
    st.divider()
    st.markdown(
        "<h2 style='font-size: 30px;'>🔍 Cat Detection</h2>",
        unsafe_allow_html=True
    )

    boxes = detect_cats(img_cv)

    if not boxes:
        st.error("❌ No cat detected. Please upload a clear image.")
        st.stop()

    st.success(f"✅ {len(boxes)} cat(s) detected!")

    selected_box = boxes[0]

    if len(boxes) > 1:
        st.warning("⚠️ Multiple cats detected. Select one.")
        index = st.selectbox("Select Cat Index", range(len(boxes)))
        selected_box = boxes[index]

    x1, y1, x2, y2 = selected_box
    cat_crop = img_cv[y1:y2, x1:x2]

    st.image(
        cv2.cvtColor(cat_crop, cv2.COLOR_BGR2RGB),
        caption="🐾 Selected Cat",
        use_container_width=True
    )

    # ---------------- PREDICTION ----------------
    st.divider()
    if st.button("🔮 Predict Cat Breed", use_container_width=True):

        result = predict_breed(cat_crop)

        st.markdown(
            "<h2 style='text-align:center; font-size: 32px;'>🐾 Prediction Result</h2>",
            unsafe_allow_html=True
        )

        if result["status"] == "unknown":
            st.warning("⚠️ This breed is not present in our training dataset")
            st.info(f"🔍 Closest matching breed: **{result['closest_breed']}**")
        else:
            st.success(f"🐱 **Breed:** {result['breed']}")
            st.info(f"📊 **Confidence Level:** {result['level']}")
