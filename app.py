import streamlit as st
import cv2
import numpy as np
from PIL import Image

from utils.yolo_utils import detect_cats
from utils.tflite_utils import predict_breed

# ---------- IOU FUNCTION ----------
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cat Breed Prediction",
    page_icon="🐱",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
    .section-card {
        background-color: #f9f9f9;
        padding: 16px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        color: #333;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------- TITLE ----------------
st.markdown("<div class='main-title'>🐱 Cat Breed Prediction App</div>", unsafe_allow_html=True)
st.write("")

# ---------------- INPUT METHOD ----------------
st.markdown("""
<div class="section-card">
    <div class="section-title">📸 Select Input Method</div>
    Upload an image or use the camera to capture a cat. Supports multiple cats in one image.
</div>
""", unsafe_allow_html=True)

option = st.radio("", ["📂 Upload Image", "📷 Use Camera"], horizontal=True)

image = None
is_grayscale = False

if option == "📂 Upload Image":
    uploaded = st.file_uploader("Choose a cat image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
else:
    cam = st.camera_input("Capture a cat image")
    if cam:
        image = Image.open(cam)

# ---------------- PROCESS ----------------
if image is not None:

    if image.mode == "L":
        is_grayscale = True

    image = image.convert("RGB")

    st.markdown("<div class='section-card'><div class='section-title'>🖼️ Input Image</div></div>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ---------------- GRAYSCALE FLOW ----------------
    if is_grayscale:
        st.warning("⚠️ Grayscale image detected. Converted to RGB automatically.")
        if st.button("🔮 Predict Cat Breed", use_container_width=True):
            result = predict_breed(img_np)
            st.markdown("<div class='section-card'><div class='section-title'>🐾 Prediction Result</div></div>", unsafe_allow_html=True)
            if result["status"] == "unknown":
                st.warning("Breed not in training dataset")
                st.info(f"Closest match: {result['closest_breed']}")
            else:
                st.success(f"Breed: {result['breed']}")
                st.info(f"Confidence: {result['level']}")
        st.stop()

    # ---------------- YOLO DETECTION ----------------
    raw_boxes = detect_cats(img_cv)

    # Sort boxes by area (small → large)
    raw_boxes = sorted(raw_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

    # --------- IoU Filtering ----------
    filtered_boxes = []
    for box in raw_boxes:
        keep = True
        for kept in filtered_boxes:
            if calculate_iou(box, kept) > 0.6:  # slightly relaxed
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    # --------- Allow both small and large boxes ----------
    h, w, _ = img_cv.shape
    image_area = h * w
    boxes = []
    min_area_ratio = 0.005  # very small faces allowed
    max_area_ratio = 0.85   # avoid huge boxes that cover most of the image

    for box in filtered_boxes:
        box_area = (box[2]-box[0])*(box[3]-box[1])
        if min_area_ratio < box_area/image_area < max_area_ratio:
            boxes.append(box)

    st.info(f"🐈 Total cats detected: {len(boxes)}")

    if len(boxes) == 0:
        st.error("❌ No cat detected. Please upload a clear image.")
        st.stop()

    # ---------------- DRAW BOXES ----------------
    boxed_img = img_cv.copy()
    crops = []
    labels = []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(boxed_img, f"Cat {i+1}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        crop = img_cv[y1:y2, x1:x2]
        crops.append(crop)
        labels.append(f"Cat {i+1}")

    st.markdown("<div class='section-card'><div class='section-title'>😻 Detected Cats</div></div>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB), caption="Detected cats (select one below)", use_container_width=True)

    # ---------------- SELECT CAT ----------------
    selected_cat = st.radio("Select the cat to predict", labels, horizontal=True)
    selected_index = labels.index(selected_cat)
    selected_crop = crops[selected_index]

    st.image(cv2.cvtColor(selected_crop, cv2.COLOR_BGR2RGB),
             caption=f"Selected {selected_cat}",
             use_container_width=True)

    # ---------------- PREDICT ----------------
    if st.button("🔮 Predict Cat Breed", use_container_width=True):
        result = predict_breed(selected_crop)
        st.markdown("<div class='section-card'><div class='section-title'>🐾 Prediction Result</div></div>", unsafe_allow_html=True)
        if result["status"] == "unknown":
            st.warning("⚠️ Breed not in training dataset")
            st.info(f"😺 Closest match: {result['closest_breed']}")
        else:
            st.success(f"😸 Breed: {result['breed']}")
            st.info(f"📊 Confidence: {result['level']}")