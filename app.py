import streamlit as st
st.set_page_config(page_title="Bottle Detection & Classification", layout="wide")

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import os

# ---------- Load Models Once ----------
@st.cache_resource
def load_models():
    assert os.path.exists("./model/yolo.pt"), "YOLO model file not found!"
    assert os.path.exists("./model/mobilenetv2-tuning.keras"), "MobileNetV2 model file not found!"
    yolo_model = YOLO("./model/yolo.pt")
    mobilenet_model = tf.keras.models.load_model("./model/mobilenetv2-tuning.keras")
    return yolo_model, mobilenet_model

yolo_model, mobilenet_model = load_models()

# ---------- Helper functions ----------

def preprocess_image(img_pil):
    """Convert PIL image to OpenCV BGR numpy array."""
    img = np.array(img_pil)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def show_image(img, title=None):
    """Show image in Streamlit (expects BGR OpenCV format)."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if title:
        st.image(img_rgb, caption=title, use_container_width=True)
    else:
        st.image(img_rgb, use_container_width=True)

def detect_and_classify(img_bgr, yolo_model, mobilenet_model):
    """Run detection with YOLO and classification with MobileNetV2."""
    results = yolo_model.predict(img_bgr)[0]
    class_labels = ['Peer Bottle', 'Plastic Bottle', 'Soda Bottle', 'Water Bottle', 'Wine Bottle']

    img_with_boxes = img_bgr.copy()

    # For each detected box
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        
        # Crop object
        cropped = img_bgr[y1:y2, x1:x2]
        
        # Preprocess for MobileNetV2
        resized = cv2.resize(cropped, (224, 224))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0)
        
        # Predict class
        preds = mobilenet_model.predict(input_img)
        pred_idx = np.argmax(preds)
        label = class_labels[pred_idx]

        # Draw bounding box and label
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} ({conf:.2f})"
        cv2.putText(img_with_boxes, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_with_boxes, len(results.boxes)

# ---------- Streamlit UI ----------

st.title("🍾 Bottle Detection & Classification App")

st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input method:", ("Upload Image", "Use Webcam"))

uploaded_file = None
camera_image = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (jpg, png, jpeg)", type=["jpg","jpeg","png"])
elif input_mode == "Use Webcam":
    camera_image = st.camera_input("Take a picture")

# Process input image
input_img_bgr = None

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert('RGB')
    input_img_bgr = preprocess_image(img_pil)
elif camera_image is not None:
    img_pil = Image.open(camera_image).convert('RGB')
    input_img_bgr = preprocess_image(img_pil)

if input_img_bgr is not None:
    st.subheader("Input Image")
    show_image(input_img_bgr)

    with st.spinner("Detecting and classifying bottles..."):
        result_img, count = detect_and_classify(input_img_bgr, yolo_model, mobilenet_model)

    st.subheader(f"Detection & Classification Results — {count} object(s) detected")
    show_image(result_img)

else:
    st.info("Please upload an image or take a picture using the webcam to start detection.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Your Name - Powered by Streamlit, Ultralytics YOLOv8, and TensorFlow")
