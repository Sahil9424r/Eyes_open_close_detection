import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("Eyes.keras")
labels = ["Closed", "Open"]
img_size = (256, 256)

st.set_page_config(page_title="Driver Drowsiness Detection")
st.title("🛑 Driver Drowsiness Detection System")
st.write("Check if the eyes are open or closed using an image or webcam.")

# Image preprocessing
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(img_size)
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# Predict function
def predict(image):
    pred = model.predict(image, verbose=0)[0][0]
    label = labels[1] if pred > 0.5 else labels[0]
    return label, pred

# Add checkbox to choose mode
st.markdown("## Choose Input Method")
use_webcam = st.checkbox("📷 Use Webcam")

if use_webcam:
    st.markdown("### Webcam Mode")
    camera = st.camera_input("Take a picture using webcam")
    if camera:
        img = Image.open(camera)
        st.image(img, caption="Captured Image", use_column_width=True)
        image = preprocess(img)
        label, confidence = predict(image)
        st.success(f"Prediction: **{label}** ({confidence:.2f})")
else:
    st.markdown("### Upload Image Mode")
    uploaded_file = st.file_uploader("📁 Upload an eye image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        image = preprocess(img)
        label, confidence = predict(image)
        st.success(f"Prediction: **{label}** ({confidence:.2f})")
