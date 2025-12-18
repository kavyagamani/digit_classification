# =====================================================
# STREAMLIT APP – DIGIT CLASSIFICATION (0–9)
# =====================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Digit Classification",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Classification")
st.write("Upload a handwritten digit image (0–9)")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/digit_cnn_model.h5")

model = load_model()

# -------------------------
# IMAGE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    # -------------------------
    # PREPROCESS IMAGE
    # -------------------------
    img = image.convert("L")       # grayscale
    img = img.resize((28, 28))     # resize
    img_array = np.array(img)
    img_array = img_array / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28, 1)

    # -------------------------
    # PREDICT
    # -------------------------
    if st.button("Predict Digit"):
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        st.success(f"✅ Predicted Digit: {digit}")

        # Optional: show confidence scores
        st.subheader("Prediction Confidence")
        for i, prob in enumerate(prediction[0]):
            st.write(f"Digit {i}: {prob:.4f}")
