import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("fruit_classifier_model.h5")

# Detect model output shape
output_shape = model.output_shape[-1]

# Define all possible class labels
all_possible_labels = ["Fresh", "Approaching Expiry", "Rotten"]

# ✅ Match class_labels length with model's output shape
class_labels = all_possible_labels[:output_shape]

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((100, 100))            # Resize to model input shape
    image = image.convert("RGB")                # Ensure RGB channels
    image = np.array(image) / 255.0             # Normalize pixel values
    image = np.expand_dims(image, axis=0)       # Add batch dimension
    return image

# Prediction function
def predict_image_with_probs(image):
    processed_image = preprocess_image(image)
    probs = model.predict(processed_image)[0]
    predicted_index = np.argmax(probs)

    # Check index safety
    if predicted_index >= len(class_labels):
        raise ValueError(f"Predicted index {predicted_index} is out of range for class_labels list.")

    predicted_label = class_labels[predicted_index]
    confidence = probs[predicted_index]
    return predicted_label, confidence

# Streamlit UI setup
st.set_page_config(page_title="Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit image to check if it's **Fresh**, **Approaching Expiry**, or **Rotten**.")

# Optionally save uploaded images
save_image = st.checkbox("💾 Save uploaded image")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Save image if selected
    if save_image:
        os.makedirs("saved_uploads", exist_ok=True)
        image.save(f"saved_uploads/{uploaded_file.name}")
        st.info(f"✅ Image saved to `saved_uploads/{uploaded_file.name}`")

    if st.button("🔍 Classify"):
        try:
            label, confidence = predict_image_with_probs(image)
            st.success(f"**🧠 Prediction:** {label}")
            st.write(f"**📊 Confidence:** {confidence:.2%}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")




