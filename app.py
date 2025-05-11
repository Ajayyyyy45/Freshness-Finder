import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("fruit_classifier_model.h5")

# Define class labels (update this with your actual classes)
class_labels = ["Fresh Apple", "Rotten Apple", "Fresh Banana", "Rotten Banana", "Fresh Orange", "Rotten Orange"]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((100, 100))            # Resize to match training input
    image = image.convert("RGB")                # Ensure 3 channels
    image = np.array(image) / 255.0             # Normalize pixel values
    image = np.expand_dims(image, axis=0)       # Add batch dimension (1, 100, 100, 3)
    return image

# Prediction function
def predict_image_with_probs(image):
    processed_image = preprocess_image(image)
    probs = model.predict(processed_image)[0]
    predicted_index = np.argmax(probs)

    if predicted_index >= len(class_labels):
        raise ValueError(f"Predicted index {predicted_index} out of range.")

    predicted_label = class_labels[predicted_index]
    confidence = probs[predicted_index]

    return predicted_label, confidence

# Ripeness status checker
def get_freshness_status(label):
    if "Rotten" in label:
        return "Rotten ❌"
    elif "Fresh" in label:
        return "Fresh ✅"
    else:
        return "Unknown"

# Streamlit App UI
st.set_page_config(page_title="Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit image and find out if it's **fresh or rotten**.")

# Option to save uploaded images
save_image = st.checkbox("💾 Save uploaded/captured image")

# File upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Save image if selected
    if save_image:
        os.makedirs("saved_uploads", exist_ok=True)
        image.save(f"saved_uploads/{uploaded_file.name}")
        st.info(f"✅ Image saved as `saved_uploads/{uploaded_file.name}`")

    if st.button("🔍 Classify"):
        try:
            label, confidence = predict_image_with_probs(image)
            status = get_freshness_status(label)

            st.success(f"**🧠 Prediction:** {label}")
            st.info(f"**🍽️ Freshness Status:** {status}")
            st.write(f"**📊 Confidence:** {confidence:.2%}")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

