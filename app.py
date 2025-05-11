import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------- Setup -------------------- #

# Load your trained model
model = load_model('fruit_classifier_model.h5')

# Define class labels
class_labels = ['Apple', 'Banana', 'Orange']  # Modify based on your model

# Create upload directory if saving is enabled
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------- Helper Functions -------------------- #

def predict_image_with_probs(img):
    img = img.resize((100, 100))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    return predicted_label, confidence, prediction

def get_freshness_level(confidence):
    if confidence > 85:
        return "🌿 Fresh"
    elif confidence > 60:
        return "🍌 Ripening"
    else:
        return "🥀 Spoiled"

def save_image(image, filename):
    image.save(os.path.join(UPLOAD_DIR, filename))

# -------------------- Streamlit App -------------------- #

st.title("🍓 Fruit Freshness Classifier")

# Sidebar options
st.sidebar.header("Settings")
save_option = st.sidebar.checkbox("Save uploaded/captured image")

# File uploader
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

# Camera input
camera_image = st.camera_input("Or capture a fruit image using your camera")

# Determine the image source
image = None
filename = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    filename = uploaded_file.name
    st.image(image, caption="Uploaded Image", use_column_width=True)
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    filename = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    st.image(image, caption="Captured Image", use_column_width=True)

# Perform classification
if image is not None and st.button("🔍 Classify"):
    with st.spinner("Analyzing image..."):
        label, confidence, all_probs = predict_image_with_probs(image)
        freshness = get_freshness_level(confidence)

        # Save if enabled
        if save_option and filename:
            save_image(image, filename)
            st.sidebar.success(f"Image saved to '{UPLOAD_DIR}/{filename}'")

    # Display results
    st.markdown(f"### 🏷️ Predicted: **{label}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")
    st.markdown(f"### 🧪 Freshness Level: **{freshness}**")

    # Bar chart
    st.subheader("🔢 Class Confidence")
    prob_df = pd.DataFrame({
        "Fruit": class_labels,
        "Confidence (%)": all_probs * 100
    })
    st.bar_chart(prob_df.set_index("Fruit"))
