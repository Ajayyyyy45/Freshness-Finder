import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Load model
try:
    model = load_model("fruit_classifier_model.h5")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Load class labels
label_file = "class_labels.txt"
if not os.path.exists(label_file):
    st.error("🚫 'class_labels.txt' not found. Please place it in the same folder as this app.")
    st.stop()

with open(label_file, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Preprocess the image
def preprocess_image(image):
    image = image.resize((100, 100))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict image class and probabilities
def predict_image_with_probs(image):
    processed = preprocess_image(image)
    probs = model.predict(processed)[0]
    predicted_index = np.argmax(probs)

    if predicted_index >= len(class_labels):
        raise ValueError(f"Predicted index {predicted_index} out of range of class labels.")

    return class_labels[predicted_index], probs[predicted_index], probs

# Determine freshness
def get_freshness_status(label):
    label_lower = label.lower()
    if "rotten" in label_lower:
        return "Rotten ❌", "Not safe to eat"
    elif "fresh" in label_lower:
        return "Fresh ✅", "Good to eat for 2-3 days"
    else:
        return "Unknown", "Unknown shelf life"

# Get top K predictions
def get_top_k_predictions(probs, k=3):
    top_indices = np.argsort(probs)[::-1][:k]
    return [(class_labels[i], probs[i]) for i in top_indices]

# Streamlit UI
st.set_page_config(page_title="Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit image to check if it's **fresh or rotten**!")

save_image = st.checkbox("💾 Save uploaded image")
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if save_image:
        os.makedirs("saved_uploads", exist_ok=True)
        image_path = os.path.join("saved_uploads", uploaded_file.name)
        image.save(image_path)
        st.info(f"✅ Image saved to `{image_path}`")

    if st.button("🔍 Classify"):
        with st.spinner("Analyzing..."):
            try:
                label, confidence, probs = predict_image_with_probs(image)
                freshness, recommendation = get_freshness_status(label)

                st.success






       
