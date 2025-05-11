import streamlit as st
import numpy as np
from PIL import Image
import os
import uuid
from tensorflow.keras.models import load_model

# Load the trained model
try:
    model = load_model("fruit_classifier_model.h5")
except Exception as e:
    st.error(f"🚫 Failed to load model: {e}")
    st.stop()

# Auto-generate class_labels.txt if missing
def generate_dummy_labels(filename, num_classes):
    with open(filename, "w") as f:
        for i in range(num_classes):
            f.write(f"class_{i}\n")
    st.warning(f"⚠️ 'class_labels.txt' not found. A placeholder with {num_classes} classes was generated.")

# Load or generate class labels
label_file = "class_labels.txt"
if not os.path.exists(label_file):
    # Try to detect number of classes from model output shape
    try:
        num_classes = model.output_shape[-1]
        generate_dummy_labels(label_file, num_classes)
    except Exception as e:
        st.error(f"🚫 Couldn't determine number of classes: {e}")
        st.stop()

# Read class labels
with open(label_file, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((100, 100))            # Resize to match model input
    image = image.convert("RGB")                # Ensure 3 channels
    image = np.array(image) / 255.0             # Normalize pixel values
    image = np.expand_dims(image, axis=0)       # Add batch dimension
    return image

# Predict and get probabilities
def predict_image_with_probs(image):
    processed_image = preprocess_image(image)
    probs = model.predict(processed_image)[0]
    predicted_index = np.argmax(probs)

    if predicted_index >= len(class_labels):
        raise ValueError(f"Predicted index {predicted_index} out of range. Check your class_labels.txt file.")

    predicted_label = class_labels[predicted_index]
    confidence = probs[predicted_index]

    return predicted_label, confidence, probs

# Determine freshness
def get_freshness_status(label):
    if "rotten" in label.lower():
        return "Rotten ❌", "Not safe to eat"
    elif "fresh" in label.lower():
        return "Fresh ✅", "Good to eat for 2-3 days"
    else:
        return "Unknown", "Unknown shelf life"

# Top K predictions
def get_top_k_predictions(probs, k=3):
    top_indices = np.argsort(probs)[::-1][:k]
    return [(class_labels[i], probs[i]) for i in top_indices]

# Streamlit UI
st.set_page_config(page_title="Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit image to find out if it's **fresh or rotten**.")

# Option to save uploaded image
save_image = st.checkbox("💾 Save uploaded image")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if save_image:
        os.makedirs("saved_uploads", exist_ok=True)
        filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        image.save(f"saved_uploads/{filename}")
        st.info(f"✅ Image saved to `saved_uploads/{filename}`")

    if st.button("🔍 Classify"):
        with st.spinner("🔍 Classifying..."):
            try:
                label, confidence, probs = predict_image_with_probs(image)
                freshness, recommendation = get_freshness_status(label)

                st.success(f"**🧠 Prediction:** {label}")
                st.info(f"**🍽️ Freshness Status:** {freshness}")
                st.write(f"**📊 Confidence:** {confidence:.2%}")
                st.write(f"**🕒 Recommendation:** {recommendation}")

                st.subheader("🔝 Top 3 Predictions:")
                top_predictions = get_top_k_predictions(probs)
                for lbl, prob in top_predictions:
                    st.write(f"- {lbl}: {prob:.2%}")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")






       
