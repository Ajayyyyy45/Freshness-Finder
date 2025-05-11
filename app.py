import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("fruit_classifier_model.h5")

# Dynamically generate class labels (e.g., "Class 0", "Class 1", ...)
output_shape = model.output_shape[-1]
class_labels = [f"Class {i}" for i in range(output_shape)]

# Manually define which class indices are fresh and which are rotten
# ⚠️ Update these lists based on your dataset's class structure
fresh_indices = list(range(0, 63))     # Example: Classes 0–62 = Fresh
rotten_indices = list(range(63, 125))  # Example: Classes 63–124 = Rotten

# Image preprocessing
def preprocess_image(image):
    image = image.resize((100, 100))            # Resize to match training input
    image = image.convert("RGB")                # Ensure 3 channels
    image = np.array(image) / 255.0             # Normalize pixel values
    image = np.expand_dims(image, axis=0)       # Add batch dimension
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

    return predicted_index, predicted_label, confidence

# Determine freshness
def get_freshness_status(index):
    if index in rotten_indices:
        return "Rotten ❌", "Not safe to eat"
    elif index in fresh_indices:
        return "Fresh ✅", "Good to eat for 2-3 days"
    else:
        return "Unknown", "Unknown shelf life"

# Streamlit UI
st.set_page_config(page_title="Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit image to see if it's **fresh or rotten**.")

# Option to save uploaded images
save_image = st.checkbox("💾 Save uploaded image")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Save uploaded image
    if save_image:
        os.makedirs("saved_uploads", exist_ok=True)
        image.save(f"saved_uploads/{uploaded_file.name}")
        st.info(f"✅ Image saved to `saved_uploads/{uploaded_file.name}`")

    if st.button("🔍 Classify"):
        try:
            index, label, confidence = predict_image_with_probs(image)
            freshness, recommendation = get_freshness_status(index)

            st.success(f"**🧠 Prediction:** {label}")
            st.info(f"**🍽️ Freshness Status:** {freshness}")
            st.write(f"**📊 Confidence:** {confidence:.2%}")
            st.write(f"**🕒 Recommendation:** {recommendation}")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")





       
