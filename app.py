import streamlit as st
import numpy as np
from PIL import Image
import os
import json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model("fruit_classifier_model.h5")

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# CNN-compatible image preprocessing
def preprocess_image(image):
    image = image.resize((100, 100))  # CNN expects 100x100 input
    image = np.array(image) / 255.0

    # Handle RGBA to RGB
    if image.shape[-1] == 4:
        image = image[..., :3]

    if image.shape != (100, 100, 3):
        raise ValueError(f"Expected image shape (100, 100, 3), got {image.shape}")

    image = np.expand_dims(image, axis=0)  # Shape: (1, 100, 100, 3)
    return image

# Prediction function
def predict_image_with_probs(image):
    processed_image = preprocess_image(image)
    probs = model.predict(processed_image)[0]
    predicted_index = int(np.argmax(probs))

    if predicted_index >= len(class_labels):
        raise ValueError(f"Predicted index {predicted_index} out of range. "
                         f"Model returned {len(probs)} outputs, "
                         f"but only {len(class_labels)} labels found.")

    predicted_label = class_labels[predicted_index]
    confidence = probs[predicted_index]

    return predicted_label, confidence, probs

# Streamlit UI
st.set_page_config(page_title="🍎 Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit image to check if it's **fresh** or **rotten**!")

save_image = st.checkbox("💾 Save uploaded image")

uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if save_image:
        os.makedirs("saved_uploads", exist_ok=True)
        image.save(f"saved_uploads/{uploaded_file.name}")
        st.info(f"✅ Image saved as `saved_uploads/{uploaded_file.name}`")

    if st.button("🔍 Classify"):
        try:
            label, confidence, probs = predict_image_with_probs(image)

            st.success(f"**🧠 Prediction:** {label}")
            st.write(f"**📊 Confidence:** {confidence:.2%}")

            # Bar chart
            fig, ax = plt.subplots()
            ax.barh(class_labels, probs, color='skyblue')
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probability")
            ax.set_title("Class Probabilities")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

