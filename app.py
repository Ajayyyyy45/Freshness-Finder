import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("fruit_classifier_model.h5")

# Define class labels (customize as per your model training)
class_labels = ["Ripe Apple", "Rotten Apple", "Ripe Banana", "Rotten Banana", "Ripe Orange", "Rotten Orange"]

# Define ripeness duration (in days) for ripe fruits (customize as needed)
ripeness_duration = {
    "Ripe Apple": 5,
    "Ripe Banana": 2,
    "Ripe Orange": 7
}

# Image preprocessing
def preprocess_image(image):
    image = image.resize((100, 100))            # Resize to match model input
    image = image.convert("RGB")                # Ensure 3 channels
    image = np.array(image) / 255.0             # Normalize
    image = np.expand_dims(image, axis=0)       # Shape: (1, 100, 100, 3)
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

    return predicted_label, confidence, probs

# Determine ripeness info
def get_ripeness_info(predicted_label):
    if "Rotten" in predicted_label:
        status = "Rotten ❌"
        duration_info = "It should be discarded."
    elif "Ripe" in predicted_label:
        days = ripeness_duration.get(predicted_label, "unknown")
        status = "Ripe ✅"
        duration_info = f"Best consumed within **{days} day(s)**." if isinstance(days, int) else "Ripeness duration unknown."
    else:
        status = "Unknown"
        duration_info = "Could not determine ripeness."
    return status, duration_info

# Streamlit UI
st.set_page_config(page_title="Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit or vegetable image to determine if it's **ripe or rotten**, and how long it will stay ripe.")

save_image = st.checkbox("💾 Save uploaded/captured image")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

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
            status, duration_info = get_ripeness_info(label)

            st.success(f"**🧠 Prediction:** {label}")
            st.info(f"**🍽️ Ripeness Status:** {status}")
            st.write(f"**📊 Confidence:** {confidence:.2%}")
            st.warning(duration_info)

            # Show bar chart of all class probabilities
            fig, ax = plt.subplots()
            ax.barh(class_labels, probs, color='skyblue')
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probability")
            ax.set_title("Class Probabilities")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")


          
