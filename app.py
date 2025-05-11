import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# --- Load login configuration ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Authenticator setup ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Login UI ---
name, auth_status, username = authenticator.login("Login", "main")

if auth_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome, **{name}** 👋")

    # --- Load class labels from file ---
    with open("class_labels.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]

    # --- Load trained model ---
    model = load_model("fruit_classifier_model.h5")

    # --- Image preprocessing ---
    def preprocess_image(image):
        image = image.resize((100, 100))
        image = image.convert("RGB")
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    # --- Predict image ---
    def predict_image(image):
        processed = preprocess_image(image)
        probs = model.predict(processed)[0]
        index = np.argmax(probs)
        if index >= len(class_labels):
            raise ValueError(f"Index {index} out of range")
        return class_labels[index], probs[index]

    # --- Determine freshness ---
    def get_freshness_status(label):
        if "rotten" in label.lower():
            return "Rotten ❌", "Not safe to eat"
        elif "fresh" in label.lower():
            return "Fresh ✅", "Good to eat for 2-3 days"
        else:
            return "Unknown", "Unknown shelf life"

    # --- App UI ---
    st.title("🍓 Freshness Finder")
    st.write("Upload a fruit image to check if it's fresh or rotten.")

    save_image = st.checkbox("💾 Save uploaded image")
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        if save_image:
            os.makedirs("saved_uploads", exist_ok=True)
            image.save(f"saved_uploads/{uploaded_file.name}")
            st.success(f"Image saved as `saved_uploads/{uploaded_file.name}`")

        if st.button("🔍 Classify"):
            try:
                label, confidence = predict_image(image)
                status, recommendation = get_freshness_status(label)

                st.success(f"**Prediction:** {label}")
                st.info(f"**Freshness:** {status}")
                st.write(f"**Confidence:** {confidence:.2%}")
                st.write(f"**Recommendation:** {recommendation}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

elif auth_status is False:
    st.error("❌ Incorrect username or password")

elif auth_status is None:
    st.warning("👤 Please enter your username and password")








       
