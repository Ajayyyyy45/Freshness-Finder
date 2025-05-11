import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Load authentication config
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Setup authenticator (compatible with streamlit-authenticator >= 0.3.0)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login UI (new API returns 3 values)
name, auth_status, username = authenticator.login(location='sidebar')

if auth_status:
    authenticator.logout("Logout", location="sidebar")
    st.sidebar.success(f"Logged in as {name}")

    # Set up the Streamlit app
    st.set_page_config(page_title="Freshness Finder", layout="centered")
    st.title("🍓 Freshness Finder")
    st.write("Upload a fruit image to check if it's **fresh or rotten**!")

    # Load model
    model_file = "fruit_classifier_model.h5"
    if not os.path.exists(model_file):
        st.error(f"🚫 '{model_file}' not found. Please place it in the app folder.")
        st.stop()
    try:
        model = load_model(model_file)
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

    # Helper functions
    def preprocess_image(image):
        image = image.resize((100, 100))
        image = image.convert("RGB")
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict_image_with_probs(image):
        processed = preprocess_image(image)
        probs = model.predict(processed)[0]
        predicted_index = np.argmax(probs)

        if predicted_index >= len(class_labels):
            raise ValueError(f"Predicted index {predicted_index} out of range of class labels.")

        return class_labels[predicted_index], probs[predicted_index], probs

    def get_freshness_status(label):
        label_lower = label.lower()
        if "rotten" in label_lower:
            return "Rotten ❌", "Not safe to eat"
        elif "fresh" in label_lower:
            return "Fresh ✅", "Good to eat for 2-3 days"
        else:
            return "Unknown", "Unknown shelf life"

    def get_top_k_predictions(probs, k=3):
        top_indices = np.argsort(probs)[::-1][:k]
        return [(class_labels[i] if i < len(class_labels) else f"Class {i}", probs[i]) for i in top_indices]

    # Image upload UI
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

                    st.success(f"**🧠 Prediction:** {label}")
                    st.info(f"**🍽️ Freshness Status:** {freshness}")
                    st.write(f"**📊 Confidence:** {confidence:.2%}")
                    st.write(f"**🕒 Recommendation:** {recommendation}")

                    st.subheader("🔝 Top 3 Predictions:")
                    for lbl, prob in get_top_k_predictions(probs):
                        st.write(f"- {lbl}: {prob:.2%}")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    st.markdown("---")
    st.caption("📦 Built with TensorFlow, Streamlit, and 💚")

elif auth_status is False:
    st.error("❌ Invalid username or password")
elif auth_status is None:
    st.warning("🔑 Please enter your username and password")
