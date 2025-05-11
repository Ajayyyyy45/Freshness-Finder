import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("fruit_classifier_model.h5")

# Correct class labels based on your notebook
class_labels = ['freshapples', 'freshbanana', 'freshoranges', 
                'rottenapples', 'rottenbanana', 'rottenoranges']

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Replace with your model's input size
    image = np.array(image) / 255.0   # Normalize if model trained with normalized data
    if image.shape[-1] == 4:
        image = image[..., :3]  # Convert RGBA to RGB if needed
    return image

# Prediction function
def predict_image_with_probs(image):
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    probs = model.predict(processed_image)[0]
    predicted_index = np.argmax(probs)

    if predicted_index >= len(class_labels):
        raise ValueError(f"Predicted index {predicted_index} is out of range for class_labels of length {len(class_labels)}")

    predicted_label = class_labels[predicted_index]
    confidence = probs[predicted_index]

    return predicted_label, confidence, probs

# Streamlit interface
st.title("🍎 Freshness Finder: Fruit/Veggie Classifier")
st.write("Upload an image of a fruit to find out if it's fresh or rotten!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        try:
            label, confidence, probs = predict_image_with_probs(image)
            st.success(f"**Prediction:** {label} \n\n**Confidence:** {confidence:.2%}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
