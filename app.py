import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from PIL import Image

# Load model
model = load_model('fruit_classifier_model.h5')

# Define class labels
class_labels = ['Apple', 'Banana', 'Orange']  # Replace with your actual labels

# Prediction function
def predict_image(img):
    img = img_to_array(img)
    img = resize(img, (100, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_class_index]
    confidence = np.max(prediction) * 100
    return predicted_label, confidence

# Streamlit UI
st.title("Fruit Classifier")

uploaded_file = st.file_uploader("Upload an image of a fruit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify"):
        label, conf = predict_image(image)
        st.success(f"Prediction: **{label}** ({conf:.2f}% confidence)")
