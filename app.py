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
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.image import resize

    # Convert and preprocess the image
    img = img_to_array(img)
    img = resize(img, (100, 100))  # Resize to match model input
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)

    # Debugging information
    print("Model prediction output:", prediction)
    print("Predicted class index:", predicted_class_index)
    print("Number of class labels:", len(class_labels))

    # Safety check
    if predicted_class_index >= len(class_labels):
        return "Unknown", 0.0

    predicted_label = class_labels[predicted_class_index]
    confidence = np.max(prediction) * 100
    return predicted_label, confidence
