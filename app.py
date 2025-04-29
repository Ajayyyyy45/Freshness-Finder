import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.image import resize

# Load your trained model
model = load_model('fruit_classifier_model.h5')

# Define your class labels (example)
class_labels = ['Apple', 'Banana', 'Orange']  # <-- Replace with your real labels

def predict_image(img):
    img = img_to_array(img)
    img = resize(img, (100, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_class_index]
    confidence = np.max(prediction) * 100
    return f"{predicted_label} ({confidence:.2f}% confidence)"

# Gradio Interface
interface = gr.Interface(fn=predict_image, inputs="image", outputs="text")

interface.launch()
