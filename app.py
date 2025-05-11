import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow\.keras.models import load\_model

# Load trained model

model = load\_model("fruit\_classifier\_model.h5")                                                                                              

# Automatically generate class labels based on model output shape

output\_shape = model.output\_shape\[-1]
class\_labels = \[f"Class {i}" for i in range(output\_shape)]

# Image preprocessing

def preprocess\_image(image):
image = image.resize((100, 100))            # Resize to match training input
image = image.convert("RGB")                # Ensure 3 channels
image = np.array(image) / 255.0             # Normalize pixel values to \[0, 1]
image = np.expand\_dims(image, axis=0)       # Add batch dimension (1, 100, 100, 3)
return image

# Prediction function

def predict\_image\_with\_probs(image):
processed\_image = preprocess\_image(image)
probs = model.predict(processed\_image)\[0]
predicted\_index = np.argmax(probs)

```
if predicted_index >= len(class_labels):
    raise ValueError(f"Predicted index {predicted_index} out of range.")

predicted_label = class_labels[predicted_index]
confidence = probs[predicted_index]

return predicted_label, confidence
```

# Streamlit App UI

st.set\_page\_config(page\_title="Freshness Finder", layout="centered")
st.title("🍓 Freshness Finder")
st.write("Upload a fruit image and find out if it's fresh or rotten.")

# Option to save uploaded images

save\_image = st.checkbox("💾 Save uploaded/captured image")

# File upload

uploaded\_file = st.file\_uploader("📤 Upload an image", type=\["jpg", "jpeg", "png"])

if uploaded\_file is not None:
image = Image.open(uploaded\_file)
st.image(image, caption="🖼️ Uploaded Image", use\_column\_width=True)

```
# Save image if selected
if save_image:
    os.makedirs("saved_uploads", exist_ok=True)
    image.save(f"saved_uploads/{uploaded_file.name}")
    st.info(f"✅ Image saved as `saved_uploads/{uploaded_file.name}`")

if st.button("🔍 Classify"):
    try:
        label, confidence = predict_image_with_probs(image)

        st.success(f"**🧠 Prediction:** {label}")
        st.write(f"**📊 Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
```

    



