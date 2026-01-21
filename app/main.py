import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# 1. STREAMLIT CONFIG
st.set_page_config(page_title="AI Plant Health Diagnostic", page_icon="🌿", layout="wide")

# 2. PATH SETUP
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "Final_Plant_Classifier.keras")
indices_path = os.path.join(working_dir, "class_indices.json")

# 3. LOAD MODEL & LABELS (With Auto-Fix Logic)
@st.cache_resource
def load_trained_model():
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_indices():
    if not os.path.exists(indices_path):
        return None
    with open(indices_path, "r") as f:
        data = json.load(f)
        # Handle inverted JSON {Name: Index} -> {Index: Name}
        first_key = list(data.keys())[0]
        if not first_key.isdigit():
            return {str(v): k for k, v in data.items()}
        return data

model = load_trained_model()
class_indices = load_class_indices()

# 4. PREPROCESSING
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. UI INTERFACE
st.title("🌿 AI Plant Health Diagnostic System")

# Safety Check
if model is None or class_indices is None:
    st.error("### ⚠️ System Files Missing")
    st.info(f"Ensure 'Final_Plant_Classifier.keras' is in 'trained_model' folder and 'class_indices.json' is in 'app' folder.")
    st.stop()

uploaded_image = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Sample", use_container_width=True)

    with col2:
        if st.button("Analyze Health Status"):
            with st.spinner("Analyzing patterns..."):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)

                idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]) * 100)
                prediction_name = class_indices.get(str(idx), "Unknown")

                st.subheader("Diagnostic Results")
                if "healthy" in prediction_name.lower():
                    st.success(f"**Status:** {prediction_name}")
                else:
                    st.warning(f"**Detected:** {prediction_name}")

                st.metric("Confidence Level", f"{confidence:.2f}%")

st.divider()
st.caption("Developed by Animesh Kumar ")