import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Rice Leaf Disease Classification",
    page_icon="🌾",
    layout="centered"
)

# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Spot"
]

# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
MODEL_FILENAME = "final_rice_disease_densenet121_random_zoom.keras"

MODEL_URL = (
    "https://github.com/Sowmiya-6466/"
    "rice-disease-densenet121/releases/download/"
    "v1.0.0/"
    "final_rice_disease_densenet121_random_zoom.keras"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


# ---------------------------------------------------------
# Download model if it is not available locally
# ---------------------------------------------------------
def ensure_model_exists():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=600
            )
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    ensure_model_exists()
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🌾 Rice Leaf Disease Classification")
st.write(
    "Upload a rice leaf image to classify it into one of six disease categories."
)

# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded rice leaf image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease", type="primary"):

        # -------------------------------------------------
        # Preprocessing
        # -------------------------------------------------
        resized_image = image.resize((224, 224))

        image_array = np.array(resized_image).astype(np.float32)

        image_array = np.expand_dims(image_array, axis=0)

        # DenseNet121 preprocessing
        image_array = tf.keras.applications.densenet.preprocess_input(
            image_array
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------
        model = load_model()

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        # -------------------------------------------------
        # Display prediction
        # -------------------------------------------------
        st.success(f"Predicted Disease: {predicted_class}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # -------------------------------------------------
        # All class probabilities
        # -------------------------------------------------
        st.subheader("Prediction Probabilities")

        probability_dict = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(probability_dict)

        # -------------------------------------------------
        # Top 3 predictions
        # -------------------------------------------------
        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(predictions)[::-1][:3]

        for rank, index in enumerate(top_indices, start=1):
            st.write(
                f"**{rank}. {CLASS_NAMES[index]}** — "
                f"{predictions[index] * 100:.2f}%"
            )


# ---------------------------------------------------------
# About the model
# ---------------------------------------------------------
with st.expander("About this model"):

    st.write("""
    **Architecture:** DenseNet121

    **Transfer learning:** ImageNet pretrained weights

    **Enhanced augmentation:** Random Zoom

    **Fine-tuning:** Last 40 DenseNet121 layers

    **Batch Normalization:** Frozen during fine-tuning

    **Input size:** 224 × 224 × 3

    **Number of classes:** 6

    **Clean test accuracy:** 95.45%
    """)
