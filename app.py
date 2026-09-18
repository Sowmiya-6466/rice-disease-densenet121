import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Rice Leaf Disease Classification",
    page_icon="🌾",
    layout="centered"
)


CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Spot"
]


MODEL_PATH = os.path.join(
    "models",
    "final_rice_disease_densenet121_random_zoom.keras"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


def predict_disease(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    ).astype(np.float32)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = tf.keras.applications.densenet.preprocess_input(
        image_array
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return predicted_class, confidence, predictions


st.title(
    "🌾 Rice Leaf Disease Classification"
)

st.write(
    "Upload a rice leaf image to classify it into "
    "one of six classes using DenseNet121 with "
    "Random Zoom augmentation and fine-tuning."
)


uploaded_file = st.file_uploader(
    "Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Rice Leaf",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict Disease",
        type="primary"
    ):

        with st.spinner(
            "Analyzing the rice leaf..."
        ):

            predicted_class, confidence, predictions = (
                predict_disease(image)
            )

        st.success(
            "Predicted Disease: " + predicted_class
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.subheader(
            "Prediction Confidence"
        )

        confidence_data = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(
            confidence_data
        )

        st.subheader(
            "Top 3 Predictions"
        )

        sorted_indices = np.argsort(
            predictions
        )[::-1][:3]

        for rank, index in enumerate(
            sorted_indices,
            start=1
        ):

            st.write(
                f"{rank}. {CLASS_NAMES[index]} — "
                f"{predictions[index] * 100:.2f}%"
            )


with st.expander("About the model"):

    st.write(
        "Architecture: DenseNet121"
    )

    st.write(
        "Transfer Learning: ImageNet pretrained weights"
    )

    st.write(
        "Augmentation: Random Zoom"
    )

    st.write(
        "Fine-Tuning: Last 40 DenseNet121 layers"
    )

    st.write(
        "Input Size: 224 × 224 × 3"
    )

    st.write(
        "Number of Classes: 6"
    )

    st.write(
        "Clean Test Accuracy: 95.45%"
    )