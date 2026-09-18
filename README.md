# Rice Leaf Disease Classification

A deep-learning application for rice leaf disease classification.

## Model

- DenseNet121 transfer learning
- ImageNet pretrained weights
- Random Zoom augmentation
- Fine-tuning of the final 40 DenseNet121 layers
- Input size: 224 x 224 x 3
- Six rice leaf classes
- Clean test accuracy: 95.45%

## Classes

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy
4. Leaf Blast
5. Leaf Scald
6. Narrow Brown Spot

## Run locally

pip install -r requirements.txt

streamlit run app.py

## Deployment

This application can be deployed using Streamlit Community Cloud.