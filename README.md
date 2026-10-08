# 🌾 Transfer Learning-Based Rice Leaf Disease Classification Using Enhanced Data Augmentation

A deep learning project that classifies rice leaf images into six categories using **DenseNet121 transfer learning**. The project also investigates data augmentation and fine-tuning to study classification performance and robustness under image-quality variations.

## 🚀 Live Demo

**Try the application:** [Rice Leaf Disease Classifier](https://rice-disease-densenet121-j6yfsqsjdytxesgs2hk24h.streamlit.app/)

Upload a rice leaf image to receive a predicted class, confidence score, and class-wise prediction probabilities.

## 🎯 Project Objectives

- Classify rice leaf images into six categories.
- Use ImageNet-pretrained DenseNet121 through transfer learning.
- Evaluate Random Zoom augmentation and fine-tuning.
- Investigate robustness under Gaussian blur, brightness, and contrast changes.
- Develop a web application for image-based predictions.

## 🦠 Disease Categories

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy
4. Leaf Blast
5. Leaf Scald
6. Narrow Brown Spot

## 🧠 Methodology

1. Prepare and split the image dataset into training, validation, and test sets.
2. Resize images to 224 × 224 pixels and apply DenseNet121 preprocessing.
3. Train a DenseNet121-based classifier using transfer learning.
4. Compare augmentation strategies using validation performance.
5. Fine-tune the last 40 DenseNet121 layers while keeping Batch Normalization layers frozen.
6. Evaluate the selected model on the clean test set and under controlled image-quality changes.
7. Deploy the model using Streamlit Community Cloud.

## 📊 Final Model Results

| Metric | Result |
|---|---:|
| Architecture | DenseNet121 |
| Selected augmentation | Random Zoom |
| Fine-tuning | Last 40 layers |
| Clean test accuracy | 95.45% |
| Test images | 264 |
| Number of classes | 6 |

The final model correctly classified 252 out of 264 clean test images. Robustness results are evaluated separately and should not be confused with clean test accuracy.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- DenseNet121
- NumPy
- Pillow
- Streamlit
- Google Colab
- Git and GitHub

## 💻 Run Locally

Clone the repository:

git clone https://github.com/Sowmiya-6466/rice-disease-densenet121.git
cd rice-disease-densenet121

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

The application downloads the trained model from the GitHub Release when it is needed.

## 📁 Repository Structure

rice-disease-densenet121/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

The trained model is provided separately through the GitHub Release.

## ⚠️ Limitations

- Performance depends on the training dataset and the image supplied.
- Predictions on unseen field images may differ from test-set performance.
- Prediction confidence is not a guarantee that the classification is correct.
- Additional evaluation on rice leaf images from diverse real-world environments is needed.

## 👩‍💻 Author

Sowmiya

