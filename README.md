# Waste Classification Using EfficientNetV2B2

This project uses a Convolutional Neural Network based on EfficientNetV2B2 to classify different types of waste images into predefined categories. It includes model training, evaluation, and deployment via a Gradio web interface.

---

## Features

- Image preprocessing and augmentation  
- Class imbalance handling using class_weight  
- Transfer learning using EfficientNetV2B2 (pretrained on ImageNet)  
- Interactive prediction via Gradio  
- Model evaluation: Confusion matrix, classification report, accuracy/loss graphs

---

## Installation

bash
pip install tensorflow matplotlib seaborn scikit-learn gradio


---

## Dataset Structure

The dataset should be organized in the following directory structure:


your_dataset_directory/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── ...
└── ...


Each subfolder represents a different class.

---

## Model Architecture

Base Model: EfficientNetV2B2 (include_top=False)  
Additional Layers:
- Data augmentation  
- GlobalAveragePooling2D  
- Dropout  
- Dense layer with Softmax

---

## Evaluation Metrics

- Accuracy and loss (training vs. validation)  
- Classification report (precision, recall, F1-score)  
- Confusion matrix  
- Per-class distribution graphs

---

## Web App with Gradio

After training, launch the Gradio interface to try the model:

python
iface.launch()


This opens a browser interface where you can upload an image and see the predicted waste category.

---

## Model Saving

The trained model is saved in Keras format:

python
model.save('Effiicientnetv2b2.keras')


You can reload it using:

python
model = tf.keras.models.load_model('Effiicientnetv2b2.keras')


---

##Troubleshooting

ImportError (e.g., tensorflow.keras.applications not found):  
Ensure you're using a compatible TensorFlow version (>=2.11). You can reinstall with:

bash
pip install --upgrade tensorflow


"End of sequence" warnings:  
These are expected when a dataset has been fully iterated during evaluation. You can safely ignore them.

---
