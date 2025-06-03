# 🔍 Violence Detection in Videos Using Deep Learning

This project is a deep learning-based system that analyzes video clips and detects the **presence of violent content**. It does this by sampling frames from the video and using a trained ResNet152V2-based model to make predictions.

---

## 🎯 Project Goal

The goal is to build an **automated content moderation tool** that can:
- Help platforms flag potentially violent video clips
- Support content reviewers by pre-screening videos
- Enhance safety features in apps involving user-generated content

---

## 🧠 How It Works

1. **Frame Sampling**:  
   Instead of analyzing every single frame (which is computationally expensive), the script **randomly selects 20 frames** from different time points in the video.

2. **Image Preprocessing**:  
   Each frame is resized to a standard shape (320x240) and prepared for input using `preprocess_input()`.

3. **Prediction**:  
   Each frame is passed through a trained **Convolutional Neural Network (CNN)**, and the results are averaged to determine the final violence score for the video.

---

## 🤖 Model Architecture

The model uses **ResNet152V2** as a base feature extractor, without pre-trained weights. It’s followed by:
- **Global Average Pooling** (to reduce dimensions)
- A **Dense Layer** with sigmoid activation for binary classification (violent or non-violent)

```python
base_model = keras.applications.ResNet152V2(
    include_top=False,
    weights=None,
    classifier_activation="softmax",
    name="resnet152v2",
)

inputs = keras.layers.Input(shape=(240, 320, 3))
x = base_model(inputs)
y = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(y)
model = keras.Model(inputs, outputs)
