# 🔍 Violence Detection in Videos Using Deep Learning

This project is a deep learning-based system that detects **violent content in video clips**. It works by analyzing a small set of frames from a video and using a trained convolutional neural network to determine whether the content is violent.

---

## 🎯 Project Goal

The objective is to create an **automated content moderation tool** that can:
- Detect potentially violent content in videos
- Assist platforms in pre-screening user-generated content
- Enable safer browsing or viewing environments

---

## 🧠 How It Works

1. **Frame Sampling**  
   Instead of analyzing every frame, the system **randomly selects 20 frames** from across the video for faster and efficient analysis.

2. **Preprocessing**  
   Each frame is resized to 320×240 pixels and processed using `preprocess_input()` (matching the ResNet preprocessing pipeline).

3. **Model Prediction**  
   Each frame is passed through a **deep neural network**. The model outputs a value between 0 and 1:
   - Close to **1**: indicates violent content
   - Close to **0**: indicates non-violent content  
   The average score across all frames determines the overall prediction for the video.

---

## 🤖 Model Architecture

The system uses **transfer learning** with a **ResNet152V2** model as the backbone. This model is pretrained on **ImageNet**, a large image dataset, and is highly effective at extracting visual features.

After the ResNet feature extraction:
- A **Global Average Pooling** layer compresses the features.
- A final **Dense layer with sigmoid activation** predicts the probability of violence.

This streamlined architecture ensures the model is both **powerful and efficient**, making it suitable for real-time or batch video analysis.

---
