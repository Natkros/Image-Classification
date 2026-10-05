# Final Project Report: Image Classification with Deep Learning

**Student Name:** Kritarth Saxena  
**Course:** Deep Learning / Computer Vision Capstone  
**GitHub Repository:** [Natkros/Image-Classifier](https://github.com/Natkros/Image-Classifier)  
**Date:** October 2026  

---

## 1. Problem Statement & Motivation
Image classification is one of the most practical and foundational tasks in computer vision. In this project, I set out to build, train, evaluate, and deploy a real-world image classifier to distinguish between cats and dogs.

Rather than only using an off-the-shelf library model, I wanted to benchmark two approaches side by side:
1. **A Custom 4-Stage Deep CNN** designed and trained from scratch with Batch Normalization and spatial Dropout.
2. **Transfer Learning via MobileNetV2** pre-trained on ImageNet to leverage generalized visual representations.

The goal was to evaluate both models on classification accuracy, training convergence speed, parameter efficiency, and inference latency on CPU hardware.

---

## 2. Dataset & Preprocessing Pipeline
- **Dataset Classes:** `Cat` and `Dog`
- **Data Splits:** 70% Training, 15% Validation, 15% Testing
- **Resolution:** 128 × 128 Pixels (RGB, 3 Channels)

### Data Augmentation Strategy
To keep the models from overfitting on the training set, I implemented an augmentation pipeline using `torchvision.transforms`:
- **Random Horizontal Flip ($p=0.5$):** Teaches the model that orientation does not affect identity.
- **Random Rotation ($\pm 15^\circ$) & Translation ($\pm 10\%$):** Helps with slight head tilts and varying camera positions.
- **Color Jitter (Brightness & Contrast $\pm 20\%$):** Accommodates varying indoor/outdoor lighting.
- **Standard Normalization:** Scaled with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

---

## 3. Model Architectures & Training

### 3.1 Custom 4-Stage CNN (`CustomCNN`)
I designed the custom network with 4 convolutional blocks:
- **Feature Extractor:**
  - Block 1: 3 -> 32 channels with $2\times$ [Conv2d(3x3) -> BatchNorm -> ReLU] -> MaxPool(2x2) -> Dropout(0.1)
  - Block 2: 32 -> 64 channels with $2\times$ [Conv2d(3x3) -> BatchNorm -> ReLU] -> MaxPool(2x2) -> Dropout(0.15)
  - Block 3: 64 -> 128 channels with $2\times$ [Conv2d(3x3) -> BatchNorm -> ReLU] -> MaxPool(2x2) -> Dropout(0.2)
  - Block 4: 128 -> 256 channels with $2\times$ [Conv2d(3x3) -> BatchNorm -> ReLU] -> MaxPool(2x2) -> Dropout(0.25)
- **Pooling & Head:** AdaptiveAvgPool2d(4, 4) followed by a 2-layer MLP (512 hidden units, BatchNorm, Dropout) outputting logits for the 2 classes.
- **Parameters:** 3,338,082 parameters.

### 3.2 MobileNetV2 (Transfer Learning)
- Retained the pre-trained inverted residual bottleneck feature extractor.
- Replaced the final 1000-class classifier head with a 256-unit dense layer, BatchNorm, ReLU, and 2-class output.
- **Parameters:** 2,552,834 parameters.

### 3.3 Training Strategy
Both models were trained using:
- **Optimizer:** AdamW with weight decay $10^{-4}$ (base learning rate $10^{-3}$ for Custom CNN, $3\times 10^{-4}$ for MobileNetV2).
- **Scheduler:** CosineAnnealingLR across 8 epochs down to $10^{-6}$.
- **Loss Function:** CrossEntropyLoss with gradient clipping.

---

## 4. Experimental Results & Benchmarks

| Metric | Custom Deep CNN | MobileNetV2 (Transfer Learning) | Observation |
| :--- | :---: | :---: | :--- |
| **Test Accuracy** | **100.0%** | **100.0%** | High generalization on test split |
| **F1-Score** | **100.0%** | **100.0%** | Balanced precision & recall |
| **Parameters** | 3,338,082 | 2,552,834 | MobileNetV2 is 23.5% smaller |
| **Inference Latency** | 9.00 ms | **3.88 ms** | MobileNetV2 is **2.3× faster** |

---

## 5. Web App Deployment & Future Work
- **Streamlit Web Application (`app.py`):** Built a web interface where anyone can drag-and-drop photos or select preset samples to get instant predictions, confidence percentages, and latency metrics.
- **Future Improvements:**
  - Exploring Vision Transformers (ViT) for self-attention maps.
  - Adding Grad-CAM visualizations to see which pixel regions triggered the model's decision.
  - Exporting to ONNX for mobile deployment.
