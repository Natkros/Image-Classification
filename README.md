# 🐾 PetVision: Cat & Dog Image Classifier

[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.27-5856D6?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Author](https://img.shields.io/badge/Author-Kritarth%20Saxena-blue)](https://github.com/Natkros)

A Computer Vision final capstone project comparing a **Custom 4-Stage Deep CNN** trained from scratch against **MobileNetV2 Transfer Learning** for classifying real-world images of cats and dogs. Includes a fully executed Jupyter notebook, evaluation charts, a formal project report, and an interactive Streamlit web app.

---

## 🔗 Live Demo & Links
- **Public Live Web App URL:** [https://early-weeks-grab.loca.lt](https://early-weeks-grab.loca.lt) *(Tunnel IP: `49.36.136.166`)*
- **GitHub Repository:** [https://github.com/Natkros/Image-Classification](https://github.com/Natkros/Image-Classification)
- **Local Web App:** Run locally via `python -m streamlit run app.py` at `http://localhost:8501`.

---

## 📌 Project Overview
- **Goal:** Classify images into `Cat` or `Dog` with high accuracy and low inference latency.
- **Architectures Evaluated:**
  1. **Custom 4-Stage CNN:** Built from scratch using double-convolution blocks, Batch Normalization, spatial Dropout, and a dense MLP head (3.34M parameters).
  2. **MobileNetV2 Transfer Learning:** Pre-trained on ImageNet with a custom classification head fine-tuned with AdamW and Cosine Annealing (2.55M parameters).
- **Core Deliverables:**
  - 📓 **Jupyter Notebook:** `image_classification_project.ipynb`
  - 📄 **Project Report (PDF):** `Project_Report.pdf`
  - 🚀 **Streamlit Web Demo:** `app.py`
  - 📊 **Visualizations:** `visualizations/` (loss curves, confusion matrix, ROC curves, predictions)

---

## 📊 Benchmark Results

| Model Architecture | Test Accuracy | F1-Score | Parameters | CPU Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Custom Deep CNN** | **100.0%** | **100.0%** | 3.34M | 9.00 ms |
| **MobileNetV2 (Transfer)** | **100.0%** | **100.0%** | **2.55M** | **3.88 ms (2.3× faster)** |

---

## 📁 Project Structure

```
├── app.py                             # Streamlit interactive web application
├── model.py                           # CustomCNN and MobileNetV2 architectures
├── dataset.py                         # Data loading, augmentations & dataset pipeline
├── train.py                           # Training loop with LR scheduling & checkpointing
├── evaluate.py                        # Model evaluation & plot generation
├── generate_report.py                 # ReportLab PDF report generation script
├── image_classification_project.ipynb # Executed student Jupyter Notebook
├── Project_Report.pdf                 # Formal 2-3 page project report (PDF)
├── Project_Report.md                  # Markdown version of project report
├── requirements.txt                   # Project dependencies
├── submission_guide.txt               # Email submission template & checklist
├── saved_models/                      # Checkpoints (.pth) & training history (.json)
│   ├── best_custom_cnn.pth
│   └── best_mobilenet_v2.pth
├── visualizations/                    # Evaluation charts & confusion matrices
└── sample_images/                     # Test sample images for demo
```

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/Natkros/Image-Classification.git
cd Image-Classifier
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to test live image classification.

### 3. Re-train Models (Optional)
```bash
python train.py --model all --epochs 8
```

### 4. Run Evaluation & Generate Plots
```bash
python evaluate.py
```

### 5. Build PDF Report
```bash
python generate_report.py
```

---

## 👤 Author
**Kritarth Saxena**  
GitHub: [@Natkros](https://github.com/Natkros)
