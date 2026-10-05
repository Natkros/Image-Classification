"""
PetVision - Cat & Dog Image Classifier
Final Project by Kritarth Saxena (@Natkros)
Built with PyTorch & Streamlit
"""
import os
import time
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

from dataset import get_transforms
from model import count_parameters, get_model

st.set_page_config(page_title="PetVision - Image Classifier", page_icon="🐾", layout="wide")

st.markdown("""
<style>
    .header-box { background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.2rem; }
    .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 1.2rem; }
    .badge { display: inline-block; padding: 0.35rem 1.1rem; border-radius: 20px; font-weight: bold; font-size: 1.2rem; color: white; }
    .cat-badge { background: #f59e0b; } .dog-badge { background: #3b82f6; }
</style>
<div class="header-box">
    <h2 style="margin:0;">🐾 PetVision: Cat & Dog Classifier</h2>
    <p style="margin:0.3rem 0 0; opacity:0.9;">Deep Learning Final Project | Developed by Kritarth Saxena</p>
</div>
""", unsafe_allow_html=True)

CLASSES = ["cat", "dog"]


@st.cache_resource
def load_model(model_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_id, num_classes=len(CLASSES), pretrained=False)
    ckpt_path = f"saved_models/best_{model_id}.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    return model.to(device).eval(), device


# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Project Controls")
    model_choice = st.radio("Choose Model Architecture:", ["MobileNetV2 (Transfer Learning)", "Custom CNN (From Scratch)"])
    model_id = "mobilenet_v2" if "MobileNet" in model_choice else "custom_cnn"
    model, device = load_model(model_id)
    params = count_parameters(model)
    st.markdown(f"**Parameters:** `{params['total_params']:,}`")
    st.markdown(f"**Hardware Device:** `{device.type.upper()}`")
    st.markdown("---")
    st.markdown("**Student Developer:** Kritarth Saxena")
    st.markdown("**GitHub:** [@Natkros](https://github.com/Natkros)")

tab_demo, tab_benchmarks = st.tabs(["🚀 Live Prediction", "📊 Evaluation & Benchmarks"])

with tab_demo:
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.subheader("1. Select or Upload an Image")
        st.write("Quick test samples:")
        sample_cols = st.columns(4)
        sample_img = None
        sample_files = sorted([f for f in os.listdir("sample_images") if f.endswith(('.png', '.jpg'))]) if os.path.exists("sample_images") else []
        for i, f in enumerate(sample_files[:4]):
            if sample_cols[i].button(f"{'🐱' if 'cat' in f else '🐶'} Sample {i+1}", key=f"s_{i}", use_container_width=True):
                sample_img = os.path.join("sample_images", f)

        uploaded = st.file_uploader("Upload your own photo (JPG / PNG):", type=["jpg", "jpeg", "png"])
        active_image = Image.open(uploaded) if uploaded else (Image.open(sample_img) if sample_img else (Image.open(os.path.join("sample_images", sample_files[0])) if sample_files else None))

        if active_image:
            st.image(active_image, caption="Current Input", use_container_width=True)

    with col_right:
        st.subheader("2. Model Classification")
        if active_image:
            _, val_transform = get_transforms(128)
            tensor = val_transform(active_image.convert("RGB")).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            lat_ms = (time.perf_counter() - t0) * 1000

            pred_idx = int(probs.argmax())
            pred_class = CLASSES[pred_idx]
            conf = probs[pred_idx] * 100
            badge_cls = "cat-badge" if pred_class == "cat" else "dog-badge"

            st.markdown(f"""
            <div class="card">
                <p style="margin:0; color:#888;">PREDICTED CLASS</p>
                <div style="margin:0.5rem 0;">
                    <span class="badge {badge_cls}">{'🐱' if pred_class == 'cat' else '🐶'} {pred_class.upper()}</span>
                </div>
                <h2 style="color:#10b981; margin:0.3rem 0;">{conf:.1f}% Confidence</h2>
                <small style="color:#aaa;">Inference latency: {lat_ms:.2f} ms on {device.type.upper()}</small>
            </div>
            """, unsafe_allow_html=True)

            st.write("#### Confidence Breakdown")
            for c, p in zip(CLASSES, probs):
                st.write(f"**{c.title()}**: {p * 100:.1f}%")
                st.progress(float(p))

with tab_benchmarks:
    st.subheader("Model Performance & Training Curves")
    c1, c2 = st.columns(2)
    for p, col in [("visualizations/training_curves.png", c1), ("visualizations/confusion_matrix.png", c2),
                  ("visualizations/roc_pr_curves.png", c1), ("visualizations/model_comparison_bar.png", c2)]:
        if os.path.exists(p):
            col.image(p, use_container_width=True)
