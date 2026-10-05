"""PetVision: upload a photo, see both models' verdicts and where they looked (Grad-CAM)."""
import glob
import json
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from dataset import CLASSES, denormalize, get_transforms
from gradcam import explain, overlay
from model import DEVICE, MODELS, load_trained

EMOJI = {"cat": "🐱", "dog": "🐶"}
st.set_page_config(page_title="PetVision", page_icon="🐾", layout="wide")
st.markdown("""
<style>
.hero {background: linear-gradient(120deg,#1e3c72,#7b2ff7 70%,#f107a3); padding: 1.6rem 2rem;
       border-radius: 16px; color: #fff; margin-bottom: 1.2rem}
.hero h1 {margin: 0; font-size: 2.2rem} .hero p {margin: .3rem 0 0; opacity: .9}
.card {border: 1px solid rgba(128,128,128,.3); border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .6rem}
.verdict {font-size: 1.9rem; font-weight: 700; margin: 0} .muted {opacity: .65; font-size: .85rem}
</style>
<div class="hero"><h1>🐾 PetVision</h1>
<p>Cat or dog? Two neural networks vote, and Grad-CAM shows what they were looking at.</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return {name: load_trained(name) for name in MODELS}


@st.cache_data
def get_summary():
    with open("saved_models/evaluation_summary.json") as f:
        return json.load(f)


def classify(model, image: Image.Image):
    x = get_transforms()[1](image.convert("RGB"))[None].to(DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        model(x)
    ms = (time.perf_counter() - start) * 1000
    probs, cam = explain(model, x)
    return probs, overlay(denormalize(x[0]), cam), ms


tab_demo, tab_bench = st.tabs(["🔍 Try it", "📊 How good is it?"])

with tab_demo:
    samples = {f"{EMOJI[Path(p).stem[:3]]} {Path(p).stem[-1]}": p for p in sorted(glob.glob("sample_images/*.png"))}
    left, right = st.columns([2, 3], gap="large")
    with left:
        upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        pick = st.radio("…or try a sample", list(samples), horizontal=True, disabled=bool(upload)) if samples else None
        image = Image.open(upload) if upload else Image.open(samples[pick]) if pick else None
        if image:
            st.image(image, use_container_width=True)
        st.caption("Trained on small 32×32 CIFAR-10 photos, so close-ups of a single pet work best.")

    with right:
        if image is None:
            st.info("Upload a photo to get started.")
        else:
            results = {n: classify(m, image) for n, m in get_models().items()}
            labels = {n: CLASSES[p.argmax()] for n, (p, _, _) in results.items()}
            if len(set(labels.values())) == 1:
                st.success(f"Both models agree: it's a {labels['mobilenet_v2']} {EMOJI[labels['mobilenet_v2']]}")
            else:
                st.warning("The models disagree. This photo is a tricky one.")
            for col, (name, (probs, heat, ms)) in zip(st.columns(2), results.items()):
                with col:
                    st.markdown(f"""<div class="card"><span class="muted">{MODELS[name]}</span>
                        <p class="verdict">{EMOJI[labels[name]]} {labels[name].title()}</p>
                        <span class="muted">{probs.max():.1%} confident · {ms:.1f} ms on {DEVICE.type.upper()}</span></div>""",
                                unsafe_allow_html=True)
                    st.image(heat, caption="Grad-CAM: red = most influential", use_container_width=True)
                    for c, p in zip(CLASSES, probs):
                        st.progress(float(p), text=f"{c.title()} {p:.1%}")

with tab_bench:
    summary = get_summary()
    cols = st.columns(len(summary))
    for col, (name, r) in zip(cols, summary.items()):
        col.metric(MODELS[name], f"{r['accuracy']:.1f}% accuracy", f"{r['latency_ms']:.1f} ms · {r['params'] / 1e6:.2f}M params", delta_color="off")
    st.caption("Scored on the 2,000 official CIFAR-10 test photos (cats vs dogs), which were never used for training or tuning.")
    for column, files in zip(st.columns(2), [("model_comparison", "training_curves", "gradcam_examples"), ("confusion_matrix", "roc_curves")]):
        for f in files:
            column.image(f"visualizations/{f}.png", use_container_width=True)
