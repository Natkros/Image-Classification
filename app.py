#!/usr/bin/env python3
"""Gradio demo: drop in an image, get the predicted scene class.

    python app.py
    python app.py --checkpoint outputs/<run>/checkpoints/best.pt --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import gradio as gr  # noqa: E402
import numpy as np   # noqa: E402
from PIL import Image  # noqa: E402

from ic.inference import Predictor  # noqa: E402

DESCRIPTION = """
Upload a photo and the model sorts it into one of six scene categories.
It was fine-tuned from ImageNet weights on the Intel Image Classification dataset,
so it does best on ordinary outdoor photographs.
"""


def latest_checkpoint(outputs_dir: Path) -> Path:
    candidates = sorted(outputs_dir.glob("*/checkpoints/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No checkpoints found under {outputs_dir}.\n"
            "Train one first:  python scripts/train.py"
        )
    return candidates[-1]


def build_interface(predictor: Predictor, show_gradcam: bool = True) -> gr.Blocks:
    def classify(image: Image.Image | None, want_gradcam: bool):
        if image is None:
            return {}, None, ""
        probabilities = predictor.predict_proba(image)
        labels = {name: float(p) for name, p in zip(predictor.classes, probabilities)}

        overlay = None
        if want_gradcam:
            from ic.gradcam import overlay_heatmap

            rgb, cam = predictor.gradcam(image)
            overlay = (overlay_heatmap(rgb, cam) * 255).astype(np.uint8)

        top_index = int(np.argmax(probabilities))
        margin = float(np.sort(probabilities)[-1] - np.sort(probabilities)[-2])
        note = (
            f"**{predictor.classes[top_index]}** at {probabilities[top_index]:.1%} confidence"
            + ("  \n(close call — the runner-up is within "
               f"{margin:.1%})" if margin < 0.15 else "")
        )
        return labels, overlay, note

    with gr.Blocks(title="Scene classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Scene classifier")
        gr.Markdown(DESCRIPTION.strip())

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Image", height=320)
                gradcam_toggle = gr.Checkbox(
                    value=show_gradcam, label="Show what the model looked at (Grad-CAM)"
                )
                submit = gr.Button("Classify", variant="primary")
                clear = gr.Button("Clear")
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=len(predictor.classes), label="Prediction")
                note_output = gr.Markdown()
                cam_output = gr.Image(label="Grad-CAM", height=280)

        sample_dir = REPO_ROOT / "assets" / "samples"
        samples = sorted(str(p) for p in sample_dir.glob("*")) if sample_dir.exists() else []
        if samples:
            gr.Examples(examples=samples, inputs=image_input, label="Try one of these")

        inputs = [image_input, gradcam_toggle]
        outputs = [label_output, cam_output, note_output]
        submit.click(classify, inputs, outputs)
        image_input.change(classify, inputs, outputs)
        clear.click(lambda: (None, {}, None, ""), None, [image_input, *outputs])

        meta = f"Model: {predictor.config.get('model', {}).get('name', 'unknown')}"
        if predictor.train_metrics.get("acc"):
            meta += f" · validation accuracy {predictor.train_metrics['acc']:.1%}"
        gr.Markdown(f"<sub>{meta}</sub>")

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="defaults to the most recent run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--no-gradcam", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(REPO_ROOT / "outputs")
    print(f"Loading {checkpoint}…")
    predictor = Predictor(checkpoint, device=args.device)

    demo = build_interface(predictor, show_gradcam=not args.no_gradcam)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
