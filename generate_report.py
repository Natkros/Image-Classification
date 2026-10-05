"""Builds Project_Report.pdf from the real numbers in saved_models/evaluation_summary.json."""
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from model import MODELS

BLUE = colors.HexColor("#1e3c72")
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, textColor=BLUE, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=4)


def build(path: str = "Project_Report.pdf"):
    res = json.load(open("saved_models/evaluation_summary.json"))
    cnn, mob = res["custom_cnn"], res["mobilenet_v2"]
    rows = [["Model", "Accuracy", "F1", "Latency", "Parameters", "Train time"]] + [
        [MODELS[n], f"{r['accuracy']:.2f}%", f"{r['f1_score']:.2f}%", f"{r['latency_ms']:.1f} ms", f"{r['params']:,}", f"{r['train_seconds']} s"]
        for n, r in res.items()]
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                               ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#94a3b8")), ("PADDING", (0, 0), (-1, -1), 5)]))
    fig = lambda f, h: Image(f"visualizations/{f}.png", width=6.8 * inch, height=h * inch)

    story = [
        Paragraph("PetVision: Cat vs Dog Image Classification", ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, textColor=BLUE, spaceAfter=2)),
        Paragraph("Final capstone report · Kritarth Saxena · github.com/Natkros/Image-Classification", BODY),
        Paragraph("1. Problem", H1),
        Paragraph("Classify photos as cat or dog, and find out whether a CNN designed from scratch can match a fine-tuned pretrained network "
                  "once accuracy, inference speed and model size are all taken into account.", BODY),
        Paragraph("2. Data", H1),
        Paragraph("The cat and dog classes of CIFAR-10 (real photographs): 9,000 images for training, 1,000 for validation and the 2,000 official test images, "
                  "which were never used for tuning. Images are upscaled to 96×96 and augmented with flips, small rotations and shifts and colour jitter.", BODY),
        Paragraph("3. Models", H1),
        Paragraph(f"<b>Custom CNN</b> ({cnn['params']:,} parameters): four double-convolution blocks (32→256 channels) with BatchNorm, max-pooling and spatial dropout, "
                  f"then a small dense head. <b>MobileNetV2</b> ({mob['params']:,} parameters): ImageNet weights with a new 2-class head, fine-tuned end to end. "
                  "Both train with AdamW and cosine learning-rate decay, keeping the checkpoint with the best validation accuracy.", BODY),
        Paragraph("4. Results", H1), table, Spacer(1, 6), fig("model_comparison", 1.7), fig("training_curves", 2.5),
        Paragraph("5. Interpretation", H1),
        Paragraph(f"MobileNetV2 reaches {mob['accuracy']:.1f}% against {cnn['accuracy']:.1f}% for the scratch CNN, trains in {mob['train_seconds']} s versus {cnn['train_seconds']} s, "
                  f"and classifies an image in {mob['latency_ms']:.1f} ms versus {cnn['latency_ms']:.1f} ms. Pretrained features matter most when data is limited. "
                  "Grad-CAM maps (below) show both networks focusing on the animal's head and body rather than the background. "
                  "The remaining errors are mostly look-alikes at 32×32 resolution.", BODY),
        fig("gradcam_examples", 2.9), fig("confusion_matrix", 2.9),
        Paragraph("6. Deployment & future work", H1),
        Paragraph("The Streamlit app (<font face='Courier'>app.py</font>) classifies uploads with both models side by side and overlays Grad-CAM. "
                  "Next steps: train on full-resolution photos (e.g. Oxford-IIIT Pets), export to ONNX, and try a Vision Transformer.", BODY),
    ]
    SimpleDocTemplate(path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=48, bottomMargin=48).build(story)
    print(f"Report written to {path}")


if __name__ == "__main__":
    build()
