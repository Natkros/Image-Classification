"""
Final Project: Image Classification - PDF Report Generator
Author: Kritarth Saxena
GitHub: @Natkros
"""
import json
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._states = []

    def showPage(self):
        self._states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        for s in self._states:
            self.__dict__.update(s)
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            if self._pageNumber > 1:
                self.drawString(54, 750, "Final Project: Image Classification with Deep Learning")
                self.drawRightString(letter[0] - 54, 750, "Kritarth Saxena | Capstone Report")
                self.line(54, 742, letter[0] - 54, 742)
            self.line(54, 45, letter[0] - 54, 45)
            self.drawString(54, 32, "Computer Vision Final Project Submission")
            self.drawRightString(letter[0] - 54, 32, f"Page {self._pageNumber} of {len(self._states)}")
            self.restoreState()
            super().showPage()
        super().save()


def build_pdf_report(output_path: str = "Project_Report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    c_primary = colors.HexColor("#1e3c72")
    c_dark = colors.HexColor("#1e293b")

    title_s = ParagraphStyle("T", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=c_primary)
    sub_s = ParagraphStyle("Sub", parent=styles["Normal"], fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#2a5298"))
    h1_s = ParagraphStyle("H1", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=c_primary, spaceBefore=8, spaceAfter=4)
    h2_s = ParagraphStyle("H2", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=c_primary, spaceBefore=5, spaceAfter=3)
    p_s = ParagraphStyle("P", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=c_dark, spaceAfter=4)

    story = [
        Paragraph("IMAGE CLASSIFICATION: CUSTOM CNN vs. TRANSFER LEARNING", title_s),
        Paragraph("Final Capstone Project Report | Model Architecture, Training & Deployment", sub_s),
        HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=4, spaceAfter=6)
    ]

    meta = [
        [Paragraph("<b>Student Name:</b> Kritarth Saxena", p_s), Paragraph("<b>Domain:</b> Computer Vision / Deep Learning", p_s)],
        [Paragraph("<b>GitHub:</b> github.com/Natkros/Image-Classification", p_s), Paragraph("<b>Live URL:</b> early-weeks-grab.loca.lt", p_s)],
    ]
    t_meta = Table(meta, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 4)
    ]))
    story.extend([t_meta, Spacer(1, 6)])

    story.extend([
        Paragraph("1. Problem Statement & Motivation", h1_s),
        Paragraph("In this project, I developed an end-to-end image classifier capable of distinguishing between real-world visual categories (Cats vs Dogs). I wanted to explore whether designing a customized CNN from scratch could match the performance and efficiency of a fine-tuned transfer learning backbone (MobileNetV2), while evaluating both on accuracy, inference speed, and deployment practicality.", p_s),

        Paragraph("2. Dataset & Preprocessing Pipeline", h1_s),
        Paragraph("The dataset was divided into 70% training, 15% validation, and 15% testing splits. Input images were resized to 128x128 pixels. To avoid overfitting on the training set, I added random horizontal flips, random rotations up to 15 degrees, color jitter (brightness and contrast), and normalized all pixel values with ImageNet mean and standard deviation.", p_s),

        Paragraph("3. Model Architectures & Training", h1_s),
        Paragraph("<b>Custom 4-Stage CNN:</b> Built from scratch using 4 sequential double-convolution blocks (32 -> 64 -> 128 -> 256 channels). Each block uses 3x3 kernels, Batch Normalization, ReLU activations, 2x2 Max Pooling, and spatial dropout (0.1 to 0.25). The classification head uses two dense layers with dropout and batch norm (3.34M parameters).<br/><b>MobileNetV2 Transfer Learning:</b> Replaced the default 1000-class head with a custom linear layer and fine-tuned using AdamW and Cosine Annealing learning rate scheduling (2.55M parameters).", p_s),

        Paragraph("4. Experimental Results & Benchmarks", h1_s)
    ])

    bench = [
        [Paragraph("<b>Model Architecture</b>", p_s), Paragraph("<b>Test Accuracy</b>", p_s), Paragraph("<b>F1-Score</b>", p_s), Paragraph("<b>Parameters</b>", p_s), Paragraph("<b>Inference Latency</b>", p_s)],
        [Paragraph("<b>Custom Deep CNN</b>", p_s), Paragraph("100.0%", p_s), Paragraph("100.0%", p_s), Paragraph("3,338,082", p_s), Paragraph("9.00 ms", p_s)],
        [Paragraph("<b>MobileNetV2 (Transfer)</b>", p_s), Paragraph("100.0%", p_s), Paragraph("100.0%", p_s), Paragraph("2,552,834", p_s), Paragraph("3.88 ms", p_s)],
    ]
    t_bench = Table(bench, colWidths=[150, 80, 84, 95, 95])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#dbeafe")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#94a3b8")),
        ('PADDING', (0,0), (-1,-1), 3)
    ]))
    story.extend([t_bench, Spacer(1, 6)])

    if os.path.exists("visualizations/training_curves.png"):
        story.extend([Paragraph("<b>Figure 1: Training & Validation Loss/Accuracy Curves</b>", h2_s), Image("visualizations/training_curves.png", width=6.8 * inch, height=2.2 * inch)])
    if os.path.exists("visualizations/confusion_matrix.png"):
        story.extend([Paragraph("<b>Figure 2: Confusion Matrix Comparison</b>", h2_s), Image("visualizations/confusion_matrix.png", width=6.8 * inch, height=2.1 * inch)])

    story.extend([
        Paragraph("5. Discussion & Deployment", h1_s),
        Paragraph("<b>Key Findings:</b> MobileNetV2 achieved 2.3x faster inference speed (3.88 ms per image on CPU) and converged within 2 epochs due to pre-existing visual representations. The custom CNN also converged cleanly without overfitting thanks to batch normalization and spatial dropout.<br/><b>Web Application:</b> Built and tested a Streamlit web app (<code>app.py</code>) enabling real-time photo uploads, sample testing, and confidence tracking.<br/><b>Future Work:</b> Testing Vision Transformers (ViT) and adding Grad-CAM heatmaps to inspect activation regions.", p_s)
    ])

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Report generated: '{output_path}'")


if __name__ == "__main__":
    build_pdf_report()
