"""Builds image_classification_project.ipynb and executes it, so every output is a real result."""
import nbformat as nbf
from nbclient import NotebookClient

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# 🐾 PetVision: Cat vs Dog Classification\n"
       "**Kritarth Saxena** · [GitHub](https://github.com/Natkros/Image-Classification)\n\n"
       "Can a CNN built from scratch keep up with a fine-tuned MobileNetV2? Both are trained on the cat and dog "
       "photos of CIFAR-10 and compared on accuracy, speed and size. Every output below comes from a real run of this notebook."),
    md("## 1. Data\nCIFAR-10's cat and dog classes: 10,000 training photos (10% held out for validation) and 2,000 official test photos, upscaled to 96×96."),
    code("import matplotlib.pyplot as plt, pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from dataset import CLASSES, denormalize, get_dataloaders\n"
         "train_dl, val_dl, test_dl, _ = get_dataloaders()\n"
         "print('train / val / test images:', *(len(d.dataset) for d in (train_dl, val_dl, test_dl)))\n"
         "fig, axes = plt.subplots(2, 8, figsize=(14, 4))\n"
         "for ax, i in zip(axes.flat, range(0, 16 * 120, 120)):\n"
         "    x, y = test_dl.dataset[i]; ax.imshow(denormalize(x)); ax.set_title(CLASSES[y]); ax.axis('off')\n"
         "plt.show()"),
    md("## 2. Models\n* **Custom CNN**: four double-conv blocks (32→256 channels) with BatchNorm and spatial dropout, trained from scratch.\n"
       "* **MobileNetV2**: ImageNet-pretrained backbone with a new 2-class head, fine-tuned end to end.\n\n"
       "Both use AdamW with cosine learning-rate decay (`python train.py`) and keep the checkpoint with the best *validation* accuracy."),
    code("from model import MODELS, count_params, get_model\n"
         "for name, label in MODELS.items(): print(f'{label:12s} {count_params(get_model(name, pretrained=False)):,} parameters')\n"
         "display(Image('visualizations/training_curves.png'))"),
    md("## 3. Results on the held-out test set"),
    code("from evaluate import score\n"
         "results = {name: score(name, test_dl) for name in MODELS}\n"
         "pd.DataFrame(results).T.rename(index=MODELS)[['accuracy', 'precision', 'recall', 'f1_score', 'latency_ms', 'params', 'train_seconds']]"),
    code("display(Image('visualizations/model_comparison.png'))\ndisplay(Image('visualizations/confusion_matrix.png'))"),
    md("## 4. What is the model looking at?\nGrad-CAM heat maps show the regions that drove each prediction. Red marks the most influential pixels."),
    code("display(Image('visualizations/gradcam_examples.png'))"),
    md("## 5. Takeaways\n"
       "* Transfer learning wins on quality: about 4 points more accuracy with fewer parameters, and training took roughly a quarter of the time.\n"
       "* The trade-off is speed: on a CPU, MobileNetV2\'s many small depthwise layers make it slower per image than the compact scratch CNN.\n"
       "* Remaining errors are mostly cat/dog look-alikes at 32×32 resolution, where even a human would hesitate.\n\n"
       "Try it live: `streamlit run app.py`."),
]

if __name__ == "__main__":
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    NotebookClient(nb, timeout=900, kernel_name="python3").execute()
    nbf.write(nb, "image_classification_project.ipynb")
    print("Notebook executed and saved.")
