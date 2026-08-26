# Image Classification

Scene classification in PyTorch: sorts natural photographs into six categories —
**buildings, forest, glacier, mountain, sea, street** — by fine-tuning an
ImageNet-pretrained convolutional network.

The repository is a complete, reproducible project rather than a single script: a
configurable training pipeline, honest evaluation with error analysis, a command-line
predictor, a browser demo, and a narrated notebook that walks the whole thing end to end.

**Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
— ~25,000 JPEGs at 150×150, split roughly 14k train / 3k test / 7k unlabelled.

---

## Table of contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Getting the data](#getting-the-data)
- [How the model works](#how-the-model-works)
- [Usage](#usage)
  - [Training](#training) · [Evaluation](#evaluation) · [Prediction](#prediction) · [Web demo](#web-demo) · [Notebook](#notebook)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [Python API](#python-api)
- [Interpreting results](#interpreting-results)
- [Design decisions](#design-decisions)
- [Extending the project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [Reproducibility](#reproducibility)

---

## Quick start

```bash
git clone <https://github.com/Natkros/Image-Classification.git> && cd intel-image-classifier
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py      
python scripts/train.py            
python app.py                        # browser demo at http://localhost:7860
```

**No Kaggle token yet?** Verify the whole pipeline on synthetic data in about a minute:

```cmd
make smoke
```

That generates a small procedural dataset, trains for two epochs, and runs evaluation.
The accuracy is meaningless — the point is that every moving part works before you spend
time on a 350 MB download.

---

## Installation

**Requirements:** Python 3.9+, and roughly 2 GB of disk for the dataset.

```cmd
pip install -r requirements.txt
```

| Group | Packages | Needed for |
|---|---|---|
| Core | `torch`, `torchvision`, `numpy`, `Pillow`, `PyYAML` | everything |
| Analysis | `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm` | metrics and figures |
| Demo | `gradio` | `app.py` |
| Notebook | `jupyter`, `ipywidgets` | `notebooks/walkthrough.ipynb` |
| Data | `kaggle` | `scripts/download_data.py` only |

**GPU.** The default `pip install torch` gives a CPU or CUDA build depending on your
platform. For a specific CUDA version, follow the selector at
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) *before* installing
the rest of `requirements.txt`.

**Apple Silicon.** Works out of the box — `device: auto` selects `mps`. Mixed precision is
CUDA-only and is skipped automatically elsewhere.

**Check what you got:**

```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Getting the data

### Option 1 — the download script (recommended)

1. Go to Kaggle → **Settings** → **API** → **Create New Token**. That downloads
   `kaggle.json`.
2. Move it to `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\` on Windows) and
   `chmod 600` it. Or export the credentials instead:
   ```cmd
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```
3. Run it:
   ```cmd
   python scripts/download_data.py
   ```

The script downloads, extracts, deletes the zip, then prints a per-class image count so
you can confirm the layout before training.

```
python scripts/download_data.py --dest data/intel   # where to put it
                                --force             # re-download even if the zip is there
                                --keep-archive      # don't delete the zip
                                --verify-only       # just check what's already on disk
```

### Option 2 — download by hand

Grab the zip from the dataset page and unzip it into `data/intel/`. Then
`python scripts/download_data.py --verify-only` to confirm the layout.

### Option 3 — synthetic data, no account needed

```cmd
python scripts/make_sample_data.py --dest data/sample --per-class 24
python scripts/train.py --set data.root=data/sample train.epochs=2 data.num_workers=0
```

### Expected layout

The loader walks down from `data.root` looking for the folder that actually holds class
subdirectories, so all of these work:

```
data/intel/seg_train/seg_train/<class>/*.jpg     # what the Kaggle zip gives you
data/intel/train/<class>/*.jpg                   # flattened
data/intel/<class>/*.jpg                         # no split folders at all
```

Recognised split names: `seg_train`/`train`/`training`, `seg_test`/`test`/`val`/`validation`,
`seg_pred`/`pred`/`unlabeled`.

**Splits.** Validation is carved out of the *training* folder with a stratified split
(`data.val_split`, default 15%) — every class keeps its proportion in both halves. The
official test folder is never touched until the final evaluation, so the headline number
stays honest.

---

## How the model works

### Why transfer learning

A network trained on ImageNet already knows edges, textures, and object parts, and those
features transfer to almost any natural-image task. What does *not* transfer is the
1000-way classifier on top, so it is discarded and replaced with a dropout + linear head
sized to six classes.

Training then runs in **two stages**, which matters more than it might look:

| Stage | Epochs | Backbone | Learning rate | Why |
|---|---|---|---|---|
| 1 | 1 – 2 | frozen | 1e-3 (head only) | A randomly initialised head produces large, noisy gradients. Letting them flow into good pretrained features damages them — the classic "catastrophic forgetting" failure |
| 2 | 3 → end | trainable | 1e-4 backbone / 1e-3 head | With a sane head in place, gently adapt the features to *these* six classes |

The switch happens at `model.unfreeze_at_epoch` (default 3). Set it to `null` to train the
head only — faster, lower ceiling.

### What else is in the loop

- **Warmup + cosine decay** — one epoch of linear warmup, then cosine annealing.
- **Discriminative learning rates** — the head and the backbone get separate parameter
  groups, rebuilt when the backbone unfreezes.
- **Label smoothing** (0.05) — softens targets, small but consistent gain.
- **Mixed precision** on CUDA, skipped automatically on CPU and MPS.
- **Gradient clipping** at norm 1.0.
- **Early stopping** on validation accuracy, patience 5.
- **Best-checkpoint saving** — weights, class list, and full config in one `.pt` file, so
  inference needs nothing else.

### Augmentation

Training views get random resized crop (scale 0.7–1.0), horizontal flip, ±12° rotation,
colour jitter, and random erasing. Validation and test get a plain resize + centre crop —
no randomness, so the numbers are reproducible.

Horizontal flip is safe here because a mirrored landscape is still the same scene. It
would be wrong for text or digits; keep that in mind if you point this at another dataset.

### Available backbones

`resnet18` · `resnet34` · `resnet50` · `efficientnet_b0` · `efficientnet_b1` ·
`mobilenet_v3_large` · `convnext_tiny`

Swapping requires no code changes:

```cmd
python scripts/train.py --set model.name=efficientnet_b0
```

---

## Usage

### Training

```bash
python scripts/train.py                                    # defaults
python scripts/train.py --set train.epochs=20 data.batch_size=64
python scripts/train.py --set model.name=resnet50 --run-name resnet50-baseline
python scripts/train.py --no-test                          # skip final test evaluation
```

| Flag | Default | Purpose |
|---|---|---|
| `--config PATH` | `configs/default.yaml` | Config file to load |
| `--set K=V [K=V …]` | — | Dotted overrides applied on top of the file |
| `--run-name NAME` | auto (`<model>_<timestamp>`) | Folder name under `outputs/` |
| `--no-test` | off | Train only; don't touch the test set |

Every run writes a self-contained folder:

```
outputs/<run_name>/
├── config.yaml                    exact config used — re-run reproduces this run
├── train.log                      per-epoch log
├── history.json                   loss / accuracy / lr / seconds per epoch
├── test_metrics.json              final metrics + confusion matrix
├── checkpoints/best.pt            weights + class names + config
└── figures/
    ├── class_distribution.png
    ├── training_history.png
    ├── confusion_matrix.png
    └── per_class_f1.png
```

### Evaluation

```cmd
python scripts/evaluate.py                                    # latest run, test split
python scripts/evaluate.py --checkpoint outputs/my-run/checkpoints/best.pt
python scripts/evaluate.py --split val --save-errors
```

| Flag | Default | Purpose |
|---|---|---|
| `--checkpoint PATH` | most recent `outputs/*/checkpoints/best.pt` | Which model to evaluate |
| `--config PATH` | the config stored in the checkpoint | Override data settings |
| `--set K=V` | — | Dotted overrides |
| `--split {test,val,train}` | `test` | Which split to score |
| `--save-errors` | off | Write `errors_<split>.csv` |
| `--out DIR` | beside the checkpoint | Where metrics and figures go |

Evaluation always runs on the **un-augmented, unshuffled** view of a split — including
`--split train`. That keeps the numbers reproducible and guarantees each row of
`errors_<split>.csv` pairs the right file path with the right prediction.

`errors_<split>.csv` columns: `path`, `true`, `predicted`, `confidence`. Sort it by
confidence descending and you get the most useful list in the project: images the model
got wrong *and* was certain about, which is where mislabeled data and learned shortcuts
surface.

### Prediction

```cmd
python scripts/predict.py photo.jpg                            # top-3 with bars
python scripts/predict.py photo.jpg --gradcam cam.png          # + attention overlay
python scripts/predict.py data/intel/seg_pred --csv out.csv    # a whole folder
python scripts/predict.py photos/ --top-k 1 --batch-size 64 --device cpu
```

| Flag | Default | Purpose |
|---|---|---|
| `target` | required | An image file **or** a folder (searched recursively) |
| `--checkpoint PATH` | most recent run | Model to use |
| `--top-k N` | 3 | How many classes to report |
| `--batch-size N` | 32 | Folder mode only |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `--csv PATH` | — | Write results (works for one image or many) |
| `--gradcam PATH` | — | Save an input-vs-attention figure (single image only) |

Single-image output:

```
photo.jpg
  1. mountain   87.31%  ████████████████████████
  2. glacier     9.12%  ███
  3. forest      2.04%  █
```

### Web demo

```cmd
python app.py                      # http://localhost:7860
python app.py --share              # temporary public link
python app.py --port 8080 --no-gradcam --checkpoint outputs/my-run/checkpoints/best.pt
```

Drop in an image and get ranked probabilities plus a Grad-CAM overlay of the pixels that
drove the decision. When the top two classes are within 15 percentage points the app says
so rather than presenting a coin-flip as a confident answer.

Drop a few JPEGs into `assets/samples/` and they appear as clickable examples.

### Notebook

```cmd
jupyter notebook notebooks/walkthrough.ipynb
```

Twenty-three cells, end to end: data exploration and class balance → why transfer learning
→ training → reading the curves → metrics and confusion matrix → the most confident
mistakes → Grad-CAM → where to go next. It imports the same modules the scripts use, so it
cannot drift out of sync with the code.

Set `DATA_ROOT`, `EPOCHS`, and `MODEL` in the second cell and run all.

---

## Configuration reference

Everything lives in `configs/default.yaml`. Any key can be overridden from the command
line without editing the file:

```cmd
python scripts/train.py --set train.lr=0.0003 augment.color_jitter=0 data.val_split=0.2
```

Values are parsed with YAML rules, so `3` becomes an int, `true` a bool, `null` a None.

### Top level

| Key | Default | Notes |
|---|---|---|
| `seed` | 42 | Seeds Python, NumPy and torch — controls the split and initialisation |
| `device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

### `data`

| Key | Default | Notes |
|---|---|---|
| `root` | `data/intel` | Where the dataset lives |
| `train_dir` / `test_dir` | `null` | Set explicitly to bypass auto-detection |
| `val_split` | 0.15 | Stratified fraction held out of the train folder. `0` disables it — see the warning below |
| `image_size` | 224 | 150 matches the source images and trains faster; 224 matches the pretrained weights and usually scores higher |
| `batch_size` | 32 | Halve it if you hit CUDA OOM |
| `num_workers` | 4 | Set to `0` on Windows or inside notebooks if you see worker crashes |
| `mean` / `std` | ImageNet stats | Leave alone while using pretrained weights |

> **`val_split: 0`** removes the validation set. Training still runs, but the best epoch is
> then chosen on *training* accuracy — never on the test set — and logged with a warning.
> Selecting checkpoints on test data would quietly inflate your headline number.

### `augment`

| Key | Default | Notes |
|---|---|---|
| `random_resized_crop` | true | Falls back to resize + centre crop when false |
| `crop_scale` | [0.7, 1.0] | Fraction of the image area kept |
| `horizontal_flip` | 0.5 | Probability; `0` disables |
| `rotation_degrees` | 12 | `0` disables |
| `color_jitter` | 0.2 | Brightness/contrast/saturation strength; `0` disables |
| `random_erasing` | 0.15 | Probability; applied after normalisation |

### `model`

| Key | Default | Notes |
|---|---|---|
| `name` | `resnet18` | See [available backbones](#available-backbones) |
| `pretrained` | true | `false` trains from scratch — expect a large accuracy drop |
| `dropout` | 0.2 | Applied in the new head |
| `freeze_backbone` | true | Stage 1 behaviour |
| `unfreeze_at_epoch` | 3 | `null` never unfreezes |

### `train`

| Key | Default | Notes |
|---|---|---|
| `epochs` | 12 | |
| `lr` | 1e-3 | Head learning rate |
| `backbone_lr` | 1e-4 | Applied once the backbone unfreezes |
| `weight_decay` | 1e-4 | |
| `label_smoothing` | 0.05 | |
| `optimizer` | `adamw` | `adamw` or `sgd` (SGD uses Nesterov momentum 0.9) |
| `scheduler` | `cosine` | `cosine`, `plateau`, or `none` |
| `warmup_epochs` | 1 | Linear warmup before cosine |
| `amp` | true | Mixed precision; ignored off CUDA |
| `grad_clip` | 1.0 | `0` disables |
| `early_stopping_patience` | 5 | `0` disables |
| `class_weights` | false | Inverse-frequency loss weights — turn on past ~3× imbalance |

### `output`

| Key | Default | Notes |
|---|---|---|
| `dir` | `outputs` | Root for run folders |
| `run_name` | `null` | Auto-generated from model + timestamp |
| `save_best_only` | true | `false` also saves every epoch |

---

## Project structure

```
intel-image-classifier/
├── configs/
│   └── default.yaml            every hyperparameter, one file
├── src/ic/
│   ├── config.py               YAML loading + dotted CLI overrides, attribute access
│   ├── data.py                 split discovery, transforms, stratified split, loaders
│   ├── model.py                backbone factory, freeze/unfreeze, parameter groups
│   ├── engine.py               train/eval loops, optimisers, schedulers, early stopping
│   ├── metrics.py              accuracy, balanced accuracy, macro F1, confusion pairs
│   ├── viz.py                  every chart in the project, one consistent style
│   ├── gradcam.py              attention heatmaps
│   ├── inference.py            Predictor — checkpoint in, predictions out
│   └── utils.py                seeding, device, logging, checkpoints, meters
├── scripts/
│   ├── download_data.py        Kaggle download + layout verification
│   ├── make_sample_data.py     synthetic dataset for offline smoke tests
│   ├── train.py                training entrypoint
│   ├── evaluate.py             metrics, confusion matrix, error CSV
│   └── predict.py              single image or folder inference
├── notebooks/
│   └── walkthrough.ipynb       narrated end-to-end walkthrough
├── app.py                      Gradio web demo
├── assets/samples/             optional example images for the demo
├── data/                       datasets (gitignored)
├── outputs/                    runs (gitignored)
├── Makefile                    shortcuts for the common commands
├── requirements.txt
└── pyproject.toml
```

`make help` lists the shortcuts: `install`, `data`, `sample`, `smoke`, `train`,
`evaluate`, `app`, `notebook`, `clean`.

---

## Python API

Everything the scripts do is importable. Add `src/` to your path (the scripts do this
automatically) and:

```python
import sys; sys.path.insert(0, "src")

from ic.config import load_config
from ic.data import build_datasets, build_dataloaders
from ic.model import build_model
from ic.engine import fit, evaluate
from ic.metrics import compute_metrics, format_summary
from ic.utils import resolve_device, set_seed, make_run_dir, get_logger

cfg = load_config("configs/default.yaml", ["train.epochs=5"])
set_seed(cfg.seed)
device = resolve_device(cfg.device)

datasets = build_datasets(cfg)
loaders, classes = build_dataloaders(cfg, datasets)

model = build_model(cfg.model.name, len(classes), freeze_backbone=True).to(device)
run_dir = make_run_dir("outputs", "api-run", cfg.model.name)
result = fit(model, loaders, cfg, device, classes, run_dir,
             get_logger("ic"), datasets["train_targets"])
```

Inference is a two-liner:

```python
from ic.inference import Predictor

predictor = Predictor("outputs/my-run/checkpoints/best.pt")
predictor.predict("photo.jpg", top_k=3)
# [('mountain', 0.873), ('glacier', 0.091), ('forest', 0.020)]

predictor.predict_proba("photo.jpg")          # raw probability vector
predictor.predict_batch(paths, batch_size=64) # list of dicts
rgb, cam = predictor.gradcam("photo.jpg")     # image + heatmap arrays
```

### Selected entry points

| Module | Key names |
|---|---|
| `ic.config` | `Config`, `load_config` |
| `ic.data` | `build_datasets`, `build_dataloaders`, `build_transforms`, `find_split_dir`, `stratified_indices`, `class_distribution`, `compute_class_weights`, `denormalize`, `UnlabeledImageDataset` |
| `ic.model` | `build_model`, `AVAILABLE_MODELS`, `set_backbone_frozen`, `param_groups`, `model_from_checkpoint`, `last_conv_layer` |
| `ic.engine` | `fit`, `evaluate`, `train_one_epoch`, `build_optimizer`, `build_scheduler`, `build_criterion` |
| `ic.metrics` | `compute_metrics`, `format_summary`, `most_confused_pairs` |
| `ic.viz` | `use_style`, `plot_history`, `plot_confusion_matrix`, `plot_per_class_metric`, `plot_class_distribution`, `plot_prediction_bars`, `show_batch` |
| `ic.inference` | `Predictor`, `collect_images` |
| `ic.gradcam` | `GradCAM`, `overlay_heatmap` |
| `ic.utils` | `set_seed`, `resolve_device`, `get_logger`, `make_run_dir`, `save_checkpoint`, `load_checkpoint`, `EarlyStopping`, `History` |

`build_datasets` returns a dict with `train` (augmented), `train_eval` (same images, same
order, no augmentation), `val`, `test`, `classes`, `train_targets`, `train_dir`, `test_dir`.

Checkpoints are plain dicts: `state_dict`, `classes`, `config`, `epoch`, `metrics`. That is
why `Predictor` needs nothing but the `.pt` file — the architecture and class order travel
with the weights.

---

## Interpreting results

Accuracy is the least interesting number in `test_metrics.json`. Read these instead:

**Balanced accuracy** — accuracy that weights every class equally. A gap from plain
accuracy means the model is leaning on the majority classes.

**Macro F1** — the average of per-class F1 scores. Drops sharply if any single class is
being handled badly, which plain accuracy will happily hide.

**The confusion matrix.** On this dataset the persistent pair is **glacier ↔ mountain**.
Snow-covered peaks genuinely belong to both categories and a fair share of the labels are
arguable, which puts a real ceiling on achievable accuracy. Worth stating plainly in a
write-up rather than chasing it with a bigger model. `most_confused_pairs()` ranks the
off-diagonal cells for you, and the script prints them after every evaluation.

**Per-class F1**, plotted weakest-first, so the class that needs attention leads.

**`errors_test.csv` sorted by confidence.** Confidently wrong predictions are the highest
signal-per-minute artifact in the project.

**The training curves.** Train and validation loss should fall together. If validation
turns upward while training keeps dropping, the model has started memorising — stop
earlier, augment harder, or raise weight decay. A visible kink at the unfreeze epoch is
expected and healthy: more parameters just came online.

Record your runs here:

| Model | Image size | Epochs | Test accuracy | Macro F1 | Notes |
|---|---|---|---|---|---|
| resnet18 | 224 | 12 | — | — | baseline |
| efficientnet_b0 | 224 | 12 | — | — | |
| resnet50 | 224 | 12 | — | — | |

---

## Design decisions

A few choices that are deliberate rather than accidental, in case you are reading this to
understand the code or defend it in a review.

**Validation comes out of the train folder, not the test folder.** Model selection needs a
signal you can look at repeatedly; the test set can only be honest if you look at it once.
The split is stratified so rare classes do not vanish from validation.

**Evaluation never uses a shuffled or augmented loader.** Scoring augmented images gives
numbers that change run to run, and a shuffled loader breaks the alignment between
predictions and file paths — which would silently corrupt the error CSV.

**Checkpoints are never selected on test accuracy.** If you disable the validation split,
the code falls back to training accuracy and warns you, rather than quietly using test.

**Train and test class lists are compared at load time.** `ImageFolder` builds its label
mapping per folder. If the two folders disagree — a stray `.ipynb_checkpoints`, a renamed
class — the integer labels would mean different things on each side and every metric would
be wrong with no error raised. The loader refuses to continue instead.

**The optimiser is rebuilt at unfreeze.** Adding parameters to an existing optimiser after
freezing does not give them state or the right learning rate; rebuilding with fresh
parameter groups does.

**One plotting module.** Every figure comes from `ic/viz.py`, so charts across the scripts,
the notebook and the demo share one visual language: single-hue ramps for magnitude, a
fixed categorical order for identity, direct labels instead of a number on every point,
and recessive grids.

---

## Extending the project

**A stronger backbone.** `--set model.name=efficientnet_b0` or `convnext_tiny` usually buys
a point or two for a modest cost.

**Higher resolution.** `--set data.image_size=299`. Slower, often worth it.

**Test-time augmentation.** Average predictions over the image and its mirror — a few lines
in `Predictor.predict_proba`, typically +0.3–0.8 points.

**Mixup / CutMix.** Helps most once plain augmentation has stopped paying.

**Your own dataset.** Point `data.root` at any folder of class subdirectories. Nothing in
the pipeline is specific to six classes or to scenes — but reconsider `horizontal_flip` if
your images have a meaningful left/right (text, digits, medical laterality).

**Ensembling.** Train two or three backbones and average their probability vectors.
`Predictor.predict_proba` returns exactly what you need.

**Deployment.** Export with `torch.onnx.export` or `torch.jit.script`. The checkpoint
already carries the class order, so the label mapping travels with the model.

On this dataset specifically, cleaning the glacier/mountain label boundary is probably
worth more than any architecture change.

---

## Troubleshooting

**`FileNotFoundError: Data root not found`** — the dataset is not where `data.root` points.
Run `python scripts/download_data.py --verify-only` to see what is actually on disk.

**`Could not find a 'train' split with class subfolders`** — the layout is not recognised.
Set `data.train_dir` and `data.test_dir` explicitly in the config.

**`ValueError: Train and test folders disagree on the class list`** — one folder has a
subdirectory the other does not. Usually `.ipynb_checkpoints` or `__MACOSX`; delete it.

**CUDA out of memory** — halve `data.batch_size`, or drop `data.image_size` to 150. Batch
16 at 224 fits comfortably in 4 GB.

**DataLoader workers crash / hang** — set `data.num_workers=0`. Common on Windows and
inside notebooks.

**Training is very slow** — check `torch.cuda.is_available()`. On CPU, use
`--set data.image_size=150 model.name=mobilenet_v3_large train.epochs=5` and expect
modest accuracy.

**`No checkpoints found under outputs/`** — train first, or pass `--checkpoint` explicitly.

**Validation accuracy far above training accuracy** — normal early on. Training accuracy is
measured on augmented images and validation on clean ones, so validation legitimately
leads for the first epochs.

**Kaggle 403 / 401** — token missing or unreadable. Confirm `~/.kaggle/kaggle.json` exists,
is valid JSON, and is `chmod 600`. You must also accept the dataset's terms on its Kaggle
page once, while signed in.

---

## Reproducibility

`seed` (default 42) seeds Python, NumPy and torch, and controls both the train/validation
split and head initialisation. Each run writes the exact `config.yaml` it used, so
`python scripts/train.py --config outputs/<run>/config.yaml` reproduces it.

Bit-exact reproducibility on GPU additionally needs deterministic cuDNN kernels — call
`set_seed(seed, deterministic=True)`, which trades some throughput for it. Ordinary runs
leave `cudnn.benchmark` on and will vary in the last decimal place.

---

## License and credits

Dataset by Puneet Bansal on Kaggle, originally published by Intel for an image
classification challenge. Check the dataset page for its licence terms before redistributing
images or a model trained on them.

Pretrained weights come from `torchvision`, trained on ImageNet.
