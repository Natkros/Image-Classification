"""
Final Project: Image Classification - Dataset Pipeline
Author: Kritarth Saxena
GitHub: @Natkros

Handles data loading, image augmentations (flips, rotations, jitter),
and local sample generation for offline testing.
"""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# Standard ImageNet mean and standard deviation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 128):
    """
    Data augmentation pipeline.
    Applies flips, mild rotation, color jitter, and small translations on train set
    to reduce overfitting and help the model generalize on real photos.
    """
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transform, val_transform


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized PyTorch tensor back into a displayable RGB numpy array."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(np.array(IMAGENET_STD) * img + np.array(IMAGENET_MEAN), 0, 1)


def create_sample_image(category: str, width: int = 128, height: int = 128, seed: int = 42) -> Image.Image:
    """Creates a sample test image for testing the pipeline offline."""
    np.random.seed(seed)
    bg = (245, 230, 210) if category == "cat" else (210, 230, 245)
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)

    # Base head circle
    h_col = (190, 130, 70) if category == "cat" else (150, 110, 80)
    draw.ellipse([width * 0.2, height * 0.25, width * 0.8, height * 0.85], fill=h_col)

    if category == "cat":
        # Pointed triangular ears
        draw.polygon([(width * 0.25, height * 0.35), (width * 0.15, height * 0.1), (width * 0.45, height * 0.25)], fill=h_col)
        draw.polygon([(width * 0.75, height * 0.35), (width * 0.85, height * 0.1), (width * 0.55, height * 0.25)], fill=h_col)
    else:
        # Floppy ears for dogs
        draw.ellipse([width * 0.1, height * 0.3, width * 0.3, height * 0.75], fill=(120, 85, 60))
        draw.ellipse([width * 0.7, height * 0.3, width * 0.9, height * 0.75], fill=(120, 85, 60))

    # Eyes and nose
    draw.ellipse([width * 0.35, height * 0.45, width * 0.45, height * 0.55], fill=(20, 20, 20))
    draw.ellipse([width * 0.55, height * 0.45, width * 0.65, height * 0.55], fill=(20, 20, 20))
    draw.polygon([(width * 0.46, height * 0.6), (width * 0.54, height * 0.6), (width * 0.5, height * 0.66)], fill=(220, 120, 120))
    return img


def prepare_dataset(data_dir: str = "data", samples_per_class: int = 200, img_size: int = 128) -> tuple[str, list[str]]:
    """Sets up train/val/test directory splits and generates test samples."""
    classes = ["cat", "dog"]
    splits = {
        "train": int(samples_per_class * 0.70),
        "val": int(samples_per_class * 0.15),
        "test": int(samples_per_class * 0.15)
    }
    os.makedirs("sample_images", exist_ok=True)
    seed = 1000

    for split, count in splits.items():
        for cls in classes:
            folder = os.path.join(data_dir, split, cls)
            os.makedirs(folder, exist_ok=True)
            existing = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))]
            for i in range(len(existing), count):
                create_sample_image(cls, img_size, img_size, seed + i).save(os.path.join(folder, f"{cls}_{i:04d}.png"))
            seed += count

    # Save distinct high-res test images in sample_images/ for UI testing
    for cls in classes:
        for idx in range(3):
            path = os.path.join("sample_images", f"sample_{cls}_{idx+1}.png")
            if not os.path.exists(path):
                create_sample_image(cls, 256, 256, 5000 + idx * 30 + (0 if cls == "cat" else 100)).save(path)

    return data_dir, classes


def get_dataloaders(data_dir: str = "data", batch_size: int = 32, img_size: int = 128):
    """Creates standard PyTorch DataLoaders for train, val, and test splits."""
    train_t, val_t = get_transforms(img_size)
    loaders = [
        DataLoader(
            ImageFolder(os.path.join(data_dir, s), transform=train_t if s == "train" else val_t),
            batch_size=batch_size,
            shuffle=(s == "train")
        )
        for s in ["train", "val", "test"]
    ]
    return loaders[0], loaders[1], loaders[2], ["cat", "dog"]
