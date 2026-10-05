"""Cat vs Dog data pipeline: real photos from CIFAR-10 (classes 3 = cat, 5 = dog)."""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

CLASSES = ["cat", "dog"]
IMG_SIZE = 96
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics


def get_transforms(img_size: int = IMG_SIZE):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    train = T.Compose([
        T.Resize((img_size, img_size)), T.RandomHorizontalFlip(), T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.2), T.RandomAffine(0, translate=(0.08, 0.08)), *norm,
    ])
    return train, T.Compose([T.Resize((img_size, img_size)), *norm])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Normalised CHW tensor -> displayable HWC array."""
    return np.clip(tensor.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)


class CatDog(Dataset):
    """The cat and dog images of CIFAR-10, relabelled 0 = cat, 1 = dog."""
    def __init__(self, train: bool, transform=None, root: str = "data"):
        base = CIFAR10(root, train=train, download=True)
        keep = [i for i, t in enumerate(base.targets) if t in (3, 5)]
        self.images = base.data[keep]
        self.labels = [int(base.targets[i] == 5) for i in keep]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        return (self.transform(img) if self.transform else img), self.labels[i]


def get_dataloaders(batch_size: int = 64, val_fraction: float = 0.1, root: str = "data"):
    """Returns train / val / test loaders. The official CIFAR test split is never used for tuning."""
    train_t, eval_t = get_transforms()
    train_set, val_set = CatDog(True, train_t, root), CatDog(True, eval_t, root)
    order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(0)).tolist()
    n_val = int(len(order) * val_fraction)
    make = lambda ds, idx, shuffle: DataLoader(Subset(ds, idx), batch_size, shuffle=shuffle)
    test_set = CatDog(False, eval_t, root)
    return (make(train_set, order[n_val:], True), make(val_set, order[:n_val], False),
            DataLoader(test_set, batch_size), CLASSES)
