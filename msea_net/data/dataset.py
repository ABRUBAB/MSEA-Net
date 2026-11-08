import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_CLASSES = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KvasirDataset(Dataset):
    """
    Dataset wrapper for Kvasir-v2 gastrointestinal images.
    Supports on-the-fly PIL image loading and transform application.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.class_names = class_names or DEFAULT_CLASSES

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, path


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def scan_kvasir_directory(data_root: str, class_names: Optional[List[str]] = None) -> List[Tuple[str, int]]:
    """
    Collect (image path, label) pairs exactly as the published experiments did
    (``load_dataset_paths`` in the Kaggle notebook / phase1_setup_and_data.py):
    classes in the given (alphabetical) order, file names sorted, flat
    ``root/class/*.jpg`` or nested ``root/class/class/*.jpg`` layouts.
    The order matters: the stratified split is defined on this ordered list.
    """
    classes = class_names or DEFAULT_CLASSES
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_root}")

    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples = []

    for cname in classes:
        cdir = root / cname
        if not cdir.exists():
            candidates = [d for d in root.iterdir() if d.is_dir() and d.name.lower() == cname.lower()]
            if not candidates:
                continue
            cdir = candidates[0]

        direct = [f for f in os.listdir(cdir)
                  if os.path.isfile(os.path.join(cdir, f)) and os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
        if direct:
            img_dir = cdir
        elif (cdir / cname).is_dir():
            img_dir = cdir / cname
        else:
            subdirs = sorted(d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d)))
            if not subdirs:
                continue
            img_dir = cdir / subdirs[0]

        idx = class_to_idx[cname]
        for fname in sorted(os.listdir(img_dir)):
            if os.path.splitext(fname)[1].lower() in VALID_EXTENSIONS:
                samples.append((str(img_dir / fname), idx))

    if not samples:
        raise ValueError(f"No image files found in {data_root} under classes: {classes}")

    return samples


def get_transforms(img_size: int = 224, is_train: bool = True) -> T.Compose:
    """
    Preprocessing used for every reported result (identical to the Kaggle notebook):

    training   : Resize(img_size+32) -> RandomCrop(img_size) -> H-flip(0.5) -> V-flip(0.5)
                 -> RandomRotation(+-15 deg) -> ColorJitter(0.2, 0.2, 0.2, 0.1)
                 -> ToTensor -> Normalize(ImageNet)
    evaluation : Resize(img_size) -> ToTensor -> Normalize(ImageNet)   (no crop)
    """
    if is_train:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (same as set_seed in the Kaggle notebook)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_data_splits(labels, train_ratio: float = 0.70, val_ratio: float = 0.15,
                       test_ratio: float = 0.15, split_seed: int = 42):
    """
    Two-stage stratified split of the published experiments: the test subset is drawn first,
    then the validation subset is drawn from the remainder. Returns (train, val, test) indices.
    """
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=split_seed)
    train_val_idx, test_idx = next(sss1.split(indices, labels))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio / (train_ratio + val_ratio), random_state=split_seed)
    tr, va = next(sss2.split(train_val_idx, labels[train_val_idx]))
    return train_val_idx[tr], train_val_idx[va], test_idx


def get_stratified_loaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    class_names: Optional[List[str]] = None,
    split_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create the stratified 70/15/15 train/validation/test DataLoaders of the published experiments.

    The partition is defined only by ``split_seed`` (42 for every reported model, seed and ablation),
    never by the training seed, so that all checkpoints are evaluated on the same 1,200 test images.
    ``seed`` is accepted for backward compatibility and ignored for partitioning; seed the training
    run itself with :func:`set_seed`.
    """
    classes = class_names or DEFAULT_CLASSES
    samples = scan_kvasir_directory(data_root, classes)
    labels = np.array([s[1] for s in samples])
    train_idx, val_idx, test_idx = create_data_splits(labels, train_ratio, val_ratio, test_ratio, split_seed)
    
    # Create datasets with respective transforms
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    train_dataset = KvasirDataset(train_samples, transform=get_transforms(img_size, is_train=True), class_names=classes)
    val_dataset = KvasirDataset(val_samples, transform=get_transforms(img_size, is_train=False), class_names=classes)
    test_dataset = KvasirDataset(test_samples, transform=get_transforms(img_size, is_train=False), class_names=classes)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, classes
