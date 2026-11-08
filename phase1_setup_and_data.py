"""
=============================================================================
MSEA-Net: Multi-Scale Evidential Attention Network
Phase 1: Setup, Configuration, and Data Pipeline
=============================================================================
KAGGLE INSTRUCTIONS:
1. Create a new Kaggle notebook
2. Add dataset: "Kvasir Dataset" (search: kvasir-dataset)
3. Set Accelerator: T4 x2 (recommended) or P100
4. Enable Internet: Settings → Internet → ON
5. Paste this as the FIRST cell
=============================================================================
"""

# ============================================================
# SECTION 1: Package Installation (Kaggle-safe)
# ============================================================
import subprocess
import sys

def install_packages():
    """Install required packages that aren't pre-installed on Kaggle."""
    packages = [
        'grad-cam',        # Grad-CAM, Grad-CAM++
        'shap',            # SHAP explanations
        'lime',            # LIME explanations
        'umap-learn',      # UMAP embeddings
    ]
    for pkg in packages:
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '-q', pkg]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"  ⚠️ Warning installing {pkg}: {e} (may still work)")
    print("✅ All packages installed successfully!")

install_packages()

# ============================================================
# SECTION 2: Imports
# ============================================================
import os
import gc
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
# AMP imports — compatible with PyTorch 2.0+ and 2.4+
try:
    from torch.amp import autocast as _autocast_base, GradScaler
    def autocast(dtype=torch.float16):
        return _autocast_base('cuda', dtype=dtype)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as TF

# timm (pre-installed on Kaggle)
import timm

# Scikit-learn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, brier_score_loss
)

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle commit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("✅ All imports successful!")


# ============================================================
# SECTION 3: Configuration
# ============================================================

class Config:
    """
    Central configuration. All hyperparameters in one place.
    Modify ONLY this class to change experiments.
    """
    # --- Paths (auto-detected for Kaggle vs local) ---
    IS_KAGGLE = os.path.exists('/kaggle')
    
    if IS_KAGGLE:
        # Kaggle paths — USE THIS EXACT DATASET:
        # "Kvasir v2" by paalh
        # URL: kaggle.com/datasets/plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset
        # 8000 files, 2 GB, 8 classes (1000 per class)
        DATA_ROOT = '/kaggle/input/kvasir-v2-a-gastrointestinal-tract-dataset'
        ALT_DATA_ROOTS = [
            '/kaggle/input/kvasir-v2-a-gastrointestinal-tract-dataset',
            '/kaggle/input/kvasir-dataset/kvasir-dataset-v2',
            '/kaggle/input/kvasir-dataset',
            '/kaggle/input/kvasir-v2-dataset',
            '/kaggle/input/the-kvasir-dataset',
        ]
        OUTPUT_DIR = '/kaggle/working/msea_net_results'
    else:
        DATA_ROOT = './data/kvasir-dataset-v2'
        OUTPUT_DIR = './msea_net_results'
    
    # --- Model ---
    BACKBONE = 'tf_efficientnetv2_s'  # EfficientNetV2-Small
    NUM_CLASSES = 8
    IMG_SIZE = 224
    
    # --- Training ---
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 3e-4
    LR_BACKBONE = 3e-5    # Lower lr for pretrained backbone
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    
    # --- EDL ---
    EDL_LAMBDA_MAX = 0.1  # KL regularization max weight
    EDL_ANNEALING = 'linear'  # 'linear' or 'cosine'
    
    # --- Seeds for reproducibility ---
    SEEDS = [42, 123, 777]  # 3 seeds for main model
    BASELINE_SEED = 42      # Single seed for baselines (saves time)
    
    # --- Data split ---
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # --- Augmentation ---
    AUG_ROTATION = 15
    AUG_HFLIP = True
    AUG_VFLIP = True
    AUG_COLOR_JITTER = (0.2, 0.2, 0.2, 0.1)
    
    # --- Mixed precision ---
    USE_AMP = True
    
    # --- Workers ---
    NUM_WORKERS = 2  # Safe for Kaggle (avoid too many)
    PIN_MEMORY = True
    
    # --- Class names ---
    CLASS_NAMES = [
        'dyed-lifted-polyps',
        'dyed-resection-margins', 
        'esophagitis',
        'normal-cecum',
        'normal-pylorus',
        'normal-z-line',
        'polyps',
        'ulcerative-colitis'
    ]

    @classmethod
    def find_data_root(cls):
        """Auto-detect the correct dataset path on Kaggle."""
        if os.path.exists(cls.DATA_ROOT):
            return cls.DATA_ROOT
        
        if cls.IS_KAGGLE:
            for alt_path in cls.ALT_DATA_ROOTS:
                if os.path.exists(alt_path):
                    cls.DATA_ROOT = alt_path
                    return alt_path
            
            # Last resort: search /kaggle/input/
            input_dir = '/kaggle/input'
            if os.path.exists(input_dir):
                for d in os.listdir(input_dir):
                    full = os.path.join(input_dir, d)
                    if os.path.isdir(full):
                        # Check if this contains class folders
                        subdirs = [x for x in os.listdir(full) if os.path.isdir(os.path.join(full, x))]
                        if len(subdirs) >= 7:  # Kvasir has 8 classes
                            cls.DATA_ROOT = full
                            return full
                        # Check one level deeper
                        for sd in subdirs:
                            deeper = os.path.join(full, sd)
                            deep_subdirs = [x for x in os.listdir(deeper) 
                                          if os.path.isdir(os.path.join(deeper, x))]
                            if len(deep_subdirs) >= 7:
                                cls.DATA_ROOT = deeper
                                return deeper
        
        raise FileNotFoundError(
            f"❌ Cannot find Kvasir dataset! Searched: {cls.DATA_ROOT}\n"
            f"Please add 'Kvasir Dataset' to your Kaggle notebook inputs."
        )


# ============================================================
# SECTION 4: GPU Setup & Compatibility
# ============================================================

def setup_device():
    """
    Setup GPU device with Kaggle-specific safeguards.
    Handles P100 (sm_60) and T4 (sm_75) compatibility.
    """
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected! Using CPU (will be very slow)")
        return torch.device('cpu')
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_mem:.1f} GB")
    print(f"🔧 CUDA: {torch.version.cuda}")
    print(f"🔧 PyTorch: {torch.__version__}")
    
    # Check architecture support
    try:
        arch_list = torch.cuda.get_arch_list()
        print(f"📋 Supported architectures: {arch_list}")
        
        capability = torch.cuda.get_device_capability(0)
        sm = f"sm_{capability[0]}{capability[1]}"
        print(f"📋 This GPU: {sm} (compute capability {capability[0]}.{capability[1]})")
        
        if sm not in arch_list and f"compute_{capability[0]}{capability[1]}" not in str(arch_list):
            print(f"⚠️ WARNING: {sm} may not be fully supported. "
                  f"If you get CUDA errors, switch to T4 x2 in Kaggle settings.")
    except Exception as e:
        print(f"⚠️ Could not check architecture: {e}")
    
    # Set memory-efficient settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Avoid TF32 on P100 (doesn't support it anyway)
    if 'P100' in gpu_name:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    print("✅ GPU setup complete!")
    return device


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Note: We intentionally keep cudnn.benchmark=True for speed
    # Setting deterministic=True would slow training significantly


# ============================================================
# SECTION 5: Dataset & Data Loading
# ============================================================

class KvasirDataset(Dataset):
    """
    Kvasir v2 Dataset.
    8 classes of GI endoscopy images, 1000 per class.
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, class_names: List[str] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or Config.CLASS_NAMES
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback: return a black image (handles corrupted files)
            print(f"⚠️ Error loading {img_path}: {e}")
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(is_training: bool = True):
    """
    Get image transforms for training or evaluation.
    Kaggle-safe: uses only standard torchvision transforms.
    """
    if is_training:
        return T.Compose([
            T.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),  # Resize slightly larger
            T.RandomCrop(Config.IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(Config.AUG_ROTATION),
            T.ColorJitter(*Config.AUG_COLOR_JITTER),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])


def load_dataset_paths(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan the dataset directory and return image paths + labels.
    Handles BOTH folder structures:
      - Flat:   data_root/class_name/*.jpg
      - Nested: data_root/class_name/class_name/*.jpg  (Kvasir v2 on Kaggle)
    """
    image_paths = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(data_root) 
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
    ])
    
    print(f"📂 Found {len(class_names)} classes: {class_names}")
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        
        # Check if images are directly in class_dir (flat structure)
        direct_images = [
            f for f in os.listdir(class_dir) 
            if os.path.isfile(os.path.join(class_dir, f)) 
            and os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        if len(direct_images) > 0:
            # Flat structure: images directly in class folder
            img_dir = class_dir
        else:
            # Nested structure: check for subfolder with same name
            nested_dir = os.path.join(class_dir, class_name)
            if os.path.isdir(nested_dir):
                img_dir = nested_dir
            else:
                # Try first subfolder
                subdirs = [d for d in os.listdir(class_dir) 
                          if os.path.isdir(os.path.join(class_dir, d))]
                if subdirs:
                    img_dir = os.path.join(class_dir, subdirs[0])
                else:
                    print(f"   ⚠️ No images found in {class_dir}")
                    continue
        
        count = 0
        for fname in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(img_dir, fname))
                labels.append(class_to_idx[class_name])
                count += 1
        
        print(f"   {class_name}: {count} images (from {os.path.basename(img_dir)}/)")
    
    print(f"📊 Total: {len(image_paths)} images across {len(class_names)} classes")
    
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"❌ No images found in {data_root}!\n"
            f"Expected structure: data_root/class_name/*.jpg OR data_root/class_name/class_name/*.jpg"
        )
    
    # Update Config class names if different from default
    Config.CLASS_NAMES = class_names
    Config.NUM_CLASSES = len(class_names)
    
    return image_paths, labels, class_names


def create_data_splits(image_paths: List[str], labels: List[int], seed: int = 42):
    """
    Create stratified train/val/test splits.
    Returns indices for each split.
    """
    labels_array = np.array(labels)
    n_total = len(labels)
    indices = np.arange(n_total)
    
    # First split: separate test set
    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=Config.TEST_RATIO, random_state=seed
    )
    train_val_idx, test_idx = next(splitter1.split(indices, labels_array))
    
    # Second split: separate validation from training
    val_ratio_adjusted = Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=seed
    )
    train_idx, val_idx = next(splitter2.split(train_val_idx, labels_array[train_val_idx]))
    
    # Map back to original indices
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    
    # Verify no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/Test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/Test overlap!"
    
    # Print distribution
    for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        counts = np.bincount(labels_array[idx], minlength=Config.NUM_CLASSES)
        print(f"   {name}: {len(idx)} images | per-class: {counts.tolist()}")
    
    return train_idx, val_idx, test_idx


def create_dataloaders(image_paths, labels, train_idx, val_idx, test_idx):
    """Create PyTorch DataLoaders for all splits."""
    
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    train_dataset = KvasirDataset(train_paths, train_labels, get_transforms(True))
    val_dataset = KvasirDataset(val_paths, val_labels, get_transforms(False))
    test_dataset = KvasirDataset(test_paths, test_labels, get_transforms(False))
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader, test_dataset


# ============================================================
# SECTION 6: Initialize Everything
# ============================================================

print("\n" + "=" * 60)
print("  MSEA-Net: Phase 1 — Setup & Data Pipeline")
print("=" * 60)

# Setup device
DEVICE = setup_device()

# Find dataset
print("\n📂 Locating dataset...")
DATA_ROOT = Config.find_data_root()
print(f"✅ Dataset found at: {DATA_ROOT}")

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"📁 Output directory: {Config.OUTPUT_DIR}")

# Load all image paths and labels
print("\n📖 Loading dataset...")
ALL_IMAGE_PATHS, ALL_LABELS, CLASS_NAMES = load_dataset_paths(DATA_ROOT)

# Create splits (using fixed seed for consistency across all experiments)
print("\n🔀 Creating train/val/test splits...")
TRAIN_IDX, VAL_IDX, TEST_IDX = create_data_splits(ALL_IMAGE_PATHS, ALL_LABELS, seed=42)

# Create dataloaders (will be recreated per-seed for training)
print("\n📦 Creating data loaders...")
TRAIN_LOADER, VAL_LOADER, TEST_LOADER, TEST_DATASET = create_dataloaders(
    ALL_IMAGE_PATHS, ALL_LABELS, TRAIN_IDX, VAL_IDX, TEST_IDX
)

print(f"\n✅ Phase 1 Complete!")
print(f"   Train samples: {len(TRAIN_IDX)}")
print(f"   Val samples:   {len(VAL_IDX)}")
print(f"   Test samples:  {len(TEST_IDX)}")
print(f"   Batch size:    {Config.BATCH_SIZE}")
print(f"   Classes:       {Config.NUM_CLASSES}")
print(f"   Device:        {DEVICE}")


# ============================================================
# SECTION 7: Dataset & Preprocessing Visualization
# ============================================================

print("\n" + "=" * 60)
print("  Dataset & Preprocessing Visualization")
print("=" * 60)

SAMPLE_DIR = os.path.join(Config.OUTPUT_DIR, 'figures', 'dataset_samples')
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --- Fig A: Raw Sample Images Grid (2 per class) ---
print("  📸 Generating raw sample images grid...")
fig, axes = plt.subplots(Config.NUM_CLASSES, 2, figsize=(8, Config.NUM_CLASSES * 3.5))
fig.suptitle('Kvasir v2 Dataset — Sample Images Per Class', fontsize=16, fontweight='bold', y=1.01)

# Collect 2 samples per class from the test set
class_samples_vis = defaultdict(list)
for idx in TEST_IDX:
    label = ALL_LABELS[idx]
    if len(class_samples_vis[label]) < 2:
        class_samples_vis[label].append(ALL_IMAGE_PATHS[idx])
    if all(len(v) >= 2 for v in class_samples_vis.values()):
        break

for class_idx in range(Config.NUM_CLASSES):
    for col in range(2):
        if class_idx in class_samples_vis and col < len(class_samples_vis[class_idx]):
            img_path = class_samples_vis[class_idx][col]
            try:
                img = Image.open(img_path).convert('RGB')
                axes[class_idx, col].imshow(img)
                if col == 0:
                    axes[class_idx, col].set_ylabel(
                        Config.CLASS_NAMES[class_idx].replace('-', '\n'),
                        fontsize=9, fontweight='bold', rotation=0, labelpad=70, ha='right'
                    )
            except Exception as e:
                axes[class_idx, col].text(0.5, 0.5, f'Error:\n{e}', 
                    ha='center', va='center', fontsize=8, transform=axes[class_idx, col].transAxes)
        axes[class_idx, col].set_xticks([])
        axes[class_idx, col].set_yticks([])

plt.tight_layout()
save_path = os.path.join(SAMPLE_DIR, 'dataset_samples_grid.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {save_path}")


# --- Fig B: Preprocessing Pipeline Visualization ---
print("  📸 Generating preprocessing pipeline visualization...")

# Pick one representative sample per class (first 4 classes for clarity)
n_show = min(4, Config.NUM_CLASSES)
fig, axes = plt.subplots(n_show, 5, figsize=(22, n_show * 4))
fig.suptitle('Image Preprocessing Pipeline', fontsize=18, fontweight='bold', y=1.02)

step_titles = ['(a) Original', '(b) Resized (256×256)', '(c) Random Crop (224×224)', 
               '(d) Augmented\n(Flip + Rotate + Color)', '(e) Normalized\n(ImageNet stats)']
for col, title in enumerate(step_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold')

# ImageNet mean/std for un-normalization display
img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])

for row_idx in range(n_show):
    class_idx = row_idx
    if class_idx not in class_samples_vis or len(class_samples_vis[class_idx]) == 0:
        continue
    
    img_path = class_samples_vis[class_idx][0]
    try:
        original = Image.open(img_path).convert('RGB')
    except Exception:
        continue
    
    # Step 1: Original
    axes[row_idx, 0].imshow(original)
    axes[row_idx, 0].set_ylabel(
        Config.CLASS_NAMES[class_idx].replace('-', '\n'),
        fontsize=9, fontweight='bold', rotation=0, labelpad=65, ha='right'
    )
    origW, origH = original.size
    axes[row_idx, 0].text(0.02, 0.02, f'{origW}×{origH}', fontsize=8, color='white',
                          transform=axes[row_idx, 0].transAxes,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Step 2: Resized
    resized = original.resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32), Image.BILINEAR)
    axes[row_idx, 1].imshow(resized)
    axes[row_idx, 1].text(0.02, 0.02, f'{Config.IMG_SIZE+32}×{Config.IMG_SIZE+32}', fontsize=8, color='white',
                          transform=axes[row_idx, 1].transAxes,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Step 3: Random Crop
    crop_transform = T.Compose([
        T.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
        T.RandomCrop(Config.IMG_SIZE),
    ])
    cropped = crop_transform(original)
    axes[row_idx, 2].imshow(cropped)
    axes[row_idx, 2].text(0.02, 0.02, f'{Config.IMG_SIZE}×{Config.IMG_SIZE}', fontsize=8, color='white',
                          transform=axes[row_idx, 2].transAxes,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Step 4: Augmented
    aug_transform = T.Compose([
        T.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
        T.RandomCrop(Config.IMG_SIZE),
        T.RandomHorizontalFlip(p=1.0),  # Force flip for demo
        T.RandomRotation(Config.AUG_ROTATION),
        T.ColorJitter(*Config.AUG_COLOR_JITTER),
    ])
    augmented = aug_transform(original)
    axes[row_idx, 3].imshow(augmented)
    axes[row_idx, 3].text(0.02, 0.02, 'Flipped+Rotated\n+ColorJitter', fontsize=7, color='white',
                          transform=axes[row_idx, 3].transAxes,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Step 5: Normalized (show as image with un-normalization for display)
    full_transform = get_transforms(is_training=True)
    normalized_tensor = full_transform(original)
    
    # Un-normalize for display
    norm_display = normalized_tensor.numpy().transpose(1, 2, 0)
    norm_display = norm_display * img_std + img_mean
    norm_display = np.clip(norm_display, 0, 1)
    axes[row_idx, 4].imshow(norm_display)
    axes[row_idx, 4].text(0.02, 0.02, 'μ=[.485,.456,.406]\nσ=[.229,.224,.225]', fontsize=7, color='white',
                          transform=axes[row_idx, 4].transAxes,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

for ax_row in axes:
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
save_path = os.path.join(SAMPLE_DIR, 'preprocessing_pipeline.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {save_path}")


# --- Fig C: Class Distribution ---
print("  📸 Generating class distribution chart...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Dataset Split — Class Distribution', fontsize=16, fontweight='bold')

labels_array = np.array(ALL_LABELS)
bar_colors = plt.cm.Set2(np.linspace(0, 1, Config.NUM_CLASSES))
short_class = [n.replace('dyed-lifted-', 'DL-').replace('dyed-resection-', 'DR-')
               .replace('normal-', 'N-').replace('ulcerative-', 'U-')
               .replace('polyps', 'Polyps').replace('esophagitis', 'Esoph.')
               for n in Config.CLASS_NAMES]

for ax, (split_name, split_idx) in zip(axes, [('Train', TRAIN_IDX), ('Validation', VAL_IDX), ('Test', TEST_IDX)]):
    counts = np.bincount(labels_array[split_idx], minlength=Config.NUM_CLASSES)
    bars = ax.bar(range(Config.NUM_CLASSES), counts, color=bar_colors, edgecolor='white')
    ax.set_xticks(range(Config.NUM_CLASSES))
    ax.set_xticklabels(short_class, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Count')
    ax.set_title(f'{split_name} (n={len(split_idx)})', fontweight='bold')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               str(cnt), ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
save_path = os.path.join(SAMPLE_DIR, 'class_distribution.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {save_path}")


# --- Fig D: Augmentation Variety (multiple augmentations of same image) ---
print("  📸 Generating augmentation variety visualization...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Data Augmentation Variety — Same Image, Different Random Transforms', fontsize=16, fontweight='bold')

# Pick one image
sample_path = class_samples_vis[0][0] if 0 in class_samples_vis else list(class_samples_vis.values())[0][0]
try:
    sample_img = Image.open(sample_path).convert('RGB')
except Exception:
    sample_img = Image.new('RGB', (224, 224), (128, 128, 128))

# Show original in top-left
axes[0, 0].imshow(sample_img)
axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

# Show 9 different augmented versions
train_transform = get_transforms(is_training=True)
for i in range(1, 10):
    row, col = divmod(i, 5)
    aug_tensor = train_transform(sample_img)
    aug_np = aug_tensor.numpy().transpose(1, 2, 0) * img_std + img_mean
    aug_np = np.clip(aug_np, 0, 1)
    axes[row, col].imshow(aug_np)
    axes[row, col].set_title(f'Augmented #{i}', fontsize=10)
    axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

plt.tight_layout()
save_path = os.path.join(SAMPLE_DIR, 'augmentation_variety.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {save_path}")

print(f"\n  📁 All dataset visualizations saved to: {SAMPLE_DIR}/")
print(f"  ✅ Phase 1 visualization complete!")
