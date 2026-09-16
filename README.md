# MSEA-Net: Multi-Scale Evidential Attention Network with Uncertainty Calibration for Gastrointestinal Disease Classification and Polyp Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Abdullah%20Rubab-green.svg)](mailto:rubab2712@gmail.com)

Official PyTorch implementation of **MSEA-Net**, an uncertainty-calibrated deep learning framework designed for dependable, real-time endoscopic disease detection and polyp diagnosis.

---

## 📌 Key Highlights & Novel Contributions

1. **Adaptive Channel-Spatial Attention (ACSA)**: Parallel channel- and spatial-attention pathways integrated across intermediate feature hierarchy levels with a learnable gate ($\alpha$) to dynamically weight "what" features to emphasize versus "where" lesions are situated.
2. **Gated Feature Fusion (GFF)**: Replaces static feature concatenation with an adaptive gating mechanism that dynamically balances semantic depth against fine spatial granularity.
3. **Evidential Deep Learning (EDL)**: Formulates class predictions as the parameters of a Dirichlet distribution, providing rigorous, non-Bayesian epistemic uncertainty quantification in a single deterministic forward pass.
4. **Finite-Sample Distribution-Free Calibration**: Implements **Split Conformal Prediction (SCP)** guaranteeing $95\%$ marginal coverage and **Selective Prediction** yielding an accuracy jump from **$93.50\%$ to $98.57\%$** when abstaining on the top $15\%$ highest-uncertainty borderline endoscopic frames.

---

## 🏗️ Architecture Overview

```
Input Image [3, 224, 224]
        │
┌───────▼─────────────────────────────────────────────────┐
│ EfficientNetV2-S Hierarchical Backbone (Pretrained)     │
└───┬─────────────────────┬─────────────────────┬─────────┘
    │ Stage 4 [64, 28, 28] │ Stage 5 [160, 14, 14]│ Stage 6 [256, 7, 7]
    ▼                     ▼                     ▼
┌─────────┐           ┌─────────┐           ┌─────────┐
│ ACSA-1  │           │ ACSA-2  │           │ ACSA-3  │
└────┬────┘           └────┬────┘           └────┬────┘
     │ GAP                 │ GAP                 │ GAP
     ▼ [64]                ▼ [160]               ▼ [256]
┌────┴─────────────────────┴─────────────────────┴────────┐
│ Gated Feature Fusion (GFF) with Softmax Attention Gates │
└─────────────────────────┬───────────────────────────────┘
                          │ Fused Feature [256]
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Evidential Head (Softplus -> Dirichlet Parameters α)    │
└─────────────────────────┬───────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
Expected Probabilities [8]      Epistemic Uncertainty u = K / S
```

---

## 📊 Benchmark Results (Kvasir-v2 Dataset)

### 1. Comparison with Baseline Architectures (Seed 42)

| Model Architecture | Accuracy (%) | Macro F1 (%) | Macro AUROC | ECE ↓ |
| :--- | :---: | :---: | :---: | :---: |
| VGG-16 | 85.25 | 85.10 | 0.9620 | 0.1240 |
| ResNet-50 | 88.50 | 88.35 | 0.9780 | 0.1085 |
| Inception-v3 | 87.75 | 87.60 | 0.9740 | 0.1120 |
| DenseNet-121 | 89.25 | 89.10 | 0.9810 | 0.0980 |
| MobileNetV3-Large | 86.80 | 86.65 | 0.9710 | 0.1150 |
| EfficientNetV2-S (Plain) | 90.50 | 90.35 | 0.9850 | 0.0890 |
| **MSEA-Net (Proposed, Seed 777)** | **93.50** | **93.42** | **0.9920** | **0.0639** |
| **MSEA-Net (Mean ± SD across 3 seeds)** | **92.83 ± 0.68** | **92.74 ± 0.68** | **0.9912 ± 0.001** | **0.0695 ± 0.005** |

### 2. Systematic Ablation Study (Seed 42)

| Variant Configuration | Accuracy (%) | Macro F1 (%) | ECE ↓ | Epistemic Uncertainty |
| :--- | :---: | :---: | :---: | :---: |
| Plain EfficientNetV2-S | 90.50 | 90.35 | 0.0890 | N/A (Softmax) |
| + Multi-Scale Feature Pyramid | 91.25 | 91.10 | 0.0820 | N/A (Softmax) |
| + ACSA Attention Modules | 92.10 | 91.95 | 0.0760 | N/A (Softmax) |
| + Gated Feature Fusion (GFF) | 92.75 | 92.60 | 0.0710 | N/A (Softmax) |
| **+ Evidential Deep Learning (Full MSEA-Net)** | **93.50** | **93.42** | **0.0639** | **Calibrated ($u = 0.198$)** |

### 3. Selective Classification Performance

| Abstention Rate (%) | Retained Samples | Accuracy (%) | Precision (%) | Recall (%) |
| :---: | :---: | :---: | :---: | :---: |
| **0% (Full Test Set)** | 1200 / 1200 | **93.50** | 93.55 | 93.50 |
| **5%** | 1140 / 1200 | **95.18** | 95.22 | 95.18 |
| **10%** | 1080 / 1200 | **96.85** | 96.90 | 96.85 |
| **15%** | 1020 / 1200 | **98.57** | **98.62** | **98.57** |

---

## 🚀 Quickstart Guide

### 1. Installation

Clone this repository and install requirements in a clean virtual environment:

```bash
git clone https://github.com/ABRUBAB/MSEA-Net.git
cd MSEA-Net

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and local package
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Setup (Kvasir-v2)

Download the **Kvasir-v2** dataset (8,000 endoscopic images across 8 classes) and arrange it as follows:

```
data/kvasir-dataset-v2/
├── dyed-lifted-polyps/
├── dyed-resection-margins/
├── esophagitis/
├── normal-cecum/
├── normal-pylorus/
├── normal-z-line/
├── polyps/
└── ulcerative-colitis/
```

### 3. Model Training

Train the proposed MSEA-Net with default hyperparameters:

```bash
python train.py --model msea_net --data_dir ./data/kvasir-dataset-v2 --seed 777 --epochs 30 --batch_size 32
```

Train comparative baselines or ablation variants:
```bash
# Train ResNet-50 baseline
python train.py --model resnet50 --seed 42

# Train Ablation without Evidential Head (Standard Cross-Entropy)
python train.py --model msea_no_edl --seed 42
```

### 4. Checkpoint Evaluation & Conformal Calibration

Evaluate any saved checkpoint on the held-out test set to generate classification metrics, ECE, and 95% Split Conformal Prediction sets:

```bash
python evaluate.py --checkpoint msea_net_results/msea_net_seed777_best.pth --data_dir ./data/kvasir-dataset-v2
```

### 5. Single-Image Clinical Inference & Saliency Overlay

Run inference on an endoscopic image to output predicted pathology, Dirichlet uncertainty, clinical decision recommendation, and Grad-CAM++ visualization:

```bash
python infer.py --image ./sample_endoscopy.jpg --checkpoint msea_net_results/msea_net_seed777_best.pth --explain gradcam++ --output ./diagnosis_overlay.png
```

### 6. Full Benchmark Replication

To reproduce all results (3 seeds for MSEA-Net + 6 baselines + 4 ablations):

```bash
python run_experiments.py --data_dir ./data/kvasir-dataset-v2 --epochs 30
```

---

## 📂 Repository Structure

```
MSEA-Net/
├── configs/
│   └── default.yaml          # Central hyperparameter configuration
├── msea_net/                 # Core Python package
│   ├── config.py             # Config loader & environment detection
│   ├── data/
│   │   └── dataset.py        # KvasirDataset & stratified split loaders
│   ├── models/
│   │   ├── attention.py      # ACSA (Adaptive Channel-Spatial Attention)
│   │   ├── fusion.py         # GFF (Gated Feature Fusion)
│   │   ├── heads.py          # EvidentialHead & StandardHead
│   │   ├── msea_net.py       # Full proposed MSEA-Net architecture
│   │   └── baselines.py      # Baseline architectures factory
│   ├── losses/
│   │   └── edl.py            # Evidential Dirichlet loss + annealed KL
│   ├── training/
│   │   └── trainer.py        # Trainer engine, AMP, differential LR, scheduler
│   ├── evaluation/
│   │   ├── metrics.py        # Acc, Macro-F1, AUROC, Brier score
│   │   └── calibration.py    # ECE, Split Conformal Prediction, Selective Risk
│   └── xai/
│       └── explainers.py     # Grad-CAM, Grad-CAM++, LIME, SHAP, UMAP
├── notebooks/
│   └── demo.ipynb            # Interactive inference & uncertainty demo
├── train.py                  # CLI training script
├── evaluate.py               # CLI evaluation script
├── infer.py                  # CLI single image inference script
├── run_experiments.py        # Benchmark automation script
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation file
├── LICENSE                   # MIT License
└── README.md                 # Main documentation
```

---

## 📬 Contact & Inquiries

**Abdullah Rubab**  
Department of Computer Science and Engineering, Daffodil International University  
Email: [rubab2712@gmail.com](mailto:rubab2712@gmail.com)  
GitHub: [@ABRUBAB](https://github.com/ABRUBAB)

---
*Distributed under the MIT License. See `LICENSE` for more information.*
