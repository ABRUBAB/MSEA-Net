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
4. **Finite-Sample Distribution-Free Calibration**: Implements **Split Conformal Prediction (SCP)** guaranteeing $95\%$ marginal coverage and **Selective Prediction** yielding an accuracy jump from **$93.50\%$ to $98.57\%$** when deferring the top $30\%$ highest-uncertainty borderline endoscopic frames ($70\%$ coverage, 840/1200 retained cases).

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

### 1. Comparison with Baseline Architectures (Kvasir-v2, N=1,200)

| Model Architecture | Accuracy (%) | Macro F1 (%) | Macro AUROC | ECE ↓ | Brier Score ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| VGG-16 | 92.08 | 92.04 | 0.9850 | 0.0413 | 0.1320 |
| ResNet-50 | 89.92 | 89.90 | 0.9920 | 0.0568 | 0.1568 |
| DenseNet-121 | 92.17 | 92.16 | 0.9899 | 0.0532 | 0.1305 |
| Inception-v3 | 90.75 | 90.75 | 0.9829 | 0.0411 | 0.1511 |
| MobileNetV3-Large | 91.58 | 91.57 | 0.9934 | 0.0648 | 0.1346 |
| EfficientNetV2-S Plain | 92.00 | 91.97 | 0.9905 | 0.0499 | 0.1272 |
| **MSEA-Net (Seed 777, Best)** | **93.50** | **93.49** | **0.9883** | **0.0632** | **0.1300** |
| **MSEA-Net (Mean ± SD, 3 seeds)** | **92.64 ± 0.68** | **92.62 ± 0.68** | **0.9883 ± 0.0010** | **0.0843 ± 0.0211** | **0.1300** |

### 2. Systematic Architectural Ablation Study (Kvasir-v2, Seed 42)

| Architectural Variant | Accuracy (%) | Macro F1 (%) | Macro AUROC | ECE ↓ | Brier Score ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MSEA-Net (Full Model)** | **92.64** | **92.62** | **0.9883** | **0.0843** | **0.1300** |
| w/o Multi-Scale (`no_multiscale`) | 91.58 | 91.56 | 0.9858 | 0.0850 | 0.1454 |
| w/o ACSA Attention (`no_attention`) | 92.67 | 92.63 | 0.9890 | 0.0894 | 0.1313 |
| w/o GFF Gated Fusion (`no_gff`) | 92.08 | 92.06 | 0.9867 | 0.0753 | 0.1368 |
| w/o EDL Head (`no_edl`) | 91.75 | 91.72 | 0.9945 | 0.0752 | 0.1262 |

### 3. Uncertainty-Governed Selective Prediction (Seed 777, N=1,200)

| Population Coverage | Deferral Rate | Retained Cases | Deferred Cases | Diagnostic Accuracy (%) | Accuracy Gain (Δ) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100% (Full Autonomous)** | 0% | 1,200 | 0 | **93.50** | Baseline |
| **95%** | 5% | 1,140 | 60 | **95.09** | +1.59% |
| **90%** | 10% | 1,080 | 120 | **96.02** | +2.52% |
| **85%** | 15% | 1,020 | 180 | **97.16** | +3.66% |
| **80%** | 20% | 960 | 240 | **97.81** | +4.31% |
| **75%** | 25% | 900 | 300 | **98.11** | +4.61% |
| **70%** | **30%** | **840** | **360** | **98.57** | **+5.07%** |
| **65%** | 35% | 780 | 420 | **98.85** | +5.35% |
| **60%** | 40% | 720 | 480 | **98.75** | +5.25% |

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
