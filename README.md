# MSEA-Net: Multi-Scale Evidential Attention Network for Uncertainty-Aware Gastrointestinal Endoscopy Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **MSEA-Net**, an eight-class gastrointestinal endoscopy image classifier that
returns, in a single forward pass, both class probabilities and an evidence-derived reliability score (Dirichlet
vacuity) used to accept or defer each prediction.

## Key components

1. **Adaptive Channel-Spatial Attention (ACSA)** on three EfficientNetV2-S stages, with a learnable gate
   $g_i=\sigma(\gamma_i)$ balancing parallel channel and spatial attention.
2. **Gated Feature Fusion (GFF)**: image-dependent softmax weights over the three projected scale descriptors.
3. **Evidential head**: non-negative evidence parameterising a Dirichlet distribution; vacuity $u=K/S$ is an
   evidence-derived reliability score (not a complete Bayesian epistemic estimate).
4. **Selective prediction**: images with $u>\tau$ are deferred; $\tau$ is fixed on validation data.

## Results on Kvasir-v2 (test subset, N = 1,200; three training runs per model, mean ± sample SD)

| Model | Accuracy (%) | Macro-F1 (%) | Macro-AUC (%) | ECE | Brier | Error-detection AUROC (%) | AURC (×100, lower better) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-16 | 91.86 ± 0.46 | 91.83 ± 0.46 | 98.84 ± 0.33 | 0.0456 ± 0.0046 | 0.1358 ± 0.0040 | 78.11 ± 3.72 | 3.51 ± 0.13 |
| ResNet-50 | 89.42 ± 0.60 | 89.41 ± 0.59 | 99.19 ± 0.05 | 0.0598 ± 0.0061 | 0.1601 ± 0.0031 | 84.71 ± 2.49 | 2.73 ± 0.42 |
| DenseNet-121 | 92.11 ± 0.17 | 92.09 ± 0.17 | 99.07 ± 0.10 | 0.0527 ± 0.0005 | 0.1275 ± 0.0029 | 80.86 ± 1.71 | 3.05 ± 0.32 |
| Inception-v3 | 90.97 ± 0.54 | 90.95 ± 0.53 | 98.66 ± 0.33 | 0.0473 ± 0.0098 | 0.1458 ± 0.0057 | 78.85 ± 2.79 | 3.95 ± 0.39 |
| MobileNetV3-Large | 91.42 ± 0.52 | 91.38 ± 0.56 | 99.28 ± 0.06 | 0.0608 ± 0.0036 | 0.1356 ± 0.0052 | 85.01 ± 0.37 | 2.18 ± 0.30 |
| EfficientNetV2-S | 92.33 ± 0.30 | 92.32 ± 0.31 | 99.03 ± 0.03 | 0.0558 ± 0.0060 | 0.1249 ± 0.0029 | 81.54 ± 0.37 | 2.58 ± 0.20 |
| **MSEA-Net** | **92.64 ± 0.83** | **92.62 ± 0.84** | **98.83 ± 0.12** | **0.0843 ± 0.0258** | **0.1320 ± 0.0125** | **87.38 ± 1.69** | **1.42 ± 0.07** |

Error detection and AURC use vacuity for MSEA-Net and 1 − max softmax for the baselines. MSEA-Net has the highest
mean accuracy and macro-F1 (significant against ResNet-50, Inception-v3 and MobileNetV3 in seed-matched McNemar
tests, not against EfficientNetV2-S, DenseNet-121 or VGG-16) and the best error ranking; it does not lead macro-AUC
or calibration (its probabilities are under-confident).

### Ablation (three matched runs)

| Variant | Accuracy (%) | Macro-F1 (%) | Macro-AUC (%) | ECE | Brier | Error-detection AUROC (%) | AURC (×100) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MSEA-Net** | **92.64 ± 0.83** | **92.62 ± 0.84** | **98.83 ± 0.12** | **0.0843 ± 0.0258** | **0.1320 ± 0.0125** | **87.38 ± 1.69** | **1.42 ± 0.07** |
| w/o multi-scale | 91.97 ± 0.38 | 91.95 ± 0.39 | 98.74 ± 0.16 | 0.0930 ± 0.0184 | 0.1401 ± 0.0064 | 87.64 ± 1.23 | 1.59 ± 0.24 |
| w/o ACSA | 92.50 ± 0.17 | 92.48 ± 0.15 | 98.85 ± 0.05 | 0.1005 ± 0.0254 | 0.1350 ± 0.0073 | 87.17 ± 1.00 | 1.57 ± 0.21 |
| w/o GFF | 91.97 ± 0.67 | 91.95 ± 0.68 | 98.63 ± 0.07 | 0.0876 ± 0.0123 | 0.1420 ± 0.0079 | 86.15 ± 1.84 | 1.75 ± 0.10 |
| w/o EDL | 91.97 ± 0.21 | 91.95 ± 0.22 | 99.40 ± 0.04 | 0.0637 ± 0.0101 | 0.1231 ± 0.0028 | 86.63 ± 0.32 | 2.00 ± 0.06 |

### Selective prediction

Vacuity thresholds fixed on the validation subset (70 % target) kept 69.3 % of test images with 98.60 % accuracy
(mean of three runs); at every coverage from 100 % to 50 % the MSEA-Net risk–coverage curve is the highest of the
seven architectures.

## External validation (frozen Kvasir-v2 models unless stated)

| Model | GastroVision acc. (%) | GastroVision bal. acc. (%) | GastroEndoNet acc., frozen (%) | GastroEndoNet shift-detection AUROC (%) | GastroEndoNet, trained from ImageNet (%) | GastroEndoNet, fine-tuned from Kvasir-v2 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-16 | 86.57 ± 0.69 | 84.02 ± 0.39 | 43.77 ± 0.21 | 75.31 ± 4.13 | 81.74 ± 0.87 | 81.97 ± 1.01 |
| ResNet-50 | 82.55 ± 0.59 | 80.60 ± 1.51 | 48.25 ± 0.19 | 84.91 ± 1.02 | 73.20 ± 1.54 | 75.38 ± 1.17 |
| DenseNet-121 | 87.62 ± 0.23 | 86.25 ± 0.71 | 51.21 ± 0.96 | 86.77 ± 1.47 | 81.80 ± 0.92 | 81.97 ± 0.76 |
| Inception-v3 | 86.09 ± 0.99 | 85.41 ± 0.22 | 44.82 ± 0.92 | 80.32 ± 0.96 | 80.46 ± 0.48 | 81.52 ± 0.42 |
| MobileNetV3-Large | 86.82 ± 0.29 | 85.88 ± 0.09 | 47.05 ± 0.33 | 81.41 ± 3.27 | 80.23 ± 1.01 | 82.69 ± 0.35 |
| EfficientNetV2-S | 87.81 ± 0.26 | 86.97 ± 0.24 | 47.62 ± 0.81 | 80.36 ± 1.22 | 84.98 ± 0.42 | 84.81 ± 0.54 |
| **MSEA-Net** | 86.51 ± 1.12 | 86.68 ± 0.79 | 44.53 ± 1.68 | 87.46 ± 1.35 | 84.20 ± 0.35 | 84.81 ± 0.39 |

GastroVision: the seven categories shared with Kvasir-v2. GastroEndoNet: four classes with different label
definitions (GERD, GERD Normal, Polyp, Polyp Normal), mapped to esophagitis, normal Z-line, polyps and normal cecum;
frozen accuracy is low for every architecture, while MSEA-Net's vacuity flags the shifted images best (AUROC for
separating GastroEndoNet from Kvasir-v2 test images). After fine-tuning from the Kvasir-v2 weights, MSEA-Net and
EfficientNetV2-S share the highest accuracy, and MSEA-Net is significantly more accurate than the other five baselines.
Both datasets were screened for exact and near-duplicate images against Kvasir-v2 before evaluation.

## Reproducibility

* **Code of record.** Every reported Kvasir-v2 number was produced by the public Kaggle notebook
  [absanim/msea-net](https://www.kaggle.com/code/absanim/msea-net), whose five cells are identical to
  `phase1_setup_and_data.py` … `phase5_evaluation_xai.py` in this repository (data: Kaggle
  `plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset`; Tesla T4, PyTorch 2.10).
* **Package.** `msea_net/` reproduces the notebook exactly (same ordered file list, seed-42 split, preprocessing and
  model outputs; verified with `revision/verify_package_split.py`). The partition is always the seed-42 split;
  `--seed` only seeds training.
* **Revision experiments** (`revision/`): `reproduce_kvasir.py` (re-runs the 13 checkpoints; 9 bit-identical,
  4 differ in 1/1,200 predictions on an RTX 5090), `train_runs.py` (extra seeds, GastroEndoNet training and
  fine-tuning), `gastroendonet_screen.py`, `gastrovision_prepare.py`, `external_eval*.py`, `analysis_*.py`,
  `benchmark_latency.py`, and the figure scripts.
* **Preprocessing.** Training: resize 256×256 → random crop 224 → horizontal/vertical flips → rotation ±15° →
  colour jitter (0.2, 0.2, 0.2, 0.1) → ImageNet normalisation. Validation/test/external: resize 224×224 →
  ImageNet normalisation (no crop).

## Quickstart

```bash
git clone https://github.com/ABRUBAB/MSEA-Net.git
cd MSEA-Net
pip install -r requirements.txt
pip install -e .
python train.py --model msea_net --data_dir ./data/kvasir-dataset-v2 --seed 777
python evaluate.py --checkpoint msea_net_results/msea_net_seed777_best.pth --data_dir ./data/kvasir-dataset-v2
python infer.py --image ./sample_endoscopy.jpg --checkpoint msea_net_results/msea_net_seed777_best.pth --explain gradcam++ --output ./overlay.png
```

Expected dataset layout: `data/kvasir-dataset-v2/<class>/*.jpg` for the eight classes (dyed-lifted-polyps,
dyed-resection-margins, esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps, ulcerative-colitis).

## Data licences

Kvasir-v2 (Simula), GastroVision (Jha et al., 2023) and GastroEndoNet (Bitto et al., Data in Brief 2025; Mendeley
Data V3, CC BY-NC-ND 4.0) are not redistributed here; download them from their original sources.

## Contact

Abdullah Rubab, Department of Computer Science and Engineering, Daffodil International University —
[rubab2712@gmail.com](mailto:rubab2712@gmail.com) · GitHub [@ABRUBAB](https://github.com/ABRUBAB)

*Distributed under the MIT License (see `LICENSE`).*
