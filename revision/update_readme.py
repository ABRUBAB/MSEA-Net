"""Regenerate the results/reproducibility parts of README.md from the revision outputs."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(HERE, "outputs", "lab")
sys.path.insert(0, HERE)
import make_comparison_figures as M  # noqa: E402

K = json.load(open(os.path.join(HERE, "outputs", "kvasir_3seed", "kvasir_reanalysis.json")))["three_seed"]
GV = json.load(open(os.path.join(HERE, "outputs", "gastrovision", "gastrovision_results.json")))["summary"]
PA = json.load(open(os.path.join(HERE, "outputs", "external", "external_validation.json")))["protocol_a_summary"]
SEEDS = (42, 123, 777)
SEL = {m: M.metrics(m) for m in M.BASE + M.ABL[1:]}


def ms(mu, sd, nd=2):
    return f"{mu:.{nd}f} ± {sd:.{nd}f}"


def acc_of(d, m):
    v = []
    for sd in SEEDS:
        z = np.load(os.path.join(LAB, d, f"{m}_seed{sd}_results.npz"))
        v.append((z["preds"] == z["targets"]).mean() * 100)
    return np.mean(v), np.std(v, ddof=1)


def row(m, bold=False):
    g = K[m]
    cells = [ms(g["accuracy"]["mean"], g["accuracy"]["sd"]), ms(g["macro_f1"]["mean"], g["macro_f1"]["sd"]),
             ms(g["macro_auc"]["mean"], g["macro_auc"]["sd"]), ms(g["ece"]["mean"], g["ece"]["sd"], 4),
             ms(g["brier"]["mean"], g["brier"]["sd"], 4), ms(*SEL[m]["err"]), ms(*SEL[m]["aurc"])]
    name = M.NAMES[m]
    if bold:
        name, cells = f"**{name}**", [f"**{c}**" for c in cells]
    return f"| {name} | " + " | ".join(cells) + " |"


base_rows = "\n".join(row(m, m == "msea_net") for m in ["vgg16", "resnet50", "densenet121", "inception_v3", "mobilenetv3", "efficientnetv2s_plain", "msea_net"])
abl_rows = "\n".join(row(m, m == "msea_net") for m in ["msea_net", "msea_no_multiscale", "msea_no_attention", "msea_no_gff", "msea_no_edl"])
import gastroendonet_shift as _sh  # noqa: E402  (shared shift-AUROC helper)
ext_rows = []
for m in ["vgg16", "resnet50", "densenet121", "inception_v3", "mobilenetv3", "efficientnetv2s_plain", "msea_net"]:
    tr, ft = acc_of("gastro_train", m), acc_of("gastro_finetune", m)
    cells = [ms(GV[m]["accuracy"]["mean"], GV[m]["accuracy"]["sd"]), ms(GV[m]["balanced_accuracy"]["mean"], GV[m]["balanced_accuracy"]["sd"]),
             ms(PA[m]["accuracy"]["mean"], PA[m]["accuracy"]["sd"]), ms(*_sh.shift_auroc(m)), ms(*tr), ms(*ft)]
    name = M.NAMES[m] if m != "msea_net" else "**MSEA-Net**"
    ext_rows.append(f"| {name} | " + " | ".join(cells) + " |")
ext_rows = "\n".join(ext_rows)

readme = f"""# MSEA-Net: Multi-Scale Evidential Attention Network for Uncertainty-Aware Gastrointestinal Endoscopy Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **MSEA-Net**, an eight-class gastrointestinal endoscopy image classifier that
returns, in a single forward pass, both class probabilities and an evidence-derived reliability score (Dirichlet
vacuity) used to accept or defer each prediction.

## Key components

1. **Adaptive Channel-Spatial Attention (ACSA)** on three EfficientNetV2-S stages, with a learnable gate
   $g_i=\\sigma(\\gamma_i)$ balancing parallel channel and spatial attention.
2. **Gated Feature Fusion (GFF)**: image-dependent softmax weights over the three projected scale descriptors.
3. **Evidential head**: non-negative evidence parameterising a Dirichlet distribution; vacuity $u=K/S$ is an
   evidence-derived reliability score (not a complete Bayesian epistemic estimate).
4. **Selective prediction**: images with $u>\\tau$ are deferred; $\\tau$ is fixed on validation data.

## Results on Kvasir-v2 (test subset, N = 1,200; three training runs per model, mean ± sample SD)

| Model | Accuracy (%) | Macro-F1 (%) | Macro-AUC (%) | ECE | Brier | Error-detection AUROC (%) | AURC (×100, lower better) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{base_rows}

Error detection and AURC use vacuity for MSEA-Net and 1 − max softmax for the baselines. MSEA-Net has the highest
mean accuracy and macro-F1 (significant against ResNet-50, Inception-v3 and MobileNetV3 in seed-matched McNemar
tests, not against EfficientNetV2-S, DenseNet-121 or VGG-16) and the best error ranking; it does not lead macro-AUC
or calibration (its probabilities are under-confident).

### Ablation (three matched runs)

| Variant | Accuracy (%) | Macro-F1 (%) | Macro-AUC (%) | ECE | Brier | Error-detection AUROC (%) | AURC (×100) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{abl_rows}

### Selective prediction

Vacuity thresholds fixed on the validation subset (70 % target) kept 69.3 % of test images with 98.60 % accuracy
(mean of three runs); at every coverage from 100 % to 50 % the MSEA-Net risk–coverage curve is the highest of the
seven architectures.

## External validation (frozen Kvasir-v2 models unless stated)

| Model | GastroVision acc. (%) | GastroVision bal. acc. (%) | GastroEndoNet acc., frozen (%) | GastroEndoNet shift-detection AUROC (%) | GastroEndoNet, trained from ImageNet (%) | GastroEndoNet, fine-tuned from Kvasir-v2 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{ext_rows}

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
"""
open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write(readme)
print("README.md regenerated")
