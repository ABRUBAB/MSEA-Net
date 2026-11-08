"""Shared helper: AUROC of the uncertainty score for separating GastroEndoNet images from Kvasir-v2 test images
(three runs; vacuity for MSEA-Net, 1 - max softmax for the baselines)."""
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "lab")
SEEDS = (42, 123, 777)
_KEEP = None


def _keep():
    global _KEEP
    if _KEEP is None:
        _KEEP = {int(r["id"]) for r in csv.DictReader(open(os.path.join(LAB, "gastroendonet_manifest.csv"))) if r["keep_external"] == "True"}
    return _KEEP


def shift_auroc(m):
    vals = []
    for sd in SEEDS:
        ze = np.load(os.path.join(LAB, "external_preds", f"{m}_seed{sd}_gastroendonet.npz"))
        zt = np.load(os.path.join(LAB, "kvasir_preds_all", f"{m}_seed{sd}_kvasir_test.npz"))
        mk = np.array([i in _keep() for i in ze["ids"]])
        f = (lambda z: z["uncertainties"] if "uncertainties" in z.files else 1 - z["probs"].max(1))
        se, st = f(ze)[mk], f(zt)
        vals.append(roc_auc_score(np.r_[np.zeros(len(st)), np.ones(len(se))], np.r_[st, se]) * 100)
    return np.mean(vals), np.std(vals, ddof=1)
