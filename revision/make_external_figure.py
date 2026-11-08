"""External-validation figure: four panels in the style of the baseline figure (MSEA-Net highlighted,
mean of three runs, error bars = sample SD).
(a) GastroVision, frozen Kvasir-v2 models: balanced accuracy on the seven shared categories
(b) GastroEndoNet, frozen: AUROC for flagging GastroEndoNet images against Kvasir-v2 test images
(c) GastroEndoNet: test accuracy after training from ImageNet weights
(d) GastroEndoNet: test accuracy after fine-tuning from the Kvasir-v2 weights
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_comparison_figures as M  # noqa: E402

LAB = os.path.join(HERE, "outputs", "lab")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "IBMED")
MODELS = M.BASE


def gastroendonet_shift(m):
    keep = {int(r["id"]) for r in csv.DictReader(open(os.path.join(LAB, "gastroendonet_manifest.csv"))) if r["keep_external"] == "True"}
    vals = []
    for s in M.SEEDS:
        ze = np.load(os.path.join(LAB, "external_preds", f"{m}_seed{s}_gastroendonet.npz"))
        zt = np.load(os.path.join(LAB, "kvasir_preds_all", f"{m}_seed{s}_kvasir_test.npz"))
        mk = np.array([i in keep for i in ze["ids"]])
        se = ze["uncertainties"][mk] if "uncertainties" in ze.files else 1 - ze["probs"][mk].max(1)
        st = zt["uncertainties"] if "uncertainties" in zt.files else 1 - zt["probs"].max(1)
        vals.append(roc_auc_score(np.r_[np.zeros(len(st)), np.ones(len(se))], np.r_[st, se]) * 100)
    return np.mean(vals), np.std(vals, ddof=1)


def trained_acc(m, d):
    """GastroEndoNet test accuracy of the runs in outputs/lab/<d> (gastro_train or gastro_finetune)."""
    vals = []
    for s in M.SEEDS:
        z = np.load(os.path.join(LAB, d, f"{m}_seed{s}_results.npz"))
        vals.append((z["preds"] == z["targets"]).mean() * 100)
    return np.mean(vals), np.std(vals, ddof=1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M.S.use_style(7)
    gv = json.load(open(os.path.join(HERE, "outputs", "gastrovision", "gastrovision_results.json")))["summary"]
    # GastroVision's shared classes are imbalanced (106-812 images), so balanced accuracy is plotted;
    # accuracy, unseen-category AUROC and all other metrics are reported in the external-validation table.
    data = {
        "(a) GastroVision (frozen): balanced accuracy (%)": {m: (gv[m]["balanced_accuracy"]["mean"], gv[m]["balanced_accuracy"]["sd"]) for m in MODELS},
        "(b) GastroEndoNet (frozen): shift AUROC (%)": {m: gastroendonet_shift(m) for m in MODELS},
        "(c) GastroEndoNet, ImageNet init.: accuracy (%)": {m: trained_acc(m, "gastro_train") for m in MODELS},
        "(d) GastroEndoNet, Kvasir-v2 init.: accuracy (%)": {m: trained_acc(m, "gastro_finetune") for m in MODELS},
    }
    fig, axs = plt.subplots(2, 2, figsize=(7.2, (0.19 * len(MODELS) + 0.62) * 2), dpi=300)
    for ax, (title, d) in zip(axs.ravel(), data.items()):
        order = sorted(MODELS, key=lambda m: d[m][0], reverse=True)
        lo = min(v[0] - v[1] for v in d.values())
        hi = max(v[0] + v[1] for v in d.values())
        span = hi - lo
        lo, hi = lo - 0.35 * span, hi + 0.25 * span
        ys = np.arange(len(order))[::-1]
        for y, m in zip(ys, order):
            mu, sd = d[m]
            ax.barh(y, mu - lo, left=lo, height=.62, color=M.HILITE if m == "msea_net" else M.MUTED, edgecolor="none")
            ax.errorbar(mu, y, xerr=sd, fmt="none", ecolor=M.INK, elinewidth=.8, capsize=2)
            ax.text(mu + sd + span * .03, y, f"{mu:.1f}", va="center", fontsize=6.5, fontweight="bold" if m == "msea_net" else "normal")
        ax.set_yticks(ys)
        ax.set_yticklabels([M.NAMES[m] for m in order], fontsize=7)
        for t, m in zip(ax.get_yticklabels(), order):
            t.set_fontweight("bold" if m == "msea_net" else "normal")
        ax.set_xlim(lo, hi)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", labelsize=6.5, length=2)
        ax.tick_params(axis="y", length=0)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        M.style(ax)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    fig.savefig(os.path.join(OUT, "fig_external.png"), dpi=300, bbox_inches="tight")
    for t, d in data.items():
        print(t, {m: f"{v[0]:.2f}±{v[1]:.2f}" for m, v in d.items()})


if __name__ == "__main__":
    main()
