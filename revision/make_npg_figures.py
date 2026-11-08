"""Redraw the data figures of the revised manuscript in the ggsci NPG palette (npg_style.py), at print size.

Figure 4  training dynamics (all_histories.json of the original Kaggle run)
Figure 5  three-seed confusion matrices          (make_kvasir_figures_tables.figure)
Figure 6  baseline comparison                    (make_comparison_figures.bars)
Figure 7  Seed 777 ROC and precision-recall curves (saved test probabilities)
Figure 8  ablation                               (make_comparison_figures.bars)
Figure 9  selective prediction, all architectures (make_comparison_figures.selective)
Figure 13 external validation                    (make_external_figure.main)
Figure 14 Grad-CAM / Grad-CAM++, Figure 15 LIME, Figure 16 UMAP (arrays from xai_compute.py)
Figures 10-12 were already drawn in NPG colours; Figure 2 is produced on the lab PC (make_preprocessing_figure.py);
Figures 1 and 3 are the authors' diagrams.

Usage: python make_npg_figures.py [--install]   (--install copies the PNGs to IBMED/FigureN.png)
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import npg_style as S  # noqa: E402

RES = os.path.join(ROOT, "results", "results", "msea_net_results")
XAI = os.path.join(HERE, "outputs", "lab", "xai_npg", "xai_seed777.npz")
OUT = os.path.join(HERE, "outputs", "figures_npg")
IBMED = os.path.join(ROOT, "IBMED")
CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
           'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
SHORT = ['DL-polyps', 'DR-margins', 'Esophagitis', 'N-cecum', 'N-pylorus', 'N-z-line', 'Polyps', 'UC']
BASELINES = ["vgg16", "resnet50", "densenet121", "inception_v3", "mobilenetv3", "efficientnetv2s_plain"]
NAMES = {"msea_net": "MSEA-Net", "vgg16": "VGG-16", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
         "inception_v3": "Inception-v3", "mobilenetv3": "MobileNetV3-Large", "efficientnetv2s_plain": "EfficientNetV2-S"}


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def tidy(ax, grid="both"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(S.AXIS)
    ax.tick_params(colors=S.INK2, length=2)
    if grid:
        ax.grid(axis=grid, color=S.GRID, lw=.5)
    ax.set_axisbelow(True)


def training_curves(path):
    plt = plt_()
    S.use_style(7)
    H = json.load(open(os.path.join(RES, "all_histories.json")))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.9))
    panels = [("train_loss", "(a) Training loss", "Loss"), ("train_acc", "(b) Training accuracy", "Accuracy"),
              ("val_f1", "(c) Validation macro-F1", "Macro-F1"), ("val_auc", "(d) Validation macro-AUC", "Macro-AUC")]
    for ax, (key, title, ylab) in zip(axs.flat, panels):
        for m in BASELINES:
            h = H[m][key]
            ax.plot(range(1, len(h) + 1), h, color=S.MODEL_COLOR[m], lw=1.1, alpha=.9, label=NAMES[m])
        for i, sd in enumerate((42, 123, 777)):
            h = H[f"msea_net_seed{sd}"][key]
            ax.plot(range(1, len(h) + 1), h, color=S.RED, lw=1.4, alpha=.75, zorder=5,
                    label="MSEA-Net (three runs)" if i == 0 else None)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        ax.set_xlim(0.5, 30.5)
        tidy(ax)
    for ax in axs[1]:
        ax.set_xlabel("Epoch")
    h, l = axs[0, 0].get_legend_handles_labels()
    order = [l.index("MSEA-Net (three runs)")] + [i for i in range(len(l)) if l[i] != "MSEA-Net (three runs)"]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.0, w_pad=1.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def roc_pr(path):
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
    from sklearn.preprocessing import label_binarize
    plt = plt_()
    S.use_style(7)
    z = np.load(os.path.join(RES, "msea_net_seed777_results.npz"))
    yb, p = label_binarize(z["targets"], classes=range(8)), z["probs"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for i in range(8):
        f, t, _ = roc_curve(yb[:, i], p[:, i])
        axs[0].plot(f, t, color=S.CLASS_COLOR[i], lw=1.1, label=f"{SHORT[i]} (AUC = {auc(f, t):.3f})")
        pr, rc, _ = precision_recall_curve(yb[:, i], p[:, i])
        axs[1].plot(rc, pr, color=S.CLASS_COLOR[i], lw=1.1, label=f"{SHORT[i]} (AP = {average_precision_score(yb[:, i], p[:, i]):.3f})")
    axs[0].plot([0, 1], [0, 1], ls="--", color=S.INK2, lw=.7)
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axs[1].set(xlabel="Recall", ylabel="Precision")
    axs[0].set_title("(a) ROC curves (one-versus-rest)", loc="left")
    axs[1].set_title("(b) Precision-recall curves", loc="left")
    axs[0].legend(loc="lower right", frameon=False, handlelength=1.4)
    axs[1].legend(loc="lower left", frameon=False, handlelength=1.4)
    for ax in axs:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.02)
        tidy(ax)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _overlay(img, rgba, w=0.45):
    return np.clip((1 - w) * img + w * rgba[..., :3], 0, 1)


def gradcam(path):
    plt = plt_()
    S.use_style(6.5)
    x = np.load(XAI)
    cols = ["Original", "Grad-CAM", "Grad-CAM++", "Overlay"]
    fig, axs = plt.subplots(8, 4, figsize=(3.45, 6.75), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for r in range(8):
        img, gc, gcpp = x["images"][r], x["gradcam"][r], x["gradcampp"][r]
        for c, im in enumerate([img, gc, gcpp, _overlay(img, S.NPG_HEAT(gcpp))]):
            ax = axs[r, c]
            ax.imshow(im, cmap=S.NPG_HEAT if c in (1, 2) else None, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=6.5)
        axs[r, 0].set_ylabel(CLASSES[r].replace("-", "\n", 1), rotation=0, ha="right", va="center", fontsize=5.8, labelpad=4)
    cax = fig.add_axes([0.42, 0.075, 0.42, 0.008])
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=S.NPG_HEAT), cax=cax, orientation="horizontal")
    cb.set_label("Normalized class activation", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.97, bottom=0.11)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _boundaries(seg, sel):
    b = np.zeros(seg.shape, bool)
    b[:-1, :] |= seg[:-1, :] != seg[1:, :]
    b[1:, :] |= seg[1:, :] != seg[:-1, :]
    b[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    b[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    return b & sel


def lime(path):
    plt = plt_()
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    S.use_style(7)
    x = np.load(XAI)
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.95), gridspec_kw={"wspace": 0.05, "hspace": 0.18})
    for k, ax in enumerate(axs.flat):
        img, seg, mask = x["images"][k], x["lime_segments"][k], x["lime_mask"][k]
        out = img.copy()
        for val, col in ((1, S.TEAL), (-1, S.RED)):
            sel = mask == val
            out[sel] = 0.55 * img[sel] + 0.45 * np.array(to_rgb(col))
        out[_boundaries(seg, mask != 0)] = 1.0
        ax.imshow(out)
        ax.set_title(CLASSES[k], fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend(handles=[Patch(color=S.TEAL, label="Supports the predicted class"),
                        Patch(color=S.RED, label="Opposes the predicted class")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def umap_fig(path):
    plt = plt_()
    S.use_style(7)
    x = np.load(XAI)
    e, y, p, u = x["umap"], x["targets"], x["preds"], x["uncertainties"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55))
    for c in range(8):
        m = y == c
        axs[0].scatter(e[m, 0], e[m, 1], s=3, color=S.CLASS_COLOR[c], alpha=.75, lw=0, label=CLASSES[c])
    axs[0].set_title("(a) Ground-truth class", loc="left")
    sc = axs[1].scatter(e[:, 0], e[:, 1], c=u, cmap=S.NPG_HEAT, s=3, alpha=.85, lw=0,
                        vmin=np.percentile(u, 1), vmax=np.percentile(u, 99))
    cb = fig.colorbar(sc, ax=axs[1], fraction=0.046, pad=0.02)
    cb.set_label("Vacuity $u$")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5)
    axs[1].set_title("(b) Evidential uncertainty", loc="left")
    ok = p == y
    axs[2].scatter(e[ok, 0], e[ok, 1], s=3, color=S.NAVY, alpha=.5, lw=0, label=f"Correct ({ok.sum():,})")
    axs[2].scatter(e[~ok, 0], e[~ok, 1], s=9, color=S.RED, marker="x", lw=.7, label=f"Incorrect ({(~ok).sum()})")
    axs[2].set_title("(c) Prediction correctness", loc="left")
    axs[2].legend(loc="best", frameon=False, markerscale=2, handletextpad=.3)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(S.AXIS)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, markerscale=3,
               bbox_to_anchor=(0.5, -0.1), handletextpad=.2, columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    import make_comparison_figures as M
    import make_external_figure as E
    import make_kvasir_figures_tables as K
    files = {}
    files[4] = os.path.join(OUT, "fig04_training_curves.png"); training_curves(files[4])
    files[5] = os.path.join(OUT, "fig05_confusion_3seeds.png"); K.figure(files[5])
    files[6] = os.path.join(OUT, "fig06_baselines.png")
    M.bars(M.BASE, files[6], [("acc", "(a) Accuracy (%)", 87.5, 94.5, False), ("f1", "(b) Macro-F1 (%)", 87.5, 94.5, False),
                               ("err", "(c) Error-detection AUROC (%)", 70, 92, False), ("aurc", "(d) AURC (×100)", 0, 4.9, True)], ncols=2)
    files[7] = os.path.join(OUT, "fig07_roc_pr.png"); roc_pr(files[7])
    files[8] = os.path.join(OUT, "fig08_ablation.png")
    M.bars(M.ABL, files[8], [("acc", "(a) Accuracy (%)", 89.5, 94.5, False), ("f1", "(b) Macro-F1 (%)", 89.5, 94.5, False),
                             ("aurc", "(c) AURC (×100)", 0, 2.6, True)], width=7.0)
    files[9] = os.path.join(OUT, "fig09_selective.png"); M.selective(files[9])
    E.OUT = OUT
    E.main()
    files[13] = os.path.join(OUT, "fig13_external.png")
    shutil.move(os.path.join(OUT, "fig_external.png"), files[13])
    files[14] = os.path.join(OUT, "fig14_gradcam.png"); gradcam(files[14])
    files[15] = os.path.join(OUT, "fig15_lime.png"); lime(files[15])
    files[16] = os.path.join(OUT, "fig16_umap.png"); umap_fig(files[16])
    if "--install" in sys.argv:
        for n, f in files.items():
            shutil.copy2(f, os.path.join(IBMED, f"Figure{n}.png"))
        print("installed:", ", ".join(f"Figure{n}" for n in files))
    print("written to", OUT)


if __name__ == "__main__":
    main()
