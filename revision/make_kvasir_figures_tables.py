"""Confusion-matrix figure for the three MSEA-Net runs and LaTeX tables derived from the matrices."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "results", "msea_net_results")
OUT = os.path.join(HERE, "outputs", "kvasir")
SHORT = ["DL-polyps", "DR-margins", "Esophagitis", "N-cecum", "N-pylorus", "N-z-line", "Polyps", "UC"]
LONG = ["Dyed-lifted polyps", "Dyed resection margins", "Esophagitis", "Normal cecum", "Normal pylorus",
        "Normal Z-line", "Polyps", "Ulcerative colitis"]


def cm_of(run):
    from sklearn.metrics import confusion_matrix
    z = np.load(os.path.join(RES, f"{run}_results.npz"))
    return confusion_matrix(z["targets"], z["preds"], labels=range(8))


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ink, ink2 = "#0b0b0b", "#52514e"
    sys.path.insert(0, HERE)
    import npg_style as S
    seq = S.NPG_SEQ_NAVY  # white -> NPG navy
    S.use_style(6.5)
    plt.rcParams.update({"xtick.color": ink2, "ytick.color": ink2})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75), dpi=300)
    for ax, s, tag in zip(axs, (42, 123, 777), "abc"):
        cm = cm_of(f"msea_net_seed{s}")
        ax.imshow(cm, cmap=seq, vmin=0, vmax=150)
        for i in range(8):
            for j in range(8):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="#ffffff" if cm[i, j] > 80 else ink, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(8)); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(8)); ax.set_yticklabels(SHORT if tag == "a" else [""] * 8, fontsize=6)
        ax.tick_params(length=0)
        ax.set_xlabel("Predicted class")
        if tag == "a":
            ax.set_ylabel("True class")
        c = int(np.trace(cm))
        ax.set_title(f"({tag}) Seed {s}" + "\n" + f"{c:,}/1,200 correct ({c / 12:.2f}%)", fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def pc(cm):
    n = cm.sum(); tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = n - tp - fn - fp
    return tp, fn, fp, tn


def table_per_class(run):
    cm = cm_of(run)
    tp, fn, fp, tn = pc(cm)
    n = cm.sum()
    lines = []
    cols = []
    for k in range(8):
        sens, spec = tp[k] / (tp[k] + fn[k]) * 100, tn[k] / (tn[k] + fp[k]) * 100
        prec, f1 = tp[k] / (tp[k] + fp[k]) * 100, 2 * tp[k] / (2 * tp[k] + fn[k] + fp[k]) * 100
        fpr, fnr, ovr = 100 - spec, 100 - sens, (tp[k] + tn[k]) / n * 100
        cols.append((sens, spec, prec, f1, fpr, fnr, ovr))
        lines.append(f"{LONG[k]} & {tp[k]} & {fn[k]} & {fp[k]} & {tn[k]:,} & {sens:.2f} & {spec:.2f} & {prec:.2f} & {f1:.2f} & {fpr:.2f} & {fnr:.2f} & {ovr:.2f}\\\\")
    m = np.array(cols).mean(0)
    lines.append(r"\midrule")
    lines.append(f"Sum / macro-average & {tp.sum():,} & {fn.sum()} & {fp.sum()} & {tn.sum():,} & {m[0]:.2f} & {m[1]:.2f} & {m[2]:.2f} & {m[3]:.2f} & {m[4]:.2f} & {m[5]:.2f} & {m[6]:.2f}\\\\")
    return "\n".join(lines), int(tp.sum()), float(m[6])


def table_f1_comparison(seed=42):
    from sklearn.metrics import f1_score
    models = ["msea_net", "efficientnetv2s_plain", "densenet121", "vgg16", "mobilenetv3", "inception_v3", "resnet50"]
    f1 = {}
    for m in models:
        z = np.load(os.path.join(RES, f"{m}_seed{seed}_results.npz"))
        f1[m] = f1_score(z["targets"], z["preds"], average=None) * 100
    lines = []
    for k in range(8):
        vals = [f1[m][k] for m in models]
        best = max(vals)
        cells = [f"\\textbf{{{v:.1f}}}" if abs(v - best) < 1e-9 else f"{v:.1f}" for v in vals]
        lines.append(f"{LONG[k]} & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    mac = [f1[m].mean() for m in models]
    best = max(mac)
    lines.append("Macro-F1 & " + " & ".join(f"\\textbf{{{v:.2f}}}" if abs(v - best) < 1e-9 else f"{v:.2f}" for v in mac) + r"\\")
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure(os.path.join(OUT, "confusion_matrices_three_seeds.png"))
    t777, c777, ovr777 = table_per_class("msea_net_seed777")
    t42, c42, ovr42 = table_per_class("msea_net_seed42")
    open(os.path.join(OUT, "table_per_class_seed777.tex"), "w").write(t777)
    open(os.path.join(OUT, "table_per_class_seed42.tex"), "w").write(t42)
    open(os.path.join(OUT, "table_f1_comparison_seed42.tex"), "w").write(table_f1_comparison(42))
    print(t777)
    print("seed777 correct", c777, "mean OvR acc", round(ovr777, 2))
    print(table_f1_comparison(42))
