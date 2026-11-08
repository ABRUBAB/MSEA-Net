"""Three-run comparison figures (baselines, ablations, selective prediction) for the revision.
MSEA-Net is highlighted in blue; comparators are neutral grey (EfficientNetV2-S, the shared-backbone
control, in orange on the curve plot). Error bars: sample SD over seeds 42, 123 and 777."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = [os.path.join(HERE, "..", "results", "results", "msea_net_results"), os.path.join(HERE, "outputs", "lab", "kvasir_extra")]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "IBMED")
SEEDS = (42, 123, 777)
NAMES = {"msea_net": "MSEA-Net", "efficientnetv2s_plain": "EfficientNetV2-S", "densenet121": "DenseNet-121",
         "vgg16": "VGG-16", "mobilenetv3": "MobileNetV3-Large", "inception_v3": "Inception-v3", "resnet50": "ResNet-50",
         "msea_no_attention": "w/o ACSA", "msea_no_gff": "w/o GFF", "msea_no_edl": "w/o EDL", "msea_no_multiscale": "w/o multi-scale"}
BASE = ["msea_net", "efficientnetv2s_plain", "densenet121", "vgg16", "mobilenetv3", "inception_v3", "resnet50"]
ABL = ["msea_net", "msea_no_attention", "msea_no_gff", "msea_no_edl", "msea_no_multiscale"]
sys.path.insert(0, HERE)
import npg_style as S  # noqa: E402  (ggsci NPG palette, serif fonts)

INK, INK2, GRID, AXIS = S.INK, S.INK2, S.GRID, S.AXIS
HILITE, MUTED = S.RED, S.COMPARATOR  # MSEA-Net vs comparators in bar charts


def load(run):
    for d in DIRS:
        f = os.path.join(d, f"{run}_results.npz")
        if os.path.exists(f):
            return np.load(f)
    raise FileNotFoundError(run)


def score(z):
    return z["uncertainties"] if "uncertainties" in z.files else 1 - z["probs"].max(1)


def metrics(m):
    from sklearn.metrics import f1_score, roc_auc_score
    out = {"acc": [], "f1": [], "aurc": [], "err": []}
    for s in SEEDS:
        z = load(f"{m}_seed{s}")
        y, p = z["targets"], z["preds"]
        corr = p == y
        sc = score(z)
        o = np.argsort(sc, kind="stable")
        out["acc"].append(corr.mean() * 100)
        out["f1"].append(f1_score(y, p, average="macro") * 100)
        out["aurc"].append((np.cumsum(1 - corr[o]) / np.arange(1, len(o) + 1)).mean() * 100)
        out["err"].append(roc_auc_score((~corr).astype(int), sc) * 100)
    return {k: (np.mean(v), np.std(v, ddof=1)) for k, v in out.items()}


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)


def bars(models, path, panels, ncols=None, width=7.2):
    """Horizontal bar panels drawn at print width (7.2 in = full text width), so fonts print at ~7 pt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.use_style(7)
    M = {m: metrics(m) for m in models}
    ncols = ncols or len(panels)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, (0.19 * len(models) + 0.62) * nrows), dpi=300)
    axs = np.atleast_1d(axs).ravel()
    for ax, (key, title, lo, hi, better_low) in zip(axs, panels):
        order = sorted(models, key=lambda m: M[m][key][0], reverse=not better_low)
        ys = np.arange(len(order))[::-1]
        for y, m in zip(ys, order):
            mu, sd = M[m][key]
            col = HILITE if m == "msea_net" else MUTED
            ax.barh(y, mu - lo, left=lo, height=.62, color=col, edgecolor="none")
            ax.errorbar(mu, y, xerr=sd, fmt="none", ecolor=INK, elinewidth=.8, capsize=2)
            ax.text(mu + sd + (hi - lo) * .03, y, f"{mu:.2f}", va="center", fontsize=6.5,
                    color=INK, fontweight="bold" if m == "msea_net" else "normal")
        ax.set_yticks(ys)
        ax.set_yticklabels([NAMES[m] for m in order], fontsize=7)
        for t, m in zip(ax.get_yticklabels(), order):
            t.set_fontweight("bold" if m == "msea_net" else "normal")
        ax.set_xlim(lo, hi)
        ax.set_title(title + (" (lower is better)" if better_low else ""), loc="left")
        ax.tick_params(axis="x", labelsize=6.5, length=2)
        ax.tick_params(axis="y", length=0)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        style(ax)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return M


def selective(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.use_style(6.5)
    cov = np.linspace(1.0, 0.5, 51)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.75), dpi=300)  # single-column width
    ends = {}
    for m in BASE[::-1]:
        curves = []
        for s in SEEDS:
            z = load(f"{m}_seed{s}")
            corr = z["preds"] == z["targets"]
            o = np.argsort(score(z), kind="stable")
            curves.append([corr[o[:int(round(c * len(o)))]].mean() * 100 for c in cov])
        mu = np.mean(curves, 0)
        if m == "msea_net":
            sd = np.std(curves, 0, ddof=1)
            ax.fill_between(cov * 100, mu - sd, mu + sd, color=S.RED, alpha=.15, lw=0)
            ax.plot(cov * 100, mu, color=S.RED, lw=1.8, label="MSEA-Net (vacuity $u$)", zorder=5)
        else:
            ax.plot(cov * 100, mu, color=S.MODEL_COLOR[m], lw=1.0, zorder=3)
        ends[m] = mu[-1]
    # direct labels at the curve ends, spaced so they never overlap
    items = sorted(ends.items(), key=lambda kv: kv[1])
    pos, gap = [], 0.42
    for _, v in items:
        pos.append(max(v, pos[-1] + gap) if pos else v)
    for (m, v), y in zip(items, pos):
        col = S.MODEL_COLOR[m]
        ax.plot([50, 49.4], [v, y], color=AXIS, lw=.6)
        ax.text(49.2, y, NAMES[m], fontsize=5.5, color=col, va="center", ha="left",
                fontweight="bold" if m == "msea_net" else "normal")
    ax.invert_xaxis()
    ax.set_xlim(100, 34)
    ax.set_xlabel("Coverage on the Kvasir-v2 test subset (%)")
    ax.set_ylabel("Accuracy of retained images (%)")
    ax.set_title("Selective prediction (mean of three runs)", loc="left")
    style(ax)
    ax.grid(axis="y", color=GRID, lw=.6)
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: (not l[i].startswith("MSEA"), not l[i].startswith("Eff")))
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=5.8, loc="lower right", handlelength=1.6)
    ax.tick_params(length=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    Mb = bars(BASE, os.path.join(OUT, "fig_baselines.png"),
              [("acc", "(a) Accuracy (%)", 87.5, 94.5, False), ("f1", "(b) Macro-F1 (%)", 87.5, 94.5, False),
               ("err", "(c) Error-detection AUROC (%)", 70, 92, False), ("aurc", "(d) AURC (×100)", 0, 4.9, True)], ncols=2)
    Ma = bars(ABL, os.path.join(OUT, "fig_ablation.png"),
              [("acc", "(a) Accuracy (%)", 89.5, 94.5, False), ("f1", "(b) Macro-F1 (%)", 89.5, 94.5, False),
               ("aurc", "(c) AURC (×100)", 0, 2.6, True)], width=7.0)
    selective(os.path.join(OUT, "fig_selective_compare.png"))
    for m, v in {**Mb, **Ma}.items():
        print(f"{m:22s} " + "  ".join(f"{k} {mu:.2f}±{sd:.2f}" for k, (mu, sd) in v.items()))
