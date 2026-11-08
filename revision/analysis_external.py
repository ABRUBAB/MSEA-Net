"""External validation analysis (GastroEndoNet) - Protocol A (frozen Kvasir-v2 models) and
Protocol B (models trained on GastroEndoNet).

Protocol A inputs: <run>_gastroendonet.npz (8-class probabilities on all 4,006 images) and the
Kvasir-v2 validation/test predictions of the same runs (<run>_kvasir_{val,test}.npz).
Mapping (pre-specified): GERD->esophagitis(2), GERD Normal->normal-z-line(5),
Polyp->polyps(6), Polyp Normal->normal-cecum(3).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

GEN = ["GERD", "GERD Normal", "Polyp", "Polyp Normal"]
MAP = np.array([2, 5, 6, 3])
COV = [1.0, .9, .8, .7, .6, .5]
PRETTY = {"msea_net": "MSEA-Net", "vgg16": "VGG-16", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
          "inception_v3": "Inception-v3", "mobilenetv3": "MobileNetV3-Large",
          "efficientnetv2s_plain": "EfficientNetV2-S", "msea_no_multiscale": "w/o multi-scale",
          "msea_no_attention": "w/o ACSA", "msea_no_gff": "w/o GFF", "msea_no_edl": "w/o EDL"}


def ece15(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    e = 0.0
    b = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(b[:-1], b[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def brier(p, y):
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


def boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        try:
            v.append(fn(ix))
        except ValueError:
            pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def per_class(cm):
    n = cm.sum(); tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = n - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"TP": tp.tolist(), "FN": fn.tolist(), "FP": fp.tolist(), "TN": tn.tolist(),
                "sensitivity": (tp / (tp + fn) * 100).tolist(), "specificity": (tn / (tn + fp) * 100).tolist(),
                "precision": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1) * 100, 0).tolist(),
                "f1": (2 * tp / (2 * tp + fn + fp) * 100).tolist(), "ovr_accuracy": ((tp + tn) / n * 100).tolist()}


def selective_at(score, correct, coverages):
    order = np.argsort(score, kind="stable")
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(score))))
        out.append(float(correct[order[:k]].mean() * 100))
    return out


def protocol_a(ext_dir, kv_dir, manifest):
    rows = list(csv.DictReader(open(manifest)))
    keep = {int(r["id"]) for r in rows if r["keep_external"] == "True"}
    res = {}
    for f in sorted(glob.glob(os.path.join(ext_dir, "*_gastroendonet.npz"))):
        run = os.path.basename(f)[: -len("_gastroendonet.npz")]
        z = np.load(f)
        m = np.array([i in keep for i in z["ids"]])
        p8, y = z["probs"][m].astype(np.float64), z["label4"][m]
        p4 = p8[:, MAP] / p8[:, MAP].sum(1, keepdims=True)
        pred4 = p4.argmax(1)
        corr4 = pred4 == y
        pred8 = p8.argmax(1)
        corr8 = pred8 == MAP[y]
        cm = confusion_matrix(y, pred4, labels=range(4))
        o = {"n": int(m.sum()), "accuracy": float(corr4.mean() * 100), "accuracy_ci95": boot(lambda ix: corr4[ix].mean() * 100, len(y)),
             "macro_f1": float(f1_score(y, pred4, average="macro") * 100),
             "macro_f1_ci95": boot(lambda ix: f1_score(y[ix], pred4[ix], average="macro") * 100, len(y)),
             "macro_auc": float(roc_auc_score(y, p4, multi_class="ovr", average="macro") * 100),
             "ece": ece15(p4, y), "brier": brier(p4, y), "accuracy_unrestricted8": float(corr8.mean() * 100),
             "confusion_matrix": cm.tolist(), "per_class": per_class(cm),
             "pred8_distribution": [np.bincount(pred8[y == k], minlength=8).tolist() for k in range(4)]}
        up = np.isin(y, [0, 1]); s_up = p8[up, 2] / (p8[up, 2] + p8[up, 5]); l_up = (y[up] == 0).astype(int)
        lo = np.isin(y, [2, 3]); s_lo = p8[lo, 6] / (p8[lo, 6] + p8[lo, 3]); l_lo = (y[lo] == 2).astype(int)
        o["auroc_gerd_vs_normal"] = float(roc_auc_score(l_up, s_up) * 100)
        o["auroc_gerd_vs_normal_ci95"] = boot(lambda ix: roc_auc_score(l_up[ix], s_up[ix]) * 100, len(l_up))
        o["auroc_polyp_vs_normal"] = float(roc_auc_score(l_lo, s_lo) * 100)
        o["auroc_polyp_vs_normal_ci95"] = boot(lambda ix: roc_auc_score(l_lo[ix], s_lo[ix]) * 100, len(l_lo))
        o["sens_spec_gerd@0.5"] = [float(((s_up >= .5) & (l_up == 1)).sum() / l_up.sum() * 100), float(((s_up < .5) & (l_up == 0)).sum() / (1 - l_up).sum() * 100)]
        o["sens_spec_polyp@0.5"] = [float(((s_lo >= .5) & (l_lo == 1)).sum() / l_lo.sum() * 100), float(((s_lo < .5) & (l_lo == 0)).sum() / (1 - l_lo).sum() * 100)]
        msp_ext = 1 - p8.max(1)
        o["err_auroc_msp"] = float(roc_auc_score((~corr4).astype(int), msp_ext) * 100)
        o["selective_msp_rank"] = selective_at(msp_ext, corr4, COV)
        kvv = os.path.join(kv_dir, f"{run}_kvasir_val.npz")
        kvt = os.path.join(kv_dir, f"{run}_kvasir_test.npz")
        if os.path.exists(kvv) and os.path.exists(kvt):
            zv, zt = np.load(kvv), np.load(kvt)
            msp_v, msp_t = 1 - zv["probs"].max(1), 1 - zt["probs"].max(1)
            corr_t = zt["preds"] == zt["targets"]
            o["locked_msp"] = []
            for c in (.9, .8, .7, .6):
                tau = float(np.quantile(msp_v, c))
                o["locked_msp"].append({"val_coverage": c, "tau": tau,
                                        "kvasir_test_coverage": float((msp_t <= tau).mean() * 100),
                                        "kvasir_test_sel_acc": float(corr_t[msp_t <= tau].mean() * 100),
                                        "external_coverage": float((msp_ext <= tau).mean() * 100),
                                        "external_sel_acc": float(corr4[msp_ext <= tau].mean() * 100) if (msp_ext <= tau).any() else float("nan")})
            if "uncertainties" in z.files and "uncertainties" in zv.files:
                u = z["uncertainties"][m].astype(np.float64)
                uv, ut = zv["uncertainties"].astype(np.float64), zt["uncertainties"].astype(np.float64)
                o["u_external_mean"], o["u_external_sd"], o["u_external_median"] = float(u.mean()), float(u.std(ddof=1)), float(np.median(u))
                o["u_kvasir_test_mean"], o["u_kvasir_test_sd"], o["u_kvasir_test_median"] = float(ut.mean()), float(ut.std(ddof=1)), float(np.median(ut))
                o["u_per_class_external"] = [float(u[y == k].mean()) for k in range(4)]
                lab = np.r_[np.zeros(len(ut)), np.ones(len(u))]
                o["shift_auroc_u"] = float(roc_auc_score(lab, np.r_[ut, u]) * 100)
                o["shift_auroc_msp"] = float(roc_auc_score(lab, np.r_[msp_t, msp_ext]) * 100)
                o["mannwhitney_p"] = float(stats.mannwhitneyu(u, ut, alternative="greater").pvalue)
                o["err_auroc_u"] = float(roc_auc_score((~corr4).astype(int), u) * 100)
                o["err_auroc_u_ci95"] = boot(lambda ix: roc_auc_score((~corr4[ix]).astype(int), u[ix]) * 100, len(u))
                o["selective_u_rank"] = selective_at(u, corr4, COV)
                o["locked_u"] = []
                for c in (.9, .8, .7, .6):
                    tau = float(np.quantile(uv, c))
                    acc_ext = float(corr4[u <= tau].mean() * 100) if (u <= tau).any() else float("nan")
                    o["locked_u"].append({"val_coverage": c, "tau": tau,
                                          "kvasir_test_coverage": float((ut <= tau).mean() * 100),
                                          "kvasir_test_sel_acc": float(corr_t[ut <= tau].mean() * 100),
                                          "external_coverage": float((u <= tau).mean() * 100),
                                          "external_sel_acc": acc_ext, "external_n_accepted": int((u <= tau).sum())})
                o["_u"] = u.tolist(); o["_ut"] = ut.tolist()
        o["_msp_ext"] = msp_ext.tolist()
        o["_corr4"] = corr4.astype(int).tolist()
        res[run] = o
    return res


def protocol_b(train_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(train_dir, "*_results.npz"))):
        run = os.path.basename(f)[: -len("_results.npz")]
        z = np.load(f)
        p, y, pr = z["probs"].astype(np.float64), z["targets"], z["preds"]
        cm = confusion_matrix(y, pr, labels=range(4))
        corr = pr == y
        o = {"n": int(len(y)), "accuracy": float(corr.mean() * 100), "accuracy_ci95": boot(lambda ix: corr[ix].mean() * 100, len(y)),
             "macro_f1": float(f1_score(y, pr, average="macro") * 100), "macro_auc": float(roc_auc_score(y, p, multi_class="ovr", average="macro") * 100),
             "ece": ece15(p, y), "brier": brier(p, y), "confusion_matrix": cm.tolist(), "per_class": per_class(cm),
             "err_auroc_msp": float(roc_auc_score((~corr).astype(int), 1 - p.max(1)) * 100) if (~corr).any() else float("nan")}
        if "uncertainties" in z.files and (~corr).any():
            u = z["uncertainties"].astype(np.float64)
            o["err_auroc_u"] = float(roc_auc_score((~corr).astype(int), u) * 100)
            o["selective_u_rank"] = selective_at(u, corr, COV)
        sj = os.path.join(train_dir, f"{run}_summary.json")
        if os.path.exists(sj):
            js = json.load(open(sj))
            o["best_epoch"], o["best_val_f1"], o["train_time_s"] = js["best_epoch"], js["best_val_f1"], js["training_time_s"]
        res[run] = o
    # seed-matched McNemar vs MSEA-Net
    tests = {}
    for run in res:
        mname, seed = run.rsplit("_seed", 1)
        ref = f"msea_net_seed{seed}"
        if mname == "msea_net" or ref not in res:
            continue
        za, zb = np.load(os.path.join(train_dir, f"{ref}_results.npz")), np.load(os.path.join(train_dir, f"{run}_results.npz"))
        ca, cb = za["preds"] == za["targets"], zb["preds"] == zb["targets"]
        b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
        tests[f"{ref}_vs_{run}"] = {"b": b, "c": c, "p": float(stats.binomtest(b, b + c, .5).pvalue) if b + c else 1.0,
                                    "acc_diff": float((ca.mean() - cb.mean()) * 100)}
    return res, tests


def summarize(res, keys):
    agg = {}
    for run in res:
        m = run.rsplit("_seed", 1)[0]
        agg.setdefault(m, []).append(run)
    out = {}
    for m, runs in agg.items():
        out[m] = {"n_runs": len(runs), "runs": sorted(runs)}
        for k in keys:
            v = np.array([res[r][k] for r in runs if k in res[r]], dtype=float)
            if len(v):
                out[m][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}
    return out


def figure(res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    seq = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": axis, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.titleweight": "bold", "axes.titlesize": 10})
    ms = sorted(r for r in res if r.startswith("msea_net_seed"))
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=300, gridspec_kw={"width_ratios": [1.05, 1, 1]})
    # (a) row-normalized confusion matrix, mean over the MSEA-Net runs
    cms = np.array([np.array(res[r]["confusion_matrix"], float) for r in ms])
    rown = (cms / cms.sum(2, keepdims=True)).mean(0) * 100
    ax = axs[0]
    ax.imshow(rown, cmap=seq, vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{rown[i, j]:.1f}", ha="center", va="center", fontsize=9, color="#ffffff" if rown[i, j] > 55 else ink)
    lab = ["GERD\n(esophagitis)", "GERD Normal\n(z-line)", "Polyp\n(polyps)", "Polyp Normal\n(cecum)"]
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=7.5); ax.set_yticks(range(4)); ax.set_yticklabels(lab, fontsize=7.5)
    ax.set_xlabel("Predicted (restricted to mapped classes)"); ax.set_ylabel("GastroEndoNet label")
    ax.set_title(f"(a) MSEA-Net, row-normalized %\n(mean of {len(ms)} runs)")
    # (b) vacuity distributions
    ax = axs[1]
    u_k = np.concatenate([res[r]["_ut"] for r in ms]); u_e = np.concatenate([res[r]["_u"] for r in ms])
    bins = np.linspace(0, max(u_k.max(), u_e.max()) * 1.02, 45)
    ax.hist(u_k, bins=bins, density=True, histtype="step", lw=2, color=blue, label="Kvasir-v2 test (in-domain)")
    ax.hist(u_e, bins=bins, density=True, histtype="step", lw=2, color=orange, label="GastroEndoNet (external)")
    for v, c in ((np.median(u_k), blue), (np.median(u_e), orange)):
        ax.axvline(v, color=c, lw=1, ls="--")
    ax.set_xlabel("Dirichlet vacuity $u=K/S$"); ax.set_ylabel("Density")
    ax.set_title("(b) Uncertainty under dataset shift")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", color=grid, lw=.6); ax.set_axisbelow(True)
    # (c) selective accuracy vs coverage on GastroEndoNet
    ax = axs[2]
    covs = np.linspace(1.0, 0.1, 19)
    def curve(key, runs):
        return np.array([selective_at(np.array(res[r][key]), np.array(res[r]["_corr4"], bool), covs) for r in runs])
    cu, cm_ = curve("_u", ms), curve("_msp_ext", ms)
    for arr, col, name in ((cu, blue, "MSEA-Net, vacuity $u$"), (cm_, orange, "MSEA-Net, max-softmax prob.")):
        mu, sd = arr.mean(0), arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(len(covs))
        ax.plot(covs * 100, mu, color=col, lw=2, label=name)
        ax.fill_between(covs * 100, mu - sd, mu + sd, color=col, alpha=.15, lw=0)
    base = [r for r in res if r.startswith("efficientnetv2s_plain_seed")]
    if base:
        cb = np.array([selective_at(np.array(res[r]["_msp_ext"]), np.array(res[r]["_corr4"], bool), covs) for r in base])
        ax.plot(covs * 100, cb.mean(0), color=aqua, lw=2, label="EfficientNetV2-S, max-softmax prob.")
    pts = np.array([[(lk["external_coverage"], lk["external_sel_acc"]) for lk in res[r]["locked_u"]] for r in ms])
    ax.scatter(pts[..., 0].mean(0), pts[..., 1].mean(0), s=36, color=blue, edgecolor="#fcfcfb", lw=1.5, zorder=5,
               label="Validation-locked thresholds (90/80/70/60%)")
    ax.invert_xaxis(); ax.set_xlabel("Coverage on GastroEndoNet (%)"); ax.set_ylabel("Accuracy of retained images (%)")
    ax.set_title("(c) Selective prediction on external data")
    ax.grid(color=grid, lw=.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for a in axs[1:]:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--kvasir-preds", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gastro-train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A = protocol_a(a.external, a.kvasir_preds, a.manifest)
    keysA = ["accuracy", "macro_f1", "macro_auc", "ece", "brier", "accuracy_unrestricted8", "auroc_gerd_vs_normal",
             "auroc_polyp_vs_normal", "err_auroc_msp", "err_auroc_u", "shift_auroc_u", "u_external_mean", "u_kvasir_test_mean"]
    out = {"protocol_a": {r: {k: v for k, v in o.items() if not k.startswith("_")} for r, o in A.items()},
           "protocol_a_summary": summarize(A, keysA)}
    if a.gastro_train and os.path.isdir(a.gastro_train):
        B, tests = protocol_b(a.gastro_train)
        out["protocol_b"] = B
        out["protocol_b_tests"] = tests
        out["protocol_b_summary"] = summarize(B, ["accuracy", "macro_f1", "macro_auc", "ece", "brier", "err_auroc_msp", "err_auroc_u"])
    json.dump(out, open(os.path.join(a.out, "external_validation.json"), "w"), indent=1)
    try:
        figure(A, os.path.join(a.out, "external_validation_figure.png"))
    except Exception as e:  # figure is optional; numbers are the primary output
        print("figure failed:", type(e).__name__, e)
    print(f"{'run':34s} {'acc':>6s} {'F1':>6s} {'AUC':>6s} {'acc8':>6s} {'GERD':>6s} {'Polyp':>6s} {'ECE':>6s} {'errU':>6s} {'errMSP':>6s} {'shiftU':>6s}")
    for r, o in A.items():
        print(f"{r:34s} {o['accuracy']:6.2f} {o['macro_f1']:6.2f} {o['macro_auc']:6.2f} {o['accuracy_unrestricted8']:6.2f} "
              f"{o['auroc_gerd_vs_normal']:6.2f} {o['auroc_polyp_vs_normal']:6.2f} {o['ece']:6.3f} {o.get('err_auroc_u', float('nan')):6.2f} "
              f"{o['err_auroc_msp']:6.2f} {o.get('shift_auroc_u', float('nan')):6.2f}")
    if "protocol_b" in out:
        for m, g in out["protocol_b_summary"].items():
            print(f"B {m:24s} n={g['n_runs']} acc {g['accuracy']['mean']:.2f}±{g['accuracy']['sd']:.2f} F1 {g['macro_f1']['mean']:.2f} AUC {g['macro_auc']['mean']:.2f} ECE {g['ece']['mean']:.3f}")


if __name__ == "__main__":
    main()
