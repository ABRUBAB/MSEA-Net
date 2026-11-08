"""GastroVision analysis: frozen Kvasir-v2 models on (a) the seven GastroVision categories that
correspond to Kvasir-v2 classes and (b) the 20 categories unseen during training.

(a) Primary: prediction restricted to the mapped classes present (renormalised probabilities);
    secondary: unrestricted eight-class argmax.  Metrics: accuracy, balanced accuracy, macro-F1,
    per-class sensitivity, with 95% bootstrap CIs.
(b) Unfamiliar-category detection: AUROC of the model's native uncertainty (vacuity for EDL
    models, 1 - max softmax otherwise) for separating unseen-category images from mapped-category
    images, and the fraction accepted at Kvasir-validation-locked thresholds.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score

KV = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
      'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--kvasir-preds", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap-threshold", type=float, default=0.97,
                    help="exclude images whose nearest Kvasir-v2 image has cosine >= this value (near-duplicate frames); "
                         "0.97 was fixed after visual inspection of the closest pairs and before any model was evaluated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = {int(r["id"]): r for r in csv.DictReader(open(a.manifest))}
    for r in rows.values():
        r["keep_external"] = str(r["keep_external"] == "True" and float(r["nn_kvasir_cos"]) < a.overlap_threshold)
    res = {}
    for f in sorted(glob.glob(os.path.join(a.preds, "*_gastroendonet.npz"))):
        run = os.path.basename(f)[: -len("_gastroendonet.npz")]
        z = np.load(f)
        keep = np.array([rows[int(i)]["keep_external"] == "True" for i in z["ids"]])
        lab = z["label4"].astype(int)
        p8 = z["probs"].astype(np.float64)
        seen = keep & (lab >= 0)
        unseen = keep & (lab < 0)
        present = sorted(set(lab[seen]))
        ys, ps = lab[seen], p8[seen]
        pr_r = np.array(present)[ps[:, present].argmax(1)]
        pr_u = ps.argmax(1)
        rng = np.random.default_rng(0)
        IX = rng.integers(0, len(ys), (2000, len(ys)))
        acc_b = (pr_r[IX] == ys[IX]).mean(1) * 100
        o = {"n_seen": int(seen.sum()), "n_unseen": int(unseen.sum()), "classes_present": [KV[c] for c in present],
             "accuracy": float((pr_r == ys).mean() * 100), "accuracy_ci95": [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))],
             "balanced_accuracy": float(recall_score(ys, pr_r, labels=present, average="macro") * 100),
             "macro_f1": float(f1_score(ys, pr_r, labels=present, average="macro") * 100),
             "accuracy_unrestricted": float((pr_u == ys).mean() * 100),
             "balanced_accuracy_unrestricted": float(recall_score(ys, pr_u, labels=present, average="macro") * 100),
             "sensitivity_per_class": {KV[c]: float((pr_r[ys == c] == c).mean() * 100) for c in present}}
        if "uncertainties" in z.files:
            sc_all = z["uncertainties"].astype(np.float64)
        else:
            sc_all = 1 - p8.max(1)
        s_seen, s_unseen = sc_all[seen], sc_all[unseen]
        o["unseen_detection_auroc"] = float(roc_auc_score(np.r_[np.zeros(len(s_seen)), np.ones(len(s_unseen))], np.r_[s_seen, s_unseen]) * 100)
        o["score_seen_mean"], o["score_unseen_mean"] = float(s_seen.mean()), float(s_unseen.mean())
        corr = pr_r == ys
        o["err_auroc"] = float(roc_auc_score((~corr).astype(int), s_seen) * 100)
        kvv = os.path.join(a.kvasir_preds, f"{run}_kvasir_val.npz")
        if os.path.exists(kvv):
            zv = np.load(kvv)
            sv = zv["uncertainties"] if "uncertainties" in z.files else 1 - zv["probs"].max(1)
            o["locked"] = []
            for c in (0.9, 0.8, 0.7):
                tau = np.quantile(sv, c)
                acc_seen = s_seen <= tau
                o["locked"].append({"val_coverage": c, "seen_accepted_pct": float(acc_seen.mean() * 100),
                                    "seen_accepted_accuracy": float(corr[acc_seen].mean() * 100) if acc_seen.any() else float("nan"),
                                    "unseen_accepted_pct": float((s_unseen <= tau).mean() * 100)})
        res[run] = o
    agg = {}
    for run in res:
        m = run.rsplit("_seed", 1)[0]
        agg.setdefault(m, []).append(run)
    summ = {}
    for m, runs in agg.items():
        summ[m] = {"n_runs": len(runs)}
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "accuracy_unrestricted", "unseen_detection_auroc", "err_auroc"):
            v = np.array([res[r][k] for r in runs])
            summ[m][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}
        if all("locked" in res[r] for r in runs):
            for j, c in enumerate((0.9, 0.8, 0.7)):
                for k in ("seen_accepted_pct", "seen_accepted_accuracy", "unseen_accepted_pct"):
                    v = np.array([res[r]["locked"][j][k] for r in runs])
                    summ[m][f"locked{int(c * 100)}_{k}"] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
        sens = {}
        for c in res[runs[0]]["sensitivity_per_class"]:
            v = np.array([res[r]["sensitivity_per_class"][c] for r in runs])
            sens[c] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        summ[m]["sensitivity_per_class"] = sens
    json.dump({"runs": res, "summary": summ}, open(os.path.join(a.out, "gastrovision_results.json"), "w"), indent=1)
    order = ["msea_net", "efficientnetv2s_plain", "densenet121", "vgg16", "mobilenetv3", "inception_v3", "resnet50",
             "msea_no_attention", "msea_no_gff", "msea_no_edl", "msea_no_multiscale"]
    print(f"{'model':22s} {'acc':>12s} {'bal.acc':>12s} {'F1':>12s} {'acc(8-way)':>12s} {'unseenAUROC':>12s} {'errAUROC':>12s} {'unseen acc@70':>14s}")
    for m in order:
        if m not in summ:
            continue
        g = summ[m]
        f = lambda k: f"{g[k]['mean']:.2f}±{g[k]['sd']:.2f}"
        lk = g.get("locked70_unseen_accepted_pct")
        lk_txt = f"{lk['mean']:.1f}" if lk else "n/a"
        print(f"{m:22s} {f('accuracy'):>12s} {f('balanced_accuracy'):>12s} {f('macro_f1'):>12s} {f('accuracy_unrestricted'):>12s} {f('unseen_detection_auroc'):>12s} {f('err_auroc'):>12s} {lk_txt:>14s}")
    print("MSEA-Net per-class sensitivity:", {k: round(v["mean"], 1) for k, v in summ["msea_net"]["sensitivity_per_class"].items()})


if __name__ == "__main__":
    main()
