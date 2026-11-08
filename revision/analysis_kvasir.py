"""Kvasir-v2 re-analysis of the saved test predictions (no retraining).

Everything is recomputed from the notebook's saved *_results.npz files:
  * confusion matrices and per-class one-vs-rest metrics for every run (supervisor's format)
  * overall metrics (accuracy, macro P/R/F1, macro-AUC, ECE-15, multiclass Brier)
  * seed-matched McNemar tests and paired bootstrap CIs (MSEA-Net vs each comparator, same seed)
  * uncertainty analysis for all three MSEA-Net seeds (error separation, AUROC, risk-coverage)
  * the supervisor's Excel sheet completed for every model, with live formulas

Usage: python analysis_kvasir.py --results <dir with *_results.npz> [--extra <dir> ...]
                                 --sir-xlsx sir_test.xlsx --out <dir>
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
           'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
SHORT = ["DL-Polyps", "DR-margins", "Esoph.", "N-cecum", "N-pylorus", "N-z-line", "Polyps", "U-colitis"]
PRETTY = {"msea_net": "MSEA-Net", "vgg16": "VGG-16", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
          "inception_v3": "Inception-v3", "mobilenetv3": "MobileNetV3-Large",
          "efficientnetv2s_plain": "EfficientNetV2-S (plain)", "msea_no_multiscale": "w/o multi-scale",
          "msea_no_attention": "w/o ACSA", "msea_no_gff": "w/o GFF", "msea_no_edl": "w/o EDL"}
COVERAGES = [1.0, .95, .9, .85, .8, .75, .7, .65, .6]


def ece15(probs, y, n_bins=15):
    """Identical to compute_ece() in phase3_train_utils.py."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    b = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(b[:-1], b[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.mean() > 0:
            e += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def brier(probs, y):
    k = probs.shape[1]
    return float(((probs - np.eye(k)[y]) ** 2).sum(1).mean())


def per_class(cm):
    n = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = n - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        r = {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
             "ACC": (tp + tn) / n * 100, "TPR": tp / (tp + fn) * 100, "FNR": fn / (tp + fn) * 100,
             "FPR": fp / (fp + tn) * 100, "TNR": tn / (fp + tn) * 100,
             "PREC": np.where(tp + fp > 0, tp / (tp + fp) * 100, 0.0),
             "F1": 2 * tp / (2 * tp + fn + fp) * 100, "ERROR": (fn + fp) / n * 100}
    return r


def overall(y, p, probs):
    return {"n": int(len(y)), "correct": int((p == y).sum()), "accuracy": float((p == y).mean() * 100),
            "macro_precision": float(precision_score(y, p, average="macro", zero_division=0) * 100),
            "macro_recall": float(recall_score(y, p, average="macro") * 100),
            "macro_f1": float(f1_score(y, p, average="macro") * 100),
            "macro_auc": float(roc_auc_score(y, probs, multi_class="ovr", average="macro") * 100),
            "ece": ece15(probs, y), "brier": brier(probs, y)}


def bootstrap_ci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def boot_idx(n, B=2000, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def macro_f1_batch(y, p, IX, k=8):
    """Macro-F1 for every bootstrap resample (rows of IX), vectorized via confusion counts."""
    B = IX.shape[0]
    codes = (y[IX] * k + p[IX]) + (np.arange(B)[:, None] * k * k)
    cm = np.bincount(codes.ravel(), minlength=B * k * k).reshape(B, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    fn, fp = cm.sum(2) - tp, cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fn + fp > 0, 2 * tp / (2 * tp + fn + fp), 0.0)
    return f1.mean(1) * 100


def auroc_batch(lab, score, IX):
    """Mann-Whitney AUROC for every bootstrap resample (ties averaged)."""
    from scipy.stats import rankdata
    out = np.empty(IX.shape[0])
    for b, ix in enumerate(IX):
        l, s = lab[ix], score[ix]
        n1 = l.sum(); n0 = len(l) - n1
        if n1 == 0 or n0 == 0:
            out[b] = np.nan
            continue
        r = rankdata(s)
        out[b] = (r[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def selective(u, correct):
    order = np.argsort(u, kind="stable")
    out = []
    for c in COVERAGES:
        k = int(round(c * len(u)))
        idx = order[:k]
        out.append({"coverage": c, "retained": k, "deferred": len(u) - k,
                    "errors": int(k - correct[idx].sum()), "accuracy": float(correct[idx].mean() * 100)})
    return out


def aurc(u, correct):
    order = np.argsort(u, kind="stable")
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(u) + 1)
    return float(risk.mean())


def uncertainty_block(u, correct):
    uc, ui = u[correct], u[~correct]
    t = stats.ttest_ind(ui, uc, equal_var=False)
    v1, v2 = uc.var(ddof=1) / len(uc), ui.var(ddof=1) / len(ui)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(uc) - 1) + v2 ** 2 / (len(ui) - 1))
    sp = np.sqrt(((len(uc) - 1) * uc.var(ddof=1) + (len(ui) - 1) * ui.var(ddof=1)) / (len(uc) + len(ui) - 2))
    err = (~correct).astype(int)
    au = roc_auc_score(err, u)
    v = auroc_batch(err, u, boot_idx(len(u), 1000))
    ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"n_correct": int(len(uc)), "n_incorrect": int(len(ui)),
            "u_correct_mean": float(uc.mean()), "u_correct_sd": float(uc.std(ddof=1)),
            "u_incorrect_mean": float(ui.mean()), "u_incorrect_sd": float(ui.std(ddof=1)),
            "welch_t": float(t.statistic), "welch_df": float(dof), "welch_p": float(t.pvalue),
            "cohens_d": float((ui.mean() - uc.mean()) / sp), "error_auroc": float(au), "error_auroc_ci95": ci,
            "aurc": aurc(u, correct), "selective": selective(u, correct)}


def load_runs(dirs):
    runs = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*_results.npz"))):
            name = os.path.basename(f)[: -len("_results.npz")]
            if name in runs:
                continue
            z = np.load(f)
            runs[name] = {k: z[k] for k in z.files}
    return runs


def split_name(run):
    m, s = run.rsplit("_seed", 1)
    return m, int(s)


def write_excel(template, out_path, runs, order):
    """Copy the supervisor's workbook and fill blocks for every run, keeping his formulas."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    shutil.copyfile(template, out_path)
    wb = openpyxl.load_workbook(out_path)
    ws = wb["Sheet1"]
    ws["O1"] = "Run used (seed) / note"
    ws["O1"].font = Font(bold=True)
    bold = Font(bold=True)
    fill = PatternFill("solid", fgColor="FFF2CC")
    blocks = {}            # run -> (first row, row of the column summaries or None)
    row = 19               # new blocks go below the supervisor's two original blocks (rows 2-17)
    for bi, run in enumerate(order):
        m, s = split_name(run)
        z = runs[run]
        cm = confusion_matrix(z["targets"], z["preds"], labels=range(8))
        pc = per_class(cm)
        if bi == 0:
            # Supervisor's own "Proposed" block (rows 2-9): keep his values/formulas, only verify them.
            assert run == "msea_net_seed777"
            for k in range(8):
                got = [ws[f"{c}{2 + k}"].value for c in "CDEF"]
                want = [int(pc[key][k]) for key in ("TP", "FN", "FP", "TN")]
                assert got == want, (k, got, want)
            ws["O2"] = "MSEA-Net Seed 777 = panel (c) of the confusion-matrix figure (values verified against the saved predictions)"
            blocks[run] = (2, None)
            continue
        if bi == 1:
            # Supervisor's empty VGG16 block (rows 10-17): fill it in his formula style.
            assert run == "vgg16_seed42"
            first = 10
            for k in range(8):
                r = first + k
                for col, key in zip("CDEF", ("TP", "FN", "FP", "TN")):
                    ws[f"{col}{r}"] = int(pc[key][k])
                ws[f"G{r}"] = f"=(C{r}+F{r})/1200*100"
                ws[f"H{r}"] = f"=C{r}/(C{r}+D{r})*100"
                ws[f"I{r}"] = f"=D{r}/(D{r}+C{r})*100"
                ws[f"J{r}"] = f"=E{r}/(E{r}+F{r})*100"
                ws[f"K{r}"] = f"=F{r}/(F{r}+E{r})*100"
                ws[f"L{r}"] = f"=C{r}/(C{r}+E{r})*100"
                ws[f"M{r}"] = f"=(2*C{r})/((2*C{r})+D{r}+E{r})*100"
                ws[f"N{r}"] = f"=(D{r}+E{r})/1200*100"
            ws["O10"] = "VGG-16, Seed 42 (Seeds 123 and 777 of every model are listed further below)"
            blocks[run] = (10, None)
            continue
        first = row
        for k in range(8):
            r = row + k
            ws.cell(r, 2, SHORT[k])
            for col, key in zip("CDEF", ("TP", "FN", "FP", "TN")):
                ws[f"{col}{r}"] = int(pc[key][k])
            ws[f"G{r}"] = f"=(C{r}+F{r})/1200*100"
            ws[f"H{r}"] = f"=C{r}/(C{r}+D{r})*100"
            ws[f"I{r}"] = f"=D{r}/(D{r}+C{r})*100"
            ws[f"J{r}"] = f"=E{r}/(E{r}+F{r})*100"
            ws[f"K{r}"] = f"=F{r}/(F{r}+E{r})*100"
            ws[f"L{r}"] = f"=IF(C{r}+E{r}=0,0,C{r}/(C{r}+E{r})*100)"
            ws[f"M{r}"] = f"=(2*C{r})/((2*C{r})+D{r}+E{r})*100"
            ws[f"N{r}"] = f"=(D{r}+E{r})/1200*100"
        ws.merge_cells(f"A{first}:A{first + 7}")
        ws[f"A{first}"] = "Proposed" if m == "msea_net" else PRETTY.get(m, m)
        ws[f"A{first}"].alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
        ws[f"O{first}"] = f"{PRETTY.get(m, m)}, Seed {s}"
        avg = first + 8
        ws[f"B{avg}"] = "Mean / sum"
        ws[f"B{avg}"].font = bold
        for col in "CDEF":
            ws[f"{col}{avg}"] = f"=SUM({col}{first}:{col}{first + 7})"
        for col in "GHIJKLMN":
            ws[f"{col}{avg}"] = f"=AVERAGE({col}{first}:{col}{first + 7})"
            ws[f"{col}{avg}"].fill = fill
        ws[f"N{avg}"] = f"=SUM(N{first}:N{first + 7})"
        ws[f"O{avg}"] = (f"Overall accuracy = sum(TP)/N = {int(pc['TP'].sum())}/1200 = "
                         f"{pc['TP'].sum() / cm.sum() * 100:.2f}%  (not the mean of column G)")
        blocks[run] = (first, avg)
        row = avg + 2
    for r in range(2, row):
        for col in "GHIJKLMN":
            ws[f"{col}{r}"].number_format = "0.00"
    ws.column_dimensions["O"].width = 70

    # Summary sheet: every overall number derived from the blocks with formulas
    sm = wb.create_sheet("Summary")
    hdr = ["Model", "Run / seed", "N", "Correct = sum(TP)", "Overall accuracy % = sum(TP)/N",
           "Mean of per-class ACC % (one-vs-rest)", "Macro precision %", "Macro recall % (= balanced acc.)",
           "Macro F1 %", "Sum of ERROR column %", "Overall error % = 100 - accuracy"]
    for j, h in enumerate(hdr, 1):
        sm.cell(1, j, h).font = bold
    for i, run in enumerate(order, 2):
        m, s = split_name(run)
        f, _ = blocks[run]
        rng = lambda c: f"Sheet1!{c}{f}:{c}{f + 7}"  # noqa: E731
        sm.cell(i, 1, "Proposed (MSEA-Net)" if m == "msea_net" else PRETTY.get(m, m))
        sm.cell(i, 2, f"Seed {s}")
        sm.cell(i, 3, f"=Sheet1!C{f}+Sheet1!D{f}+Sheet1!E{f}+Sheet1!F{f}")
        sm.cell(i, 4, f"=SUM({rng('C')})")
        sm.cell(i, 5, f"=D{i}/C{i}*100")
        sm.cell(i, 6, f"=AVERAGE({rng('G')})")
        sm.cell(i, 7, f"=AVERAGE({rng('L')})")
        sm.cell(i, 8, f"=AVERAGE({rng('H')})")
        sm.cell(i, 9, f"=AVERAGE({rng('M')})")
        sm.cell(i, 10, f"=SUM({rng('N')})")
        sm.cell(i, 11, f"=100-E{i}")
        for j in range(5, 12):
            sm.cell(i, j).number_format = "0.00"
    for j, w in enumerate([24, 11, 7, 16, 22, 24, 16, 22, 11, 18, 22], 1):
        sm.column_dimensions[openpyxl.utils.get_column_letter(j)].width = w

    ex = wb.create_sheet("Why the numbers differ")
    text = [
        "How the confusion matrix maps to the reported metrics (8 classes, N = 1,200 test images, 150 per class)",
        "",
        "1. Overall (multi-class) accuracy = (sum of the diagonal) / N = sum(TP) / 1200.  For Seed 777: 1,122/1,200 = 93.50%.",
        "2. Column G (ACC) is the ONE-VS-REST accuracy of each class, (TP+TN)/1200.  It treats each class as a separate",
        "   binary problem in which ~1,050 of the 1,200 images are true negatives, so it is always high (96-99.8%).",
        "   Its mean (98.38% for Seed 777) is NOT the overall accuracy and is not reported in the paper.",
        "3. Every misclassified image is counted twice in the one-vs-rest view: once as a FN of its true class and once",
        "   as a FP of the predicted class.  Hence sum(FN) = sum(FP) = number of errors (78 for Seed 777) and the",
        "   ERROR column sums to 2 x 6.50% = 13.00%.",
        "4. Because the test set is balanced, macro recall (mean TPR) = overall accuracy = balanced accuracy.",
        "5. Macro F1 = mean of column M (93.49% for Seed 777); macro precision = mean of column L.",
        "6. Which run is which: every model was trained three times (Seeds 42, 123, 777); comparisons are matched by seed,",
        "   and the headline results are the mean of the three runs. The original detailed analyses used Seed 777.",
        "   92.58% (Seed 42), 91.83% (Seed 123) and 93.50% (Seed 777) are different runs, each traceable to its own matrix.",
    ]
    for i, t in enumerate(text, 1):
        ex.cell(i, 1, t)
    ex["A1"].font = bold
    ex.column_dimensions["A"].width = 120

    cmws = wb.create_sheet("Confusion matrices")
    r0 = 1
    for run in order:
        m, s = split_name(run)
        z = runs[run]
        cm = confusion_matrix(z["targets"], z["preds"], labels=range(8))
        cmws.cell(r0, 1, f"{PRETTY.get(m, m)} - Seed {s}  (rows = true class, columns = predicted class)").font = bold
        for j, n in enumerate(SHORT, 2):
            cmws.cell(r0 + 1, j, n).font = bold
        cmws.cell(r0 + 1, 10, "Row total").font = bold
        for i in range(8):
            cmws.cell(r0 + 2 + i, 1, SHORT[i]).font = bold
            for j in range(8):
                c = cmws.cell(r0 + 2 + i, j + 2, int(cm[i, j]))
                if i == j:
                    c.fill = fill
            cmws.cell(r0 + 2 + i, 10, f"=SUM(B{r0 + 2 + i}:I{r0 + 2 + i})")
        tr = r0 + 10
        cmws.cell(tr, 1, "Diagonal (correct)").font = bold
        cmws.cell(tr, 2, "=" + "+".join(f"{chr(66 + i)}{r0 + 2 + i}" for i in range(8)))
        cmws.cell(tr, 4, "Accuracy %").font = bold
        cmws.cell(tr, 5, f"=B{tr}/SUM(J{r0 + 2}:J{r0 + 9})*100")
        cmws.cell(tr, 5).number_format = "0.00"
        r0 += 13
    wb.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--sir-xlsx")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs([a.results] + a.extra)

    summary = {"runs": {}, "per_class": {}, "confusion_matrices": {}}
    for run, z in runs.items():
        y, p, pr = z["targets"], z["preds"], z["probs"]
        o = overall(y, p, pr)
        IX = boot_idx(len(y), 2000, 0)
        acc_b = (p[IX] == y[IX]).mean(1) * 100
        f1_b = macro_f1_batch(y, p, IX)
        o["accuracy_ci95"] = [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))]
        o["macro_f1_ci95"] = [float(np.percentile(f1_b, 2.5)), float(np.percentile(f1_b, 97.5))]
        corr = p == y
        o["err_auroc_msp"] = float(roc_auc_score((~corr).astype(int), 1 - pr.max(1)))
        ent = -(pr * np.log(np.clip(pr, 1e-12, 1))).sum(1)
        o["err_auroc_entropy"] = float(roc_auc_score((~corr).astype(int), ent))
        if "uncertainties" in z:
            o["uncertainty"] = uncertainty_block(z["uncertainties"], corr)
        o["selective_msp"] = selective(1 - pr.max(1), corr)
        cm = confusion_matrix(y, p, labels=range(8))
        summary["runs"][run] = o
        summary["confusion_matrices"][run] = cm.tolist()
        summary["per_class"][run] = {k: np.asarray(v).round(4).tolist() for k, v in per_class(cm).items()}

    # three-seed summaries (sample SD) for every model that has 42/123/777
    agg = {}
    models = sorted({split_name(r)[0] for r in runs})
    for m in models:
        seeds = [s for s in (42, 123, 777) if f"{m}_seed{s}" in runs]
        if len(seeds) < 2:
            continue
        rr = [summary["runs"][f"{m}_seed{s}"] for s in seeds]
        agg[m] = {"seeds": seeds}
        for key in ("accuracy", "macro_f1", "macro_auc", "ece", "brier", "macro_precision", "err_auroc_msp"):
            v = np.array([r[key] for r in rr])
            agg[m][key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": v.tolist()}
        if all("uncertainty" in r for r in rr):
            for key in ("error_auroc", "cohens_d", "aurc", "u_correct_mean", "u_incorrect_mean"):
                v = np.array([r["uncertainty"][key] for r in rr])
                agg[m][key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": v.tolist()}
            sel = np.array([[x["accuracy"] for x in r["uncertainty"]["selective"]] for r in rr])
            agg[m]["selective_accuracy"] = {"coverages": COVERAGES, "mean": sel.mean(0).tolist(),
                                            "sd": sel.std(0, ddof=1).tolist(), "values": sel.tolist()}
    summary["three_seed"] = agg

    # seed-matched McNemar (exact binomial) and paired bootstrap CIs: MSEA-Net vs comparators
    tests = {}
    for run in runs:
        m, s = split_name(run)
        ref = f"msea_net_seed{s}"
        if m == "msea_net" or ref not in runs:
            continue
        a_, b_ = runs[ref], runs[run]
        assert np.array_equal(a_["targets"], b_["targets"])
        y = a_["targets"]
        ca, cb = a_["preds"] == y, b_["preds"] == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_exact = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
        IX = boot_idx(len(y), 2000, 1)
        diffs = (ca[IX].mean(1) - cb[IX].mean(1)) * 100
        f1d = macro_f1_batch(y, a_["preds"], IX) - macro_f1_batch(y, b_["preds"], IX)
        tests[f"{ref}_vs_{run}"] = {"seed": s, "msea_only_correct": n01, "comparator_only_correct": n10,
                                    "mcnemar_exact_p": p_exact,
                                    "acc_diff_pp": float((ca.mean() - cb.mean()) * 100),
                                    "acc_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                    "f1_diff_pp": float((f1_score(y, a_["preds"], average="macro") - f1_score(y, b_["preds"], average="macro")) * 100),
                                    "f1_diff_ci95": [float(np.percentile(f1d, 2.5)), float(np.percentile(f1d, 97.5))]}
    summary["paired_tests"] = tests

    with open(os.path.join(a.out, "kvasir_reanalysis.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # flat per-class CSV (supervisor's columns) for every run
    import csv
    with open(os.path.join(a.out, "per_class_metrics_all_runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "class", "TP", "FN", "FP", "TN", "ACC_ovr", "TPR", "FNR", "FPR", "TNR", "PREC", "F1", "ERROR"])
        for run in sorted(runs):
            pc = summary["per_class"][run]
            for k in range(8):
                w.writerow([run, SHORT[k]] + [pc[c][k] for c in ("TP", "FN", "FP", "TN", "ACC", "TPR", "FNR", "FPR", "TNR", "PREC", "F1", "ERROR")])

    if a.sir_xlsx:
        order = ["msea_net_seed777", "vgg16_seed42", "msea_net_seed42", "msea_net_seed123", "resnet50_seed42",
                 "densenet121_seed42", "inception_v3_seed42", "mobilenetv3_seed42", "efficientnetv2s_plain_seed42"]
        order += sorted(r for r in runs if r not in order)
        write_excel(a.sir_xlsx, os.path.join(a.out, "sir_test_completed.xlsx"), runs, [r for r in order if r in runs])

    # console digest
    print(f"{'run':38s} {'acc':>6s} {'F1':>6s} {'AUC':>6s} {'ECE':>7s} {'Brier':>7s}")
    for run in sorted(runs):
        o = summary["runs"][run]
        print(f"{run:38s} {o['accuracy']:6.2f} {o['macro_f1']:6.2f} {o['macro_auc']:6.2f} {o['ece']:7.4f} {o['brier']:7.4f}")
    for k, t in tests.items():
        print(f"{k:60s} b={t['msea_only_correct']:3d} c={t['comparator_only_correct']:3d} p={t['mcnemar_exact_p']:.4f} "
              f"dAcc={t['acc_diff_pp']:+.2f} CI={t['acc_diff_ci95'][0]:+.2f},{t['acc_diff_ci95'][1]:+.2f}")
    for m, g in agg.items():
        line = f"{m:24s} acc {g['accuracy']['mean']:.2f}±{g['accuracy']['sd']:.2f}  brier {g['brier']['mean']:.4f}±{g['brier']['sd']:.4f}"
        if "error_auroc" in g:
            line += f"  errAUROC {g['error_auroc']['mean']:.4f}±{g['error_auroc']['sd']:.4f}  sel70 {g['selective_accuracy']['mean'][6]:.2f}"
        print(line)


if __name__ == "__main__":
    main()
