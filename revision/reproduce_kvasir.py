"""Rebuild the notebook's Kvasir-v2 split and re-run the 13 saved checkpoints.

Checks that the revision environment reproduces the published predictions before any new
number is produced, and saves validation-set predictions (needed for validation-locked
uncertainty thresholds and validation-based seed selection).

Outputs (in --out):
  kvasir_split_manifest.csv              index, relative path, label, class name, split
  preds/<run>_kvasir_{val,test}.npz      probs, preds, targets, features, uncertainties (EDL)
  reproduction_report.json               agreement with the notebook's saved test predictions
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
           'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']


def find_kvasir_root(base):
    """Return the folder whose sub-folders are the eight Kvasir-v2 classes."""
    for dirpath, dirnames, _ in os.walk(base):
        if set(CLASSES).issubset(set(dirnames)):
            return dirpath
    raise FileNotFoundError(f"No folder with the 8 Kvasir-v2 class folders under {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import torch
    import msea_legacy as L
    from sklearn.metrics import accuracy_score, f1_score

    L.Config.NUM_WORKERS = a.workers
    device = L.setup_device()
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)

    root = find_kvasir_root(a.data)
    paths, labels, classes = L.load_dataset_paths(root)
    assert classes == CLASSES, classes
    tr, va, te = L.create_data_splits(paths, labels, seed=42)
    split_of = {int(i): s for s, idx in (("train", tr), ("val", va), ("test", te)) for i in idx}
    with open(os.path.join(a.out, "kvasir_split_manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "relpath", "label", "class", "split", "order_in_split"])
        order = {}
        for s, idx in (("train", tr), ("val", va), ("test", te)):
            for k, i in enumerate(idx):
                order[int(i)] = k
        for i, (p, y) in enumerate(zip(paths, labels)):
            w.writerow([i, os.path.relpath(p, root).replace("\\", "/"), y, classes[y], split_of[i], order[i]])

    _, val_loader, test_loader, _ = L.create_dataloaders(paths, labels, tr, va, te)

    report = {"kvasir_root": root, "n_images": len(paths),
              "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "runs": {}}
    for ck in sorted(glob.glob(os.path.join(a.ckpt, "*_best.pth"))):
        run = os.path.basename(ck)[: -len("_best.pth")]
        t0 = time.time()
        state = torch.load(ck, map_location="cpu", weights_only=False)
        mtype = state["model_type"]
        model = L.create_model(mtype, 8, pretrained=False)
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device).eval()
        use_edl = mtype.startswith("msea") and "no_edl" not in mtype
        entry = {"model_type": mtype, "seed": state.get("seed"), "best_epoch": state.get("best_epoch"),
                 "best_val_f1_saved": float(state.get("best_val_f1", float("nan")))}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            m = L.evaluate(model, loader, device, use_edl)
            save = {k: m[k] for k in ("probs", "preds", "targets", "features")}
            if m["uncertainties"] is not None:
                save["uncertainties"] = m["uncertainties"]
            np.savez_compressed(os.path.join(a.out, "preds", f"{run}_kvasir_{split}.npz"), **save)
            entry[f"{split}_accuracy"] = float(accuracy_score(m["targets"], m["preds"]))
            entry[f"{split}_f1_macro"] = float(f1_score(m["targets"], m["preds"], average="macro"))
            entry[f"{split}_brier"] = float(m["brier"])
            entry[f"{split}_ece"] = float(m["ece"])
            if split == "test":
                stored_path = os.path.join(a.ckpt, f"{run}_results.npz")
                if os.path.exists(stored_path):
                    st = np.load(stored_path)
                    entry["targets_identical"] = bool(np.array_equal(st["targets"], m["targets"]))
                    entry["pred_agreement"] = float((st["preds"] == m["preds"]).mean())
                    entry["n_pred_diff"] = int((st["preds"] != m["preds"]).sum())
                    entry["max_abs_prob_diff"] = float(np.abs(st["probs"] - m["probs"]).max())
                    entry["stored_accuracy"] = float((st["preds"] == st["targets"]).mean())
                    if "uncertainties" in st.files and m["uncertainties"] is not None:
                        entry["max_abs_unc_diff"] = float(np.abs(st["uncertainties"] - m["uncertainties"]).max())
        entry["seconds"] = round(time.time() - t0, 1)
        report["runs"][run] = entry
        print(json.dumps({run: entry}), flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(a.out, "reproduction_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("REPRODUCTION DONE", flush=True)


if __name__ == "__main__":
    main()
