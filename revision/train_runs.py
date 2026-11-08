"""Train models with the unchanged notebook training code (msea_legacy.train_model).

--dataset kvasir        Kvasir-v2 with the notebook's split (seed 42); used for the extra
                        seeds (123, 777) of the baselines and ablations, so that every model
                        has three matched seeds.
--dataset gastroendonet Protocol B: training from ImageNet initialisation on the grouped,
                        stratified 70/15/15 GastroEndoNet split from gastroendonet_screen.py,
                        4 classes, identical recipe (30 epochs, AdamW, cosine LR, AMP,
                        best-validation-macro-F1 checkpoint).

Runs whose results file already exists are skipped (resumable).
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEN_CLASSES = ["GERD", "GERD Normal", "Polyp", "Polyp Normal"]


def gastro_loaders(L, manifest, root):
    from torch.utils.data import DataLoader
    with open(manifest) as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] in ("train", "val", "test")]
    sets = {}
    for s in ("train", "val", "test"):
        rr = [r for r in rows if r["split"] == s]
        sets[s] = L.KvasirDataset([os.path.join(root, r["relpath"]) for r in rr],
                                  [int(r["label4"]) for r in rr],
                                  L.get_transforms(s == "train"), class_names=GEN_CLASSES)
    C = L.Config
    kw = dict(batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY, persistent_workers=True)
    return (DataLoader(sets["train"], shuffle=True, drop_last=True, **kw),
            DataLoader(sets["val"], shuffle=False, **kw),
            DataLoader(sets["test"], shuffle=False, **kw),
            {s: len(v) for s, v in sets.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kvasir", "gastroendonet"], required=True)
    ap.add_argument("--data", required=True, help="Kvasir root (folder with class folders) or GastroEndoNet root")
    ap.add_argument("--manifest", help="gastroendonet_manifest.csv (gastroendonet only)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init-from", nargs="*", default=[],
                    help="folders with Kvasir-v2 checkpoints <model>_seed<seed>_best.pth: fine-tune from them "
                         "(all layers except the re-initialised final output layer) instead of ImageNet weights")
    a = ap.parse_args()

    import torch
    import msea_legacy as L
    from reproduce_kvasir import find_kvasir_root

    L.Config.NUM_WORKERS = a.workers
    device = L.setup_device()
    os.makedirs(a.out, exist_ok=True)

    current = {"seed": None}
    if a.init_from:
        original_create = L.create_model

        def create_from_kvasir(model_type, num_classes, pretrained=True, **kw):
            """Same architecture, initialised from the matching Kvasir-v2 checkpoint (same model, same seed).
            Tensors whose shape differs (only the final K-way output layer) keep their fresh initialisation."""
            model = original_create(model_type, num_classes, pretrained=False)
            ck = next((os.path.join(d, f"{model_type}_seed{current['seed']}_best.pth") for d in a.init_from
                       if os.path.exists(os.path.join(d, f"{model_type}_seed{current['seed']}_best.pth"))), None)
            if ck is None:
                raise FileNotFoundError(f"no Kvasir checkpoint for {model_type} seed {current['seed']}")
            sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
            msd = model.state_dict()
            load = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
            model.load_state_dict(load, strict=False)
            fresh = [k for k in msd if k not in load]
            print(f"   init from {os.path.basename(ck)}: {len(load)}/{len(msd)} tensors loaded; re-initialised: {fresh}", flush=True)
            return model

        L.create_model = create_from_kvasir      # train_model() looks the factory up at call time

    if a.dataset == "kvasir":
        paths, labels, _ = L.load_dataset_paths(find_kvasir_root(a.data))
        tr, va, te = L.create_data_splits(paths, labels, seed=42)
        train_loader, val_loader, test_loader, _ = L.create_dataloaders(paths, labels, tr, va, te)
        sizes = {"train": len(tr), "val": len(va), "test": len(te)}
    else:
        L.Config.NUM_CLASSES = len(GEN_CLASSES)
        L.Config.CLASS_NAMES = list(GEN_CLASSES)
        train_loader, val_loader, test_loader, sizes = gastro_loaders(L, a.manifest, a.data)
    print("split sizes:", sizes, flush=True)

    for seed in a.seeds:
        for m in a.models:
            if os.path.exists(os.path.join(a.out, f"{m}_seed{seed}_results.npz")):
                print(f"skip {m} seed {seed} (done)", flush=True)
                continue
            t0 = time.time()
            current["seed"] = seed
            r = L.train_model(m, seed, device, train_loader, val_loader, test_loader, save_dir=a.out)
            tm = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in r["test_metrics"].items()
                  if k in ("accuracy", "f1_macro", "f1_weighted", "precision", "recall", "auc_macro", "ece", "brier", "loss")}
            summ = {"model": m, "seed": seed, "dataset": a.dataset, "best_epoch": r["best_epoch"],
                    "best_val_f1": float(r["best_val_f1"]), "training_time_s": r["training_time"],
                    "wall_s": time.time() - t0, "test": tm, "history": r["history"], "split_sizes": sizes}
            with open(os.path.join(a.out, f"{m}_seed{seed}_summary.json"), "w") as fh:
                json.dump(summ, fh, indent=1, default=float)
    print("TRAIN RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
