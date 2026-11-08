"""Protocol A: frozen external validation of the Kvasir-v2-trained checkpoints on GastroEndoNet.

No retraining, fine-tuning, re-calibration or threshold tuning on GastroEndoNet.  Images get
exactly the notebook's deterministic evaluation preprocessing (resize to 224x224, ImageNet
normalisation) and each checkpoint is run once (AMP fp16, as in the notebook's evaluate()).
The full 8-class Kvasir-v2 output is stored; the pre-specified label mapping
(gastroendonet_screen.TO_KVASIR) is applied later in the analysis.

Output: preds/<run>_gastroendonet.npz with ids, label4, probs (N x 8), uncertainties (EDL runs),
features.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def predict(model, loader, device, L):
    import torch
    probs, unc, feats = [], [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with L.autocast(dtype=torch.float16):
                out = model(x)
            probs.append(out["probs"].float().cpu().numpy())
            feats.append(out["features"].float().cpu().numpy())
            if "uncertainty" in out:
                unc.append(out["uncertainty"].float().cpu().numpy())
    res = {"probs": np.concatenate(probs), "features": np.concatenate(feats)}
    if unc:
        res["uncertainties"] = np.concatenate(unc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gastro", required=True)
    ap.add_argument("--ckpt", required=True, nargs="+", help="checkpoint folders (globbed for *_best.pth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import torch
    import msea_legacy as L
    from torch.utils.data import DataLoader

    device = L.setup_device()
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    with open(a.manifest) as fh:
        rows = list(csv.DictReader(fh))
    ids = np.array([int(r["id"]) for r in rows])
    label4 = np.array([int(r["label4"]) for r in rows])
    paths = [os.path.join(a.gastro, r["relpath"]) for r in rows]
    ds = L.KvasirDataset(paths, label4.tolist(), L.get_transforms(False))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers, pin_memory=True)

    ckpts = sorted(p for d in a.ckpt for p in glob.glob(os.path.join(d, "*_best.pth")))
    for ck in ckpts:
        run = os.path.basename(ck)[: -len("_best.pth")]
        dst = os.path.join(a.out, "preds", f"{run}_gastroendonet.npz")
        if os.path.exists(dst):
            print("skip", run, flush=True)
            continue
        t0 = time.time()
        state = torch.load(ck, map_location="cpu", weights_only=False)
        n_cls = state["model_state_dict"][[k for k in state["model_state_dict"] if k.endswith(".weight")][-1]].shape[0]
        if n_cls != 8:            # skip models trained on GastroEndoNet itself
            continue
        model = L.create_model(state["model_type"], 8, pretrained=False)
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device).eval()
        res = predict(model, loader, device, L)
        np.savez_compressed(dst, ids=ids, label4=label4, **res)
        print(f"{run}: done in {time.time() - t0:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("EXTERNAL EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
