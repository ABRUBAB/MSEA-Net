"""Faster variant of external_eval.py for large external sets: every image is decoded and resized
once with the identical torchvision transform (Resize((224, 224)) on the PIL image), cached as
uint8, and converted to the normalised tensor on the GPU (ToTensor = /255, ImageNet mean/SD).
Outputs are identical in content and format to external_eval.py."""
import argparse
import csv
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(path):
    import torchvision.transforms as T
    from PIL import Image
    return np.asarray(T.Resize((224, 224))(Image.open(path).convert("RGB")), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import torch
    import msea_legacy as L

    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))
    ids = np.array([int(r["id"]) for r in rows])
    label4 = np.array([int(r["label4"]) for r in rows])
    cache = os.path.join(a.out, "image_cache_224_uint8.npy")
    if os.path.exists(cache):
        imgs = np.load(cache)
    else:
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex:
            imgs = np.stack(list(ex.map(_load, [os.path.join(a.root, r["relpath"]) for r in rows], chunksize=64)))
        np.save(cache, imgs)
        print(f"cached {imgs.shape} in {time.time() - t0:.0f}s", flush=True)
    device = L.setup_device()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for ck in sorted(p for d in a.ckpt for p in glob.glob(os.path.join(d, "*_best.pth"))):
        run = os.path.basename(ck)[: -len("_best.pth")]
        dst = os.path.join(a.out, "preds", f"{run}_gastroendonet.npz")
        if os.path.exists(dst):
            continue
        state = torch.load(ck, map_location="cpu", weights_only=False)
        sd = state["model_state_dict"]
        if sd[[k for k in sd if k.endswith(".weight")][-1]].shape[0] != 8:
            continue
        t0 = time.time()
        model = L.create_model(state["model_type"], 8, pretrained=False)
        model.load_state_dict(sd)
        model = model.to(device).eval()
        probs, unc, feats = [], [], []
        with torch.no_grad():
            for s in range(0, len(imgs), 128):
                x = torch.from_numpy(imgs[s:s + 128]).to(device).permute(0, 3, 1, 2).float().div(255)
                x = (x - mean) / std
                with L.autocast(dtype=torch.float16):
                    out = model(x)
                probs.append(out["probs"].float().cpu().numpy())
                feats.append(out["features"].float().cpu().numpy())
                if "uncertainty" in out:
                    unc.append(out["uncertainty"].float().cpu().numpy())
        res = {"probs": np.concatenate(probs), "features": np.concatenate(feats)}
        if unc:
            res["uncertainties"] = np.concatenate(unc)
        np.savez_compressed(dst, ids=ids, label4=label4, **res)
        print(f"{run}: done in {time.time() - t0:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("EXTERNAL EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
