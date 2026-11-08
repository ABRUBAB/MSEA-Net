"""Figure of the exact preprocessing pipeline (phase1 get_transforms) applied step by step.

Training branch (stochastic, re-drawn for every image in every epoch):
  Resize 256x256 -> RandomCrop 224 -> RandomHorizontalFlip(0.5) -> RandomVerticalFlip(0.5)
  -> RandomRotation(+-15 deg) -> ColorJitter(0.2, 0.2, 0.2, 0.1) -> ToTensor -> Normalize(ImageNet)
Validation / test / external branch (deterministic): Resize 224x224 -> ToTensor -> Normalize.

To make each step visible, one set of random parameters is drawn per example (seeded) and the
torchvision functional operations are applied in the same order as the composed transform.
(ColorJitter applies its four adjustments in a random order; the drawn order is shown.)
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def show_norm(t):
    """Display a normalised tensor: map z-scores in [-2.5, 2.5] linearly to [0, 1]."""
    a = t.permute(1, 2, 0).numpy()
    return np.clip((a + 2.5) / 5.0, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvasir-manifest", required=True)
    ap.add_argument("--kvasir-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from reproduce_kvasir import find_kvasir_root

    root = find_kvasir_root(a.kvasir_root)
    with open(a.kvasir_manifest) as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "train"]
    picks = []
    for cls in ("esophagitis", "dyed-lifted-polyps", "polyps"):
        picks.append(next(r for r in rows if r["class"] == cls))

    torch.manual_seed(7)                      # ColorJitter.get_params uses the global generator
    g = torch.Generator().manual_seed(7)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8})
    fig = plt.figure(figsize=(15.5, 10.4), dpi=300)
    gs = fig.add_gridspec(len(picks) * 2, 7, hspace=0.42, wspace=0.06)
    titles = ["(a) Original", "(b) Resize 256×256", "(c) Random crop 224×224", "(d) Random flips",
              "(e) Rotation ±15°", "(f) Colour jitter", "(g) ImageNet normalisation"]
    for i, r in enumerate(picks):
        img = Image.open(os.path.join(root, r["relpath"])).convert("RGB")
        w0, h0 = img.size
        im256 = TF.resize(img, [256, 256])                      # T.Resize((256, 256)), bilinear + antialias
        top, left = [int(torch.randint(0, 33, (1,), generator=g)) for _ in range(2)]
        crop = TF.crop(im256, top, left, 224, 224)               # T.RandomCrop(224)
        hf, vf = [bool(torch.rand(1, generator=g) < 0.5) for _ in range(2)]
        if i == 0:
            hf = True                                            # show at least one flip
        fl = TF.hflip(crop) if hf else crop
        fl = TF.vflip(fl) if vf else fl
        ang = float(torch.empty(1).uniform_(-15, 15, generator=g))
        rot = TF.rotate(fl, ang)                                 # nearest, fill 0 (black corners)
        cj = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
        fn_idx, b, c, s, h = T.ColorJitter.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)
        jit = rot
        names = []
        for fid in fn_idx:
            if fid == 0:
                jit = TF.adjust_brightness(jit, b); names.append(f"b={b:.2f}")
            elif fid == 1:
                jit = TF.adjust_contrast(jit, c); names.append(f"c={c:.2f}")
            elif fid == 2:
                jit = TF.adjust_saturation(jit, s); names.append(f"s={s:.2f}")
            else:
                jit = TF.adjust_hue(jit, h); names.append(f"h={h:+.2f}")
        norm = TF.normalize(TF.to_tensor(jit), MEAN, STD)
        tiles = [img, im256, crop, fl, rot, jit, show_norm(norm)]
        notes = [f"{w0}×{h0} px", "aspect ratio not kept", f"offset (x={left}, y={top}) px",
                 f"H-flip: {'yes' if hf else 'no'}, V-flip: {'yes' if vf else 'no'}", f"angle = {ang:+.1f}°",
                 ", ".join(names), "z-scores shown as [−2.5, 2.5]"]
        for j, (t, n) in enumerate(zip(tiles, notes)):
            ax = fig.add_subplot(gs[2 * i, j])
            ax.imshow(t)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=9, fontweight="bold")
            ax.set_xlabel(n, fontsize=7)
            if j == 0:
                ax.set_ylabel(r["class"].replace("-", " ").capitalize() + "\ntraining (random)", fontsize=8, fontweight="bold")
        ev = TF.resize(img, [224, 224])
        evn = TF.normalize(TF.to_tensor(ev), MEAN, STD)
        for j, (t, n, ttl) in enumerate([(img, f"{w0}×{h0} px", "Original"),
                                          (ev, "whole field of view kept", "Resize 224×224 (no crop)"),
                                          (show_norm(evn), "same mean/SD as training", "ImageNet normalisation")]):
            ax = fig.add_subplot(gs[2 * i + 1, [0, 1, 6][j]])
            ax.imshow(t)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ttl, fontsize=8)
            ax.set_xlabel(n, fontsize=7)
            if j == 0:
                ax.set_ylabel("Val / test / external\n(deterministic)", fontsize=8, fontweight="bold")
        ax = fig.add_subplot(gs[2 * i + 1, 2:6])
        ax.axis("off")
        ax.text(0.5, 0.5, "no cropping, flipping, rotation or colour jitter\n(applied identically to Kvasir-v2 validation/test and external images)",
                ha="center", va="center", fontsize=8, style="italic")
    fig.savefig(a.out, bbox_inches="tight", dpi=300)
    print("saved", a.out)


if __name__ == "__main__":
    main()
