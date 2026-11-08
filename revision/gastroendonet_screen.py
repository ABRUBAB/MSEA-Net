"""Integrity screen of GastroEndoNet (original images only) before external validation.

1. Inventory: class folders, counts, image sizes and formats.
2. Exact duplicates: SHA-1 of the file bytes and of the decoded RGB pixels.
3. Near-duplicates inside GastroEndoNet: 64-bit DCT perceptual hash (pHash) and cosine
   similarity of ImageNet-pretrained EfficientNetV2-S embeddings; linked images form
   near-duplicate groups (connected components).
4. Overlap with Kvasir-v2 (all 8,000 images): same criteria; overlapping images are excluded
   from the external test so that no Kvasir training image can re-appear as an "external" one.
5. Grouped, class-stratified 70/15/15 split (StratifiedGroupKFold, 20 folds, seed 42) for the
   in-domain GastroEndoNet training protocol, so near-duplicates never cross subsets.

The class mapping to the Kvasir-v2 label space is fixed here, before any model is evaluated.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEN_CLASSES = ["GERD", "GERD Normal", "Polyp", "Polyp Normal"]
_NORM = {"gerd": "GERD", "gerdnormal": "GERD Normal", "normalgerd": "GERD Normal",
         "polyp": "Polyp", "polyps": "Polyp", "polypnormal": "Polyp Normal",
         "polypsnormal": "Polyp Normal", "normalpolyp": "Polyp Normal"}
KVASIR = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
          'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
# Pre-specified mapping (fixed before evaluation). Polyp Normal -> normal-cecum is a proxy:
# Kvasir-v2 has no generic "normal colon" class; normal-cecum is its only normal colonic class.
TO_KVASIR = {"GERD": 2, "GERD Normal": 5, "Polyp": 6, "Polyp Normal": 3}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Duplicate criteria, chosen after visual inspection of contact sheets in cosine bins
# (first pass): cosine >= 0.99 pairs are the same image (re-saved, cropped or annotated copies);
# 0.95-0.99 pairs are mostly consecutive frames of the same scene.  pHash is reported only:
# the octagonal black field-of-view border shared by most endoscopy frames makes it link
# unrelated images (median nearest-neighbour distance 10/64 bits; Hamming 4-6 between visibly
# different GastroEndoNet and Kvasir-v2 scenes).
COS_DUP = 0.99        # same image -> keep one copy; copies with different labels = label conflict
COS_GROUP = 0.95      # same scene -> same group, never split across train/val/test
COS_CROSS = 0.99      # GastroEndoNet vs Kvasir-v2 -> overlap (excluded)


def scan(base):
    rows, skipped = [], {}
    for dirpath, _, files in os.walk(base):
        imgs = sorted(f for f in files if os.path.splitext(f)[1].lower() in IMG_EXT)
        if not imgs:
            continue
        cls = _NORM.get(re.sub(r"[^a-z]", "", os.path.basename(dirpath).lower()))
        if cls is None:
            skipped[dirpath] = len(imgs)
            continue
        rows += [(os.path.join(dirpath, f), cls) for f in imgs]
    return rows, skipped


def phash64(img):
    from scipy.fft import dctn
    a = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    d = dctn(a, norm="ortho")[:8, :8].flatten()
    return np.packbits(d > np.median(d[1:])).view(">u8")[0]


def hamming(a, b):
    """Pairwise Hamming distances between uint64 hash vectors a (n) and b (m)."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int16)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1).astype(np.int16)


def file_hashes(path):
    raw = open(path, "rb").read()
    img = Image.open(path)
    img.load()
    rgb = img.convert("RGB")
    return (hashlib.sha1(raw).hexdigest(), hashlib.sha1(rgb.tobytes()).hexdigest(),
            rgb.size, img.format, img.mode, phash64(rgb))


def embed(paths, device, workers):
    import torch
    import timm
    import msea_legacy as L
    from torch.utils.data import DataLoader
    model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=0).to(device).eval()
    ds = L.KvasirDataset(list(paths), [0] * len(paths), L.get_transforms(False))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            f = model(x.to(device))
            out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


def contact_sheet(pairs, path, title):
    from PIL import ImageDraw
    th, pad, cols = 150, 6, 4
    n = len(pairs)
    rows_ = (n + cols - 1) // cols
    W, H = cols * (2 * th + 3 * pad), rows_ * (th + 26) + 30
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((6, 6), title, fill="black")
    for k, (pa, pb, label) in enumerate(pairs):
        r, c = divmod(k, cols)
        x0, y0 = c * (2 * th + 3 * pad), 30 + r * (th + 26)
        for j, p in enumerate((pa, pb)):
            im = Image.open(p).convert("RGB")
            im.thumbnail((th, th))
            sheet.paste(im, (x0 + pad + j * (th + pad), y0))
        dr.text((x0 + pad, y0 + th + 4), label, fill="black")
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gastro", required=True)
    ap.add_argument("--kvasir-manifest", required=True)
    ap.add_argument("--kvasir-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="reuse embeddings/hashes from screen_features.npz")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, skipped = scan(a.gastro)
    print("GastroEndoNet images:", len(rows), "| skipped folders:", skipped, flush=True)
    meta = [file_hashes(p) for p, _ in rows]
    g_paths = [p for p, _ in rows]
    g_cls = [c for _, c in rows]

    from reproduce_kvasir import find_kvasir_root
    kroot = find_kvasir_root(a.kvasir_root)
    with open(a.kvasir_manifest) as fh:
        kv = list(csv.DictReader(fh))
    k_paths = [os.path.join(kroot, r["relpath"]) for r in kv]
    feat_path = os.path.join(a.out, "screen_features.npz")
    if a.reuse and os.path.exists(feat_path):
        z = np.load(feat_path)
        g_emb, k_emb = z["g_emb"].astype(np.float32), z["k_emb"].astype(np.float32)
        g_ph, k_ph = z["g_ph"], z["k_ph"]
    else:
        g_ph = np.array([m[5] for m in meta], dtype=np.uint64)
        k_ph = np.array([phash64(Image.open(p).convert("RGB")) for p in k_paths], dtype=np.uint64)
        g_emb = embed(g_paths, device, a.workers)
        k_emb = embed(k_paths, device, a.workers)
        np.savez_compressed(feat_path, g_emb=g_emb.astype(np.float16), k_emb=k_emb.astype(np.float16), g_ph=g_ph, k_ph=k_ph)
    print("embeddings", g_emb.shape, k_emb.shape, flush=True)

    n = len(rows)
    ham_gg = hamming(g_ph, g_ph)
    cos_gg = g_emb @ g_emb.T
    np.fill_diagonal(ham_gg, 99)
    np.fill_diagonal(cos_gg, -1)

    def components(adj):
        dsu = DSU(n)
        for i, j in zip(*np.nonzero(np.triu(adj, 1))):
            dsu.union(int(i), int(j))
        roots = [dsu.find(i) for i in range(n)]
        ids = {r: k for k, r in enumerate(sorted(set(roots)))}
        return [ids[r] for r in roots]

    # --- duplicates (same image): exact pixel hash or cosine >= COS_DUP
    same_pix = np.zeros((n, n), dtype=bool)
    by_pix = {}
    for i, m in enumerate(meta):
        by_pix.setdefault(m[1], []).append(i)
    for idx in by_pix.values():
        for i in idx:
            for j in idx:
                if i != j:
                    same_pix[i, j] = True
    dup = components((cos_gg >= COS_DUP) | same_pix)
    dup_members = {}
    for i, d in enumerate(dup):
        dup_members.setdefault(d, []).append(i)
    conflict = [len({g_cls[j] for j in dup_members[dup[i]]}) > 1 for i in range(n)]
    representative = [min(dup_members[dup[i]]) == i for i in range(n)]
    exact_of = ["" if representative[i] else str(min(dup_members[dup[i]])) for i in range(n)]
    # --- same-scene groups (for the split): cosine >= COS_GROUP
    group = components(cos_gg >= COS_GROUP)
    gsize = np.bincount(group)
    conflict_groups = {group[i] for i in range(n) if conflict[i]}
    # --- overlap with Kvasir-v2
    ham_gk = np.concatenate([hamming(g_ph[s:s + 512], k_ph) for s in range(0, n, 512)])
    cos_gk = g_emb @ k_emb.T
    nn_ham, nn_ham_j = ham_gk.min(1), ham_gk.argmin(1)
    nn_cos, nn_cos_j = cos_gk.max(1), cos_gk.argmax(1)
    overlap = nn_cos >= COS_CROSS

    # --- grouped stratified split (protocol B)
    from sklearn.model_selection import StratifiedGroupKFold
    y4 = np.array([GEN_CLASSES.index(c) for c in g_cls])
    usable = np.array([representative[i] and not conflict[i] and not overlap[i] for i in range(n)])
    split = np.array(["excluded"] * n, dtype=object)
    idx = np.nonzero(usable)[0]
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    for f, (_, te) in enumerate(sgkf.split(idx, y4[idx], np.array(group)[idx])):
        split[idx[te]] = "test" if f < 3 else ("val" if f < 6 else "train")

    with open(os.path.join(a.out, "gastroendonet_manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "relpath", "class", "label4", "kvasir_label", "width", "height", "format",
                    "file_sha1", "pixel_sha1", "duplicate_of", "group_id", "group_size", "label_conflict",
                    "nn_kvasir_cos", "nn_kvasir_cos_relpath", "nn_kvasir_cos_split",
                    "nn_kvasir_ham", "nn_kvasir_ham_relpath", "kvasir_overlap", "keep_external", "split"])
        for i in range(n):
            keep_ext = bool(usable[i])
            w.writerow([i, os.path.relpath(g_paths[i], a.gastro).replace("\\", "/"), g_cls[i], int(y4[i]),
                        TO_KVASIR[g_cls[i]], meta[i][2][0], meta[i][2][1], meta[i][3], meta[i][0], meta[i][1],
                        exact_of[i], group[i], int(gsize[group[i]]), bool(conflict[i]),
                        round(float(nn_cos[i]), 4), kv[nn_cos_j[i]]["relpath"], kv[nn_cos_j[i]]["split"],
                        int(nn_ham[i]), kv[nn_ham_j[i]]["relpath"], bool(overlap[i]), bool(keep_ext), split[i]])

    sizes = {}
    for m in meta:
        sizes[f"{m[2][0]}x{m[2][1]}"] = sizes.get(f"{m[2][0]}x{m[2][1]}", 0) + 1
    rep = {
        "n_images": n, "per_class": {c: g_cls.count(c) for c in GEN_CLASSES},
        "formats": {f: sum(1 for m in meta if m[3] == f) for f in {m[3] for m in meta}},
        "sizes_top": sorted(sizes.items(), key=lambda kv_: -kv_[1])[:10],
        "thresholds": {"COS_DUP": COS_DUP, "COS_GROUP": COS_GROUP, "COS_CROSS": COS_CROSS},
        "exact_pixel_duplicate_copies": sum(len(v) - 1 for v in by_pix.values()),
        "duplicate_copies_removed": int(sum(1 for i in range(n) if not representative[i])),
        "duplicate_clusters": int(sum(1 for v in dup_members.values() if len(v) > 1)),
        "exact_file_duplicates": n - len({m[0] for m in meta}),
        "n_groups": len(set(group)), "n_in_multi_groups": int(sum(gsize[g] > 1 for g in group)),
        "largest_group": int(gsize.max()),
        "label_conflict_clusters": int(sum(1 for v in dup_members.values() if len({g_cls[j] for j in v}) > 1)),
        "label_conflict_images": int(sum(conflict)),
        "kvasir_overlap": int(overlap.sum()),
        "kvasir_overlap_by_class": {c: int(sum(overlap[i] for i in range(n) if g_cls[i] == c)) for c in GEN_CLASSES},
        "kvasir_overlap_in_kvasir_train": int(sum(overlap[i] and kv[nn_cos_j[i]]["split"] == "train" for i in range(n))),
        "nn_cos_quantiles": np.quantile(nn_cos, [0, .5, .9, .99, 1]).round(4).tolist(),
        "nn_ham_quantiles": np.quantile(nn_ham, [0, .01, .1, .5, 1]).tolist(),
        "within_ham_min_quantiles": np.quantile(ham_gg.min(1), [0, .01, .1, .5]).tolist(),
        "within_cos_max_quantiles": np.quantile(cos_gg.max(1), [.5, .9, .99, 1]).round(4).tolist(),
        "keep_external": int(usable.sum()),
        "keep_external_by_class": {c: int(sum(1 for i in range(n) if g_cls[i] == c and usable[i])) for c in GEN_CLASSES},
        "split_counts": {s: {c: int(sum(1 for i in range(n) if split[i] == s and g_cls[i] == c)) for c in GEN_CLASSES} for s in ("train", "val", "test", "excluded")},
    }
    with open(os.path.join(a.out, "screen_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1), flush=True)

    # contact sheets for visual verification of the closest pairs
    top = np.argsort(-nn_cos)[:24]
    contact_sheet([(g_paths[i], k_paths[nn_cos_j[i]], f"cos={nn_cos[i]:.3f} ham={ham_gk[i, nn_cos_j[i]]}") for i in top],
                  os.path.join(a.out, "sheet_cross_top_cos.png"), "GastroEndoNet (left) vs nearest Kvasir-v2 (right), by embedding cosine")
    top = np.argsort(nn_ham)[:24]
    contact_sheet([(g_paths[i], k_paths[nn_ham_j[i]], f"ham={nn_ham[i]} cos={cos_gk[i, nn_ham_j[i]]:.3f}") for i in top],
                  os.path.join(a.out, "sheet_cross_top_ham.png"), "GastroEndoNet (left) vs nearest Kvasir-v2 (right), by pHash distance")
    iu = np.argwhere(np.triu(cos_gg >= COS_GROUP, 1))
    if len(iu):
        sel = iu[np.argsort(-cos_gg[iu[:, 0], iu[:, 1]])][:24]
        contact_sheet([(g_paths[i], g_paths[j], f"{g_cls[i][:10]}|{g_cls[j][:10]} c={cos_gg[i, j]:.3f} h={ham_gg[i, j]}") for i, j in sel],
                      os.path.join(a.out, "sheet_within_top.png"), "Closest GastroEndoNet pairs (near-duplicate candidates)")
    print("SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
