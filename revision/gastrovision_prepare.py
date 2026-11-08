"""Prepare GastroVision (Jha et al., 2023) as a second, label-compatible external test set.

* Pre-specified mapping (fixed before any model is evaluated): the eight GastroVision categories
  that correspond to Kvasir-v2 classes are mapped to them; all other GastroVision categories are
  'unseen' (label -1) and are used only to test whether uncertainty rises for unfamiliar findings.
* Duplicate / overlap screening exactly as for GastroEndoNet: cosine similarity of ImageNet
  EfficientNetV2-S embeddings; >= 0.99 = same image. GastroVision images matching any Kvasir-v2
  image are excluded (the dataset includes relabelled HyperKvasir images), and within-GastroVision
  duplicates are reduced to one copy.
Output: gastrovision_manifest.csv (columns compatible with external_eval.py: id, relpath, label4
= Kvasir label or -1), gastrovision_report.json.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KVASIR = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
          'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
# normalised GastroVision folder name -> Kvasir-v2 index (pre-specified)
MAP = {"dyedliftedpolyps": 0, "dyedliftedpolyp": 0,
       "dyedresectionmargins": 1, "dyedresectionmargin": 1,
       "esophagitis": 2,
       "cecum": 3,
       "pylorus": 4,
       "gastroesophagealjunctionnormalzline": 5, "normalzline": 5,
       "colonpolyps": 6,
       "ulcerativecolitis": 7}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
COS_DUP = 0.99


def norm(name):
    return re.sub(r"[^a-z]", "", name.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gastrovision", required=True)
    ap.add_argument("--kvasir-features", required=True, help="screen_features.npz from gastroendonet_screen.py (k_emb)")
    ap.add_argument("--kvasir-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    from gastroendonet_screen import embed, DSU

    rows = []
    for dirpath, _, files in os.walk(a.gastrovision):
        imgs = sorted(f for f in files if os.path.splitext(f)[1].lower() in IMG_EXT)
        if imgs:
            cls = os.path.basename(dirpath)
            rows += [(os.path.join(dirpath, f), cls) for f in imgs]
    classes = sorted({c for _, c in rows})
    mapped = {c: MAP.get(norm(c), -1) for c in classes}
    print("GastroVision images:", len(rows), "| folders:", len(classes), flush=True)
    for c in classes:
        print(f"   {c!r:60s} n={sum(1 for _, x in rows if x == c):5d} -> {KVASIR[mapped[c]] if mapped[c] >= 0 else 'unseen'}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [p for p, _ in rows]
    g = embed(paths, device, a.workers)
    k = np.load(a.kvasir_features)["k_emb"].astype(np.float32)
    kv = list(csv.DictReader(open(a.kvasir_manifest)))
    cross = g @ k.T
    nn_cos, nn_j = cross.max(1), cross.argmax(1)
    overlap = nn_cos >= COS_DUP
    n = len(rows)
    gg = g @ g.T
    np.fill_diagonal(gg, -1)
    dsu = DSU(n)
    for i, j in zip(*np.nonzero(np.triu(gg >= COS_DUP, 1))):
        dsu.union(int(i), int(j))
    root = [dsu.find(i) for i in range(n)]
    members = {}
    for i, r in enumerate(root):
        members.setdefault(r, []).append(i)
    conflict = [len({rows[j][1] for j in members[root[i]]}) > 1 for i in range(n)]
    representative = [min(members[root[i]]) == i for i in range(n)]
    keep = [representative[i] and not conflict[i] and not overlap[i] for i in range(n)]

    with open(os.path.join(a.out, "gastrovision_manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "relpath", "class", "label4", "kvasir_class", "nn_kvasir_cos", "nn_kvasir_split",
                    "kvasir_overlap", "duplicate_of", "label_conflict", "keep_external"])
        for i, (p, c) in enumerate(rows):
            w.writerow([i, os.path.relpath(p, a.gastrovision).replace("\\", "/"), c, mapped[c],
                        KVASIR[mapped[c]] if mapped[c] >= 0 else "unseen", round(float(nn_cos[i]), 4),
                        kv[nn_j[i]]["split"], bool(overlap[i]), "" if representative[i] else min(members[root[i]]),
                        bool(conflict[i]), bool(keep[i])])
    rep = {"n_images": n, "n_folders": len(classes),
           "per_class": {c: sum(1 for _, x in rows if x == c) for c in classes},
           "mapping": {c: (KVASIR[v] if v >= 0 else "unseen") for c, v in mapped.items()},
           "kvasir_overlap_total": int(overlap.sum()),
           "kvasir_overlap_by_class": {c: int(sum(overlap[i] for i in range(n) if rows[i][1] == c)) for c in classes},
           "kvasir_overlap_in_kvasir_split": {s: int(sum(1 for i in range(n) if overlap[i] and kv[nn_j[i]]["split"] == s)) for s in ("train", "val", "test")},
           "duplicate_copies_removed": int(sum(1 for i in range(n) if not representative[i])),
           "label_conflict_images": int(sum(conflict)),
           "kept": int(sum(keep)),
           "kept_mapped_by_kvasir_class": {KVASIR[v]: int(sum(1 for i in range(n) if keep[i] and mapped[rows[i][1]] == v)) for v in range(8)},
           "kept_unseen": int(sum(1 for i in range(n) if keep[i] and mapped[rows[i][1]] < 0)),
           "nn_cos_quantiles": np.quantile(nn_cos, [0, .5, .9, .99, 1]).round(4).tolist()}
    json.dump(rep, open(os.path.join(a.out, "gastrovision_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    # contact sheet of the closest GastroVision-Kvasir pairs (visual check of the overlap rule)
    from gastroendonet_screen import contact_sheet
    kroot = os.path.dirname(os.path.dirname(os.path.join(a.kvasir_manifest)))
    top = np.argsort(-nn_cos)[:24]
    from reproduce_kvasir import find_kvasir_root
    kr = find_kvasir_root(os.environ.get("KVASIR_BASE", r"F:\projects\msea-net-revision\data\kvasir-v2"))
    contact_sheet([(paths[i], os.path.join(kr, kv[nn_j[i]]["relpath"]), f"cos={nn_cos[i]:.3f} {rows[i][1][:18]}") for i in top],
                  os.path.join(a.out, "sheet_gastrovision_vs_kvasir.png"), "GastroVision (left) vs nearest Kvasir-v2 (right)")
    print("GASTROVISION PREP DONE", flush=True)


if __name__ == "__main__":
    main()
