"""Raw explainability data for the Seed 777 MSEA-Net, computed exactly as in phase5_evaluation_xai.py,
so that Figures 14-16 can be redrawn (ggsci NPG palette) by make_npg_figures.py without changing content.

* Grad-CAM and Grad-CAM++ (pytorch-grad-cam): first test image of each class, target = true class,
  target layer = last block of the EfficientNetV2-S backbone, model output = evidence.
* LIME (lime_image): same images, default quickshift segmentation, hide_color=0, num_samples=200,
  top label; random_seed fixed at 42 (the notebook left it unset).
* UMAP (umap-learn): 256-d fused features of the 1,200 test images saved by the notebook,
  n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42.

Usage: python xai_compute.py --data <dir containing the Kvasir-v2 class folders> --ckpt <checkpoint dir> --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
           'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def find_kvasir_root(base):
    for dirpath, dirnames, _ in os.walk(base):
        if set(CLASSES).issubset(set(dirnames)):
            return dirpath
    raise FileNotFoundError(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    import torch.nn as nn
    from PIL import Image
    import msea_legacy as L

    device = L.setup_device()
    root = find_kvasir_root(a.data)
    paths, labels, classes = L.load_dataset_paths(root)
    assert classes == CLASSES
    _, _, te = L.create_data_splits(paths, labels, seed=42)
    test_paths, test_labels = [paths[i] for i in te], [labels[i] for i in te]
    test_ds = L.KvasirDataset(test_paths, test_labels, L.get_transforms(False))
    first = {c: test_labels.index(c) for c in range(8)}  # phase5: first test image of each class

    state = torch.load(os.path.join(a.ckpt, "msea_net_seed777_best.pth"), map_location="cpu", weights_only=False)
    model = L.create_model("msea_net", 8, pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()

    imgs, cams, campps, preds, uncs = [], [], [], [], []
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, x):
            o = self.model(x)
            return o["evidence"] if "evidence" in o else o["logits"]

    target_layers = [model.backbone.blocks[-1][-1]]
    cam_gc = GradCAM(model=Wrapper(model), target_layers=target_layers)
    cam_gcpp = GradCAMPlusPlus(model=Wrapper(model), target_layers=target_layers)
    for c in range(8):
        x, y = test_ds[first[c]]
        img = np.clip(x.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1).astype(np.float32)
        inp = x.unsqueeze(0).to(device)
        cams.append(cam_gc(input_tensor=inp, targets=[ClassifierOutputTarget(c)])[0])
        campps.append(cam_gcpp(input_tensor=inp, targets=[ClassifierOutputTarget(c)])[0])
        with torch.no_grad():
            o = model(inp)
        preds.append(int(o["probs"].argmax(1).item()))
        uncs.append(float(o["uncertainty"].item()))
        imgs.append(img)
    del cam_gc, cam_gcpp

    from lime import lime_image
    tf = L.get_transforms(False)

    def predict_fn(batch_np):
        batch = torch.stack([tf(Image.fromarray((b * 255).astype(np.uint8))) for b in batch_np]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return model(batch)["probs"].float().cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=42)
    segs, lime_masks, lime_labels, lime_weights = [], [], [], []
    for c in range(8):
        exp = explainer.explain_instance(imgs[c].astype(np.float64), predict_fn, top_labels=1, hide_color=0,
                                         num_samples=200, random_seed=42)
        lab = exp.top_labels[0]
        _, mask = exp.get_image_and_mask(lab, positive_only=False, num_features=10, hide_rest=False)
        segs.append(exp.segments.astype(np.int32))
        lime_masks.append(mask.astype(np.int8))
        lime_labels.append(int(lab))
        w = np.zeros(int(exp.segments.max()) + 1, np.float32)
        for f, v in exp.local_exp[lab]:
            w[f] = v
        lime_weights.append(w)

    import umap
    z = np.load(os.path.join(a.ckpt, "msea_net_seed777_results.npz"))
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, metric="cosine").fit_transform(z["features"])

    maxseg = max(len(w) for w in lime_weights)
    np.savez_compressed(
        os.path.join(a.out, "xai_seed777.npz"),
        images=np.stack(imgs), gradcam=np.stack(cams), gradcampp=np.stack(campps),
        sample_pred=np.array(preds), sample_unc=np.array(uncs), sample_label=np.arange(8),
        lime_segments=np.stack(segs), lime_mask=np.stack(lime_masks), lime_label=np.array(lime_labels),
        lime_weights=np.stack([np.pad(w, (0, maxseg - len(w))) for w in lime_weights]),
        umap=emb.astype(np.float32), targets=z["targets"], preds=z["preds"], uncertainties=z["uncertainties"])
    import importlib.metadata as md
    info = {"test_positions": first, "files": {CLASSES[c]: os.path.relpath(test_paths[first[c]], root) for c in range(8)},
            "versions": {p: md.version(p) for p in ("grad-cam", "lime", "umap-learn", "scikit-image", "torch")},
            "sample_pred": preds, "lime_label": lime_labels}
    json.dump(info, open(os.path.join(a.out, "xai_seed777.json"), "w"), indent=1)
    print("XAI DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
