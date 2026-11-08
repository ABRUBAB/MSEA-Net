"""Check that the public msea_net package reproduces the notebook pipeline exactly:
identical ordered file list and train/val/test split, identical evaluation and (seeded) training
transforms, and identical model outputs from a saved checkpoint."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True, help="msea_net_seed777_best.pth")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from PIL import Image
    import msea_legacy as L
    from reproduce_kvasir import find_kvasir_root
    from msea_net.data import dataset as P
    from msea_net.models.msea_net import MSEANet

    root = find_kvasir_root(a.data)
    lp, ll, _ = L.load_dataset_paths(root)
    ltr, lva, lte = L.create_data_splits(lp, ll, seed=42)
    ps = P.scan_kvasir_directory(root)
    pp, pl = [s[0] for s in ps], [s[1] for s in ps]
    ptr, pva, pte = P.create_data_splits(pl, split_seed=42)
    rep = {"same_file_order": [os.path.normpath(x) for x in lp] == [os.path.normpath(x) for x in pp],
           "same_labels": list(ll) == list(pl),
           "same_train": np.array_equal(ltr, ptr), "same_val": np.array_equal(lva, pva), "same_test": np.array_equal(lte, pte)}

    img = Image.open(lp[lte[0]]).convert("RGB")
    rep["eval_transform_identical"] = bool(torch.equal(L.get_transforms(False)(img), P.get_transforms(224, False)(img)))
    torch.manual_seed(0); t1 = L.get_transforms(True)(img)
    torch.manual_seed(0); t2 = P.get_transforms(224, True)(img)
    rep["train_transform_identical_same_seed"] = bool(torch.equal(t1, t2))

    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    m1 = L.create_model("msea_net", 8, pretrained=False); m1.load_state_dict(st); m1.eval()
    m2 = MSEANet(num_classes=8, pretrained=False); m2.load_state_dict(st, strict=True); m2.eval()
    x = torch.stack([L.get_transforms(False)(Image.open(lp[i]).convert("RGB")) for i in lte[:16]])
    with torch.no_grad():
        o1, o2 = m1(x), m2(x)
    rep["model_max_abs_prob_diff"] = float((o1["probs"] - o2["probs"]).abs().max())
    rep["model_max_abs_unc_diff"] = float((o1["uncertainty"] - o2["uncertainty"]).abs().max())
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "package_verification.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("VERIFY DONE")


if __name__ == "__main__":
    main()
