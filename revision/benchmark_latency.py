"""Parameter count, FLOPs and inference latency of MSEA-Net and the baselines, with a fully
reported protocol (the original notebook did not measure latency).

Protocol: eval mode, torch.no_grad, input 1x3x224x224 already on the GPU, 50 warm-up
iterations, 300 timed iterations, torch.cuda.synchronize() around every iteration,
perf_counter timing; fp16 autocast (as used in the notebook) and fp32.  Throughput uses
batch 32.  Preprocessing (JPEG decode + resize + normalise, CPU) is timed separately on 200
Kvasir test images.  FLOPs are counted with torch.utils.flop_counter (1 multiply-add = 2 FLOPs).
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = ["msea_net", "efficientnetv2s_plain", "vgg16", "resnet50", "densenet121", "inception_v3", "mobilenetv3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kvasir-manifest")
    ap.add_argument("--kvasir-root")
    a = ap.parse_args()
    import torch
    import msea_legacy as L
    from torch.utils.flop_counter import FlopCounterMode

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "cpu": platform.processor(), "os": platform.platform(),
           "protocol": __doc__.strip(), "models": {}}
    for m in MODELS:
        model = L.create_model(m, 8, pretrained=False).to(dev).eval()
        n_params = sum(p.numel() for p in model.parameters())
        x1 = torch.randn(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                model(x1)
            flops = fc.get_total_flops()
        out = {"params": n_params, "gflops": flops / 1e9}
        for prec in ("fp16", "fp32"):
            times = []
            with torch.no_grad():
                for it in range(350):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if prec == "fp16":
                        with L.autocast(dtype=torch.float16):
                            model(x1)
                    else:
                        model(x1)
                    torch.cuda.synchronize()
                    if it >= 50:
                        times.append((time.perf_counter() - t0) * 1000)
            out[f"latency_ms_{prec}_mean"] = statistics.mean(times)
            out[f"latency_ms_{prec}_sd"] = statistics.stdev(times)
            out[f"latency_ms_{prec}_median"] = statistics.median(times)
            out[f"fps_{prec}"] = 1000 / statistics.mean(times)
        xb = torch.randn(32, 3, 224, 224, device=dev)
        with torch.no_grad():
            for _ in range(10):
                with L.autocast(dtype=torch.float16):
                    model(xb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                with L.autocast(dtype=torch.float16):
                    model(xb)
            torch.cuda.synchronize()
        out["throughput_img_s_fp16_b32"] = 32 * 50 / (time.perf_counter() - t0)
        res["models"][m] = out
        print(m, json.dumps(out), flush=True)
        del model
        torch.cuda.empty_cache()

    if a.kvasir_manifest:
        import csv
        from PIL import Image
        from reproduce_kvasir import find_kvasir_root
        kroot = find_kvasir_root(a.kvasir_root)
        tf = L.get_transforms(False)
        with open(a.kvasir_manifest) as fh:
            rows = [r for r in csv.DictReader(fh) if r["split"] == "test"][:200]
        ts = []
        for r in rows:
            t0 = time.perf_counter()
            tf(Image.open(os.path.join(kroot, r["relpath"])).convert("RGB"))
            ts.append((time.perf_counter() - t0) * 1000)
        res["preprocess_ms_cpu_mean"] = statistics.mean(ts)
        res["preprocess_ms_cpu_sd"] = statistics.stdev(ts)
    with open(os.path.join(a.out, "latency_benchmark.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("BENCHMARK DONE", flush=True)


if __name__ == "__main__":
    main()
