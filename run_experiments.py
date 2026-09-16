import os
import sys
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete benchmark suite (Proposed + Baselines + Ablations).")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root directory.")
    parser.add_argument("--output_dir", type=str, default="./msea_net_results", help="Directory to save all results.")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per run.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import numpy as np
        import torch
        from msea_net.config import load_config, find_dataset_path
        from msea_net.data.dataset import get_stratified_loaders
        from msea_net.models.baselines import build_baseline_model
        from msea_net.training.trainer import Trainer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}\nPlease install repository requirements: pip install -r requirements.txt")
        sys.exit(1)

    config = load_config(args.config)
    data_root = find_dataset_path(args.data_dir or config['data']['data_root'])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("🔬 MSEA-Net: Full Reproducible Benchmark Suite")
    print(f"   Dataset:  {data_root}")
    print(f"   Device:   {device}")
    print(f"   Output:   {output_dir}")
    print("==================================================================\n")

    benchmark_records = {}

    # 1. Proposed Model (MSEA-Net) - 3 Seeds
    print("▶ STAGE 1: Training Proposed MSEA-Net (3 Seeds: 42, 123, 777)")
    msea_seeds = config['training'].get('seeds', [42, 123, 777])
    msea_metrics = []

    for seed in msea_seeds:
        train_loader, val_loader, test_loader, _ = get_stratified_loaders(
            data_root=data_root,
            batch_size=args.batch_size,
            img_size=config['data']['img_size'],
            seed=seed,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
        )
        model = build_baseline_model("msea_net", num_classes=config['data']['num_classes'])
        trainer = Trainer(
            model=model,
            device=device,
            num_classes=config['data']['num_classes'],
            lr=config['training']['lr'],
            lr_backbone=config['training']['lr_backbone'],
            weight_decay=config['training']['weight_decay'],
            use_edl=True,
            edl_lambda_max=config['edl']['lambda_max'],
            edl_annealing=config['edl']['annealing'],
            use_amp=config['training']['use_amp'],
            save_dir=str(output_dir),
            model_tag=f"msea_net_seed{seed}",
        )
        res = trainer.fit(train_loader, val_loader, test_loader, epochs=args.epochs)
        msea_metrics.append(res['test_metrics'])
        benchmark_records[f"msea_net_seed{seed}"] = res['test_metrics']

    # 2. Baselines (Seed 42)
    print("\n▶ STAGE 2: Training 6 Comparative Baseline Architectures (Seed 42)")
    baselines = [
        'vgg16',
        'resnet50',
        'densenet121',
        'inception_v3',
        'mobilenetv3',
        'efficientnetv2s_plain',
    ]
    train_loader, val_loader, test_loader, _ = get_stratified_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        img_size=config['data']['img_size'],
        seed=42,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )

    for bname in baselines:
        try:
            model = build_baseline_model(bname, num_classes=config['data']['num_classes'])
            trainer = Trainer(
                model=model,
                device=device,
                num_classes=config['data']['num_classes'],
                lr=config['training']['lr'],
                lr_backbone=config['training']['lr_backbone'],
                weight_decay=config['training']['weight_decay'],
                use_edl=False,
                use_amp=config['training']['use_amp'],
                save_dir=str(output_dir),
                model_tag=f"{bname}_seed42",
            )
            res = trainer.fit(train_loader, val_loader, test_loader, epochs=args.epochs)
            benchmark_records[bname] = res['test_metrics']
        except Exception as e:
            print(f"⚠️ Error training baseline {bname}: {e}")
            benchmark_records[bname] = {'error': str(e)}

    # 3. Ablation Study Variants (Seed 42)
    print("\n▶ STAGE 3: Training 4 Systematic Ablation Variants (Seed 42)")
    ablations = [
        'msea_no_multiscale',
        'msea_no_attention',
        'msea_no_gff',
        'msea_no_edl',
    ]
    for aname in ablations:
        try:
            model = build_baseline_model(aname, num_classes=config['data']['num_classes'])
            use_edl = (aname != 'msea_no_edl')
            trainer = Trainer(
                model=model,
                device=device,
                num_classes=config['data']['num_classes'],
                lr=config['training']['lr'],
                lr_backbone=config['training']['lr_backbone'],
                weight_decay=config['training']['weight_decay'],
                use_edl=use_edl,
                edl_lambda_max=config['edl']['lambda_max'],
                edl_annealing=config['edl']['annealing'],
                use_amp=config['training']['use_amp'],
                save_dir=str(output_dir),
                model_tag=f"{aname}_seed42",
            )
            res = trainer.fit(train_loader, val_loader, test_loader, epochs=args.epochs)
            benchmark_records[aname] = res['test_metrics']
        except Exception as e:
            print(f"⚠️ Error training ablation {aname}: {e}")
            benchmark_records[aname] = {'error': str(e)}

    # Save summary json
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"\n✅ All experiments concluded. Comprehensive metrics written to: {summary_path}")


if __name__ == "__main__":
    main()
