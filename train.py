import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train MSEA-Net or baseline architectures.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--model", type=str, default="msea_net", help="Model name or ablation variant.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing Kvasir-v2 dataset classes.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for classification head.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split and initialization.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints and metrics.")
    parser.add_argument("--device", type=str, default=None, help="Computation device ('cuda' or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from msea_net.config import load_config, find_dataset_path
        from msea_net.data.dataset import get_stratified_loaders
        from msea_net.models.baselines import build_baseline_model
        from msea_net.training.trainer import Trainer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}\nPlease install repository requirements: pip install -r requirements.txt")
        sys.exit(1)

    config = load_config(args.config)

    # Resolve settings
    data_root = find_dataset_path(args.data_dir or config['data']['data_root'])
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['data']['batch_size']
    lr = args.lr or config['training']['lr']
    lr_backbone = config['training']['lr_backbone']
    weight_decay = config['training']['weight_decay']
    output_dir = args.output_dir or config['output']['save_dir']
    num_classes = config['data']['num_classes']
    img_size = config['data']['img_size']

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧 Training Configuration:")
    print(f"   Model:        {args.model}")
    print(f"   Dataset:      {data_root}")
    print(f"   Device:       {device}")
    print(f"   Seed:         {args.seed}")
    print(f"   Epochs:       {epochs}")
    print(f"   Batch Size:   {batch_size}")
    print(f"   Output Dir:   {output_dir}\n")

    # DataLoaders
    train_loader, val_loader, test_loader, class_names = get_stratified_loaders(
        data_root=data_root,
        batch_size=batch_size,
        img_size=img_size,
        seed=args.seed,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )

    # Model
    model = build_baseline_model(args.model, num_classes=num_classes, pretrained=config['model']['pretrained'])
    use_edl = ('edl' not in args.model) and ('baseline' not in args.model) and (args.model in ['msea_net', 'msea_no_multiscale', 'msea_no_attention', 'msea_no_gff'])

    # Trainer
    model_tag = f"{args.model}_seed{args.seed}"
    trainer = Trainer(
        model=model,
        device=device,
        num_classes=num_classes,
        lr=lr,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
        use_edl=use_edl,
        edl_lambda_max=config['edl']['lambda_max'],
        edl_annealing=config['edl']['annealing'],
        use_amp=config['training']['use_amp'],
        save_dir=output_dir,
        model_tag=model_tag,
    )

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        patience=config['training']['patience'],
    )

    print(f"\n✅ Training workflow complete for {model_tag}. Checkpoint saved to {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
