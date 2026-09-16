import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MSEA-Net or baseline checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--model", type=str, default="msea_net", help="Model architecture.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to Kvasir-v2 dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--seed", type=int, default=777, help="Dataset split seed.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import numpy as np
        import torch
        from sklearn.metrics import classification_report, confusion_matrix
        from msea_net.config import load_config, find_dataset_path
        from msea_net.data.dataset import get_stratified_loaders
        from msea_net.models.baselines import build_baseline_model
        from msea_net.evaluation.metrics import compute_classification_metrics
        from msea_net.evaluation.calibration import compute_ece, split_conformal_prediction
    except ImportError as e:
        print(f"❌ Missing dependency: {e}\nPlease install repository requirements: pip install -r requirements.txt")
        sys.exit(1)

    config = load_config(args.config)
    data_root = find_dataset_path(args.data_dir or config['data']['data_root'])

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🔍 Evaluating Checkpoint: {args.checkpoint} on {device}")

    # Load DataLoaders
    _, val_loader, test_loader, class_names = get_stratified_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        img_size=config['data']['img_size'],
        seed=args.seed,
    )

    # Instantiate model and load weights
    model = build_baseline_model(args.model, num_classes=config['data']['num_classes'], pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Evaluate on test set
    @torch.no_grad()
    def eval_subset(loader):
        all_probs, all_targets, all_unc = [], [], []
        for images, targets, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = outputs['probs'].cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            if 'uncertainty' in outputs:
                all_unc.append(outputs['uncertainty'].cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        preds = np.argmax(probs, axis=1)
        unc = np.concatenate(all_unc, axis=0) if all_unc else None
        return preds, targets, probs, unc

    preds, targets, probs, unc = eval_subset(test_loader)
    metrics = compute_classification_metrics(targets, preds, probs, num_classes=len(class_names))
    ece = compute_ece(probs, targets)

    print("\n" + "=" * 60)
    print("📊 OVERALL TEST METRICS")
    print("=" * 60)
    print(f"  Accuracy:         {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1:         {metrics['f1_macro'] * 100:.2f}%")
    print(f"  Macro Precision:  {metrics['precision_macro'] * 100:.2f}%")
    print(f"  Macro Recall:     {metrics['recall_macro'] * 100:.2f}%")
    print(f"  Macro AUROC:      {metrics['auc_macro']:.4f}")
    print(f"  Expected Cal. Err (ECE): {ece:.4f}")
    if unc is not None:
        print(f"  Mean Evidential Uncertainty: {np.mean(unc):.4f} (Std: {np.std(unc):.4f})")
    print("=" * 60)

    print("\nDetailed Classification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4))

    # Split Conformal Prediction using Validation set as Calibration
    val_preds, val_targets, val_probs, _ = eval_subset(val_loader)
    scp_res = split_conformal_prediction(val_probs, val_targets, probs, targets, alpha=0.05)
    print("\n🎯 Split Conformal Prediction (Target Coverage = 95.0%):")
    print(f"  Empirical Test Coverage: {scp_res['empirical_coverage']*100:.2f}%")
    print(f"  Mean Prediction Set Size: {scp_res['mean_set_size']:.2f} classes")
    print(f"  Single-Class Certainty:   {scp_res['single_class_pct']*100:.2f}%")


if __name__ == "__main__":
    main()
