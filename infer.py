import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Single image inference and uncertainty estimation with MSEA-Net.")
    parser.add_argument("--image", type=str, required=True, help="Path to input endoscopy image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth).")
    parser.add_argument("--model", type=str, default="msea_net", help="Model architecture name.")
    parser.add_argument("--explain", type=str, default="gradcam++", choices=["none", "gradcam", "gradcam++"], help="Visual explanation method.")
    parser.add_argument("--output", type=str, default="./prediction_cam.png", help="Path to save explanation overlay.")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('cuda' or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from PIL import Image
        import numpy as np
        import torch
        import torchvision.transforms as T
        from msea_net.models.baselines import build_baseline_model
        from msea_net.data.dataset import DEFAULT_CLASSES, IMAGENET_MEAN, IMAGENET_STD
        from msea_net.xai.explainers import generate_gradcam_heatmap, GRADCAM_AVAILABLE
    except ImportError as e:
        print(f"❌ Missing dependency: {e}\nPlease install repository requirements: pip install -r requirements.txt")
        sys.exit(1)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load and preprocess image
    raw_img = Image.open(image_path).convert('RGB')
    orig_resized = raw_img.resize((224, 224))
    rgb_norm = np.array(orig_resized, dtype=np.float32) / 255.0

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(raw_img).unsqueeze(0).to(device)

    # Load model
    model = build_baseline_model(args.model, num_classes=len(DEFAULT_CLASSES), pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(tensor)
        probs = output['probs'].cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_class = DEFAULT_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

    print("\n" + "=" * 50)
    print(f"🔬 MSEA-Net Clinical Diagnostic Prediction")
    print("=" * 50)
    print(f"  Input File:           {image_path.name}")
    print(f"  Predicted Category:   {pred_class}")
    print(f"  Softmax/Dirichlet Conf: {confidence * 100:.2f}%")

    if 'uncertainty' in output:
        unc = float(output['uncertainty'].cpu().numpy()[0])
        strength = float(output['strength'].cpu().numpy()[0])
        print(f"  Epistemic Uncertainty: {unc:.4f} (Scale: 0=Certain, 8=Total Ignorance)")
        print(f"  Dirichlet Evidence (S): {strength:.2f}")

        # Selective prediction clinical decision rule
        if unc < 0.20:
            status = "✅ Automated Diagnostic Acceptance (High Certainty)"
        elif unc < 0.40:
            status = "⚠️ Secondary Clinical Verification Recommended"
        else:
            status = "🛑 Abstain / Flag for Senior Endoscopist Review"
        print(f"  Clinical Action Rule:  {status}")

    print("=" * 50)

    # Explainability
    if args.explain != "none":
        if not GRADCAM_AVAILABLE:
            print("⚠️ grad-cam library not installed. Install with: pip install grad-cam")
            return

        print(f"🎨 Generating {args.explain.upper()} visual attention map...")
        cam_vis = generate_gradcam_heatmap(
            model=model,
            image_tensor=tensor,
            original_rgb=rgb_norm,
            target_class=pred_idx,
            method=args.explain
        )
        cam_img = Image.fromarray(cam_vis)
        cam_img.save(args.output)
        print(f"  Saved heatmap visualization to: {args.output}")


if __name__ == "__main__":
    main()
