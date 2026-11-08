"""
=============================================================================
MSEA-Net: Multi-Scale Evidential Attention Network
Phase 5: Evaluation, Statistical Tests, XAI, and Publication Figures
=============================================================================
THIS CAN BE:
  Option A: Pasted as Cell 5 in the SAME notebook after Phase 4
  Option B: Pasted into a SECOND Kaggle notebook (if you ran out of time)
      - If Option B: You must also paste Phase 1 + Phase 2 + Phase 3 first
      - Then add Notebook 1's output as an input dataset
      - Change RESULTS_DIR below to point to the input path

⚠️ IMPORTANT: If using as a SECOND notebook, set this path:
    RESULTS_DIR = '/kaggle/input/YOUR-NOTEBOOK1-OUTPUT/msea_net_results'
=============================================================================
"""

# ============================================================
# SECTION 0: Setup for Second Notebook (if needed)
# ============================================================

# If running in a second notebook, uncomment and set this:
# RESULTS_DIR = '/kaggle/input/msea-net-training-output/msea_net_results'
# Config.OUTPUT_DIR = RESULTS_DIR

# Try importing XAI libraries
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    print("⚠️ grad-cam not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'grad-cam'])
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not available, will skip SHAP visualizations")
    SHAP_AVAILABLE = False

try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️ LIME not available, will skip LIME visualizations")
    LIME_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️ umap-learn not available, will skip UMAP visualizations")
    UMAP_AVAILABLE = False

from scipy import stats as scipy_stats

# Figure output directory
FIG_DIR = os.path.join(Config.OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

print("✅ Phase 5 setup complete!")


# ============================================================
# SECTION 1: Load All Results
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 1: Loading Results")
print("=" * 60)

def load_results(model_type, seed=None):
    """Load saved results from .npz files."""
    if seed is None:
        seed = Config.BASELINE_SEED
    path = os.path.join(Config.OUTPUT_DIR, f"{model_type}_seed{seed}_results.npz")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return dict(data)
    else:
        print(f"  ⚠️ Results not found: {path}")
        return None

# Load proposed model results (best seed by F1)
best_seed_idx = np.argmax([r['test_metrics']['f1_macro'] for r in msea_results])
best_seed = Config.SEEDS[best_seed_idx]
proposed_results = load_results('msea_net', best_seed)

# Load summary
summary_path = os.path.join(Config.OUTPUT_DIR, 'training_summary.json')
with open(summary_path, 'r') as f:
    training_summary = json.load(f)

print(f"  ✅ Loaded proposed model results (best seed: {best_seed})")
print(f"  ✅ Training summary loaded")


# ============================================================
# SECTION 2: Statistical Analysis
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 2: Statistical Analysis")
print("=" * 60)

def mcnemar_test(preds_a, preds_b, targets):
    """McNemar's test for comparing two classifiers."""
    correct_a = (preds_a == targets)
    correct_b = (preds_b == targets)
    
    # Contingency: a_correct & b_wrong, a_wrong & b_correct
    b_only = np.sum(~correct_a & correct_b)  # b correct, a wrong
    a_only = np.sum(correct_a & ~correct_b)   # a correct, b wrong
    
    # McNemar's test with continuity correction
    if b_only + a_only == 0:
        return 1.0, 0.0
    
    chi2 = (abs(b_only - a_only) - 1) ** 2 / (b_only + a_only + 1e-10)
    p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)
    
    return p_value, chi2


def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except Exception:
            continue
    
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return np.mean(scores), lower, upper


def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2+1e-10))
    return (mean1 - mean2) / (pooled_std + 1e-10)


# Build statistical comparison table
print("\n  --- McNemar's Test: Proposed vs Baselines ---")
print(f"  {'Baseline':<25} {'p-value':>10} {'Significant':>12}")
print(f"  {'-'*47}")

proposed_preds = proposed_results['preds']
proposed_targets = proposed_results['targets']

stat_results = {}

for name in BASELINE_MODELS + ABLATION_MODELS:
    res = load_results(name)
    if res is not None:
        p_val, chi2 = mcnemar_test(proposed_preds, res['preds'], proposed_targets)
        sig = "✅ Yes" if p_val < 0.05 else "❌ No"
        print(f"  {name:<25} {p_val:>10.4f} {sig:>12}")
        stat_results[name] = {'p_value': p_val, 'chi2': chi2}

# Bootstrap CI for proposed model AUC
from sklearn.metrics import roc_auc_score as _auc_fn

def auc_metric(y_true, y_pred_probs):
    try:
        return roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    except:
        return 0.0

proposed_probs = proposed_results['probs']

# Bootstrap for key metrics
print("\n  --- Bootstrap 95% CI (n=1000) for Proposed Model ---")
acc_mean, acc_lo, acc_hi = bootstrap_ci(
    accuracy_score, proposed_targets, proposed_preds
)
print(f"  Accuracy: {acc_mean:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")

f1_metric = lambda y, p: f1_score(y, p, average='macro', zero_division=0)
f1_mean, f1_lo, f1_hi = bootstrap_ci(f1_metric, proposed_targets, proposed_preds)
print(f"  F1 Macro: {f1_mean:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")


# ============================================================
# SECTION 3: Selective Prediction Analysis
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 3: Selective Prediction Analysis")
print("=" * 60)

def _has_uncertainties(results_dict):
    """Safely check if results contain uncertainty data."""
    if results_dict is None:
        return False
    u = results_dict.get('uncertainties', None)
    if u is None:
        return False
    if isinstance(u, np.ndarray) and u.size == 0:
        return False
    return True

if _has_uncertainties(proposed_results):
    uncertainties = proposed_results['uncertainties']
    
    coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    selective_acc = []
    
    print(f"\n  {'Coverage':>10} {'Accuracy':>10} {'# Retained':>12} {'# Deferred':>12}")
    print(f"  {'-'*44}")
    
    sorted_idx = np.argsort(uncertainties)
    
    for cov in coverages:
        n_keep = int(len(uncertainties) * cov)
        keep_idx = sorted_idx[:n_keep]
        
        acc = accuracy_score(proposed_targets[keep_idx], proposed_preds[keep_idx])
        selective_acc.append(acc)
        
        n_deferred = len(uncertainties) - n_keep
        print(f"  {cov*100:>8.0f}% {acc:>10.4f} {n_keep:>12} {n_deferred:>12}")
    
    # Save selective prediction data
    sel_pred_data = {'coverages': coverages, 'accuracies': selective_acc}
    np.savez(os.path.join(Config.OUTPUT_DIR, 'selective_prediction.npz'), **sel_pred_data)
else:
    print("  ⚠️ No uncertainty data available (model may not use EDL)")
    selective_acc = None
    coverages = None


# ============================================================
# SECTION 4: OOD Detection Analysis
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 4: Out-of-Distribution (OOD) Detection")
print("=" * 60)

if _has_uncertainties(proposed_results):
    uncertainties = proposed_results['uncertainties']
    correct_mask = proposed_preds == proposed_targets
    
    # Uncertainty for correct vs incorrect predictions
    unc_correct = uncertainties[correct_mask]
    unc_incorrect = uncertainties[~correct_mask]
    
    print(f"  Correct predictions:   mean_unc = {unc_correct.mean():.4f} ± {unc_correct.std():.4f}")
    print(f"  Incorrect predictions: mean_unc = {unc_incorrect.mean():.4f} ± {unc_incorrect.std():.4f}")
    
    if len(unc_incorrect) > 1 and len(unc_correct) > 1:
        t_stat, t_pval = scipy_stats.ttest_ind(unc_incorrect, unc_correct, equal_var=False)
        d = cohens_d(unc_incorrect, unc_correct)
        print(f"  t-test: t={t_stat:.3f}, p={t_pval:.6f}")
        print(f"  Cohen's d: {d:.3f} ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")
    
    # AUROC for misclassification detection
    from sklearn.metrics import roc_auc_score
    mis_labels = (~correct_mask).astype(int)
    try:
        mis_auc = roc_auc_score(mis_labels, uncertainties)
        print(f"  Misclassification detection AUROC: {mis_auc:.4f}")
    except:
        mis_auc = 0.0
else:
    print("  ⚠️ Skipping OOD analysis (no uncertainty data)")


# ============================================================
# SECTION 5: Robustness Analysis
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 5: Robustness Analysis (Perturbation Testing)")
print("=" * 60)

def evaluate_robustness(model, test_dataset, device, perturbation_fn, severity_levels):
    """Evaluate model under image perturbations."""
    results_per_severity = []
    
    for severity in severity_levels:
        all_preds = []
        all_targets = []
        all_uncs = []
        
        for idx in range(min(len(test_dataset), 500)):  # Use subset for speed
            img, label = test_dataset[idx]
            
            # Apply perturbation to the already-transformed tensor
            img_perturbed = perturbation_fn(img, severity)
            img_batch = img_perturbed.unsqueeze(0).to(device)
            
            with torch.no_grad():
                if Config.USE_AMP:
                    with autocast(dtype=torch.float16):
                        output = model(img_batch)
                else:
                    output = model(img_batch)
            
            pred = output['probs'].argmax(dim=1).item()
            all_preds.append(pred)
            all_targets.append(label)
            
            if 'uncertainty' in output:
                all_uncs.append(output['uncertainty'].item())
        
        acc = accuracy_score(all_targets, all_preds)
        mean_unc = np.mean(all_uncs) if all_uncs else 0.0
        results_per_severity.append({
            'severity': severity,
            'accuracy': acc,
            'mean_uncertainty': mean_unc,
        })
    
    return results_per_severity


# Define perturbations
def gaussian_noise(img, severity):
    if severity == 0:
        return img.clone()
    noise = torch.randn_like(img) * severity
    return torch.clamp(img + noise, -3, 3)

def gaussian_blur(img, severity):
    if severity == 0:
        return img.clone()
    k = max(3, int(severity * 10) // 2 * 2 + 1)
    return TF.gaussian_blur(img, kernel_size=[k, k])

def brightness_shift(img, severity):
    if severity == 0:
        return img.clone()
    return torch.clamp(img + severity, -3, 3)

# Load best model for robustness testing
print("  Loading best model for robustness testing...")
best_ckpt_path = os.path.join(Config.OUTPUT_DIR, f"msea_net_seed{best_seed}_best.pth")
if os.path.exists(best_ckpt_path):
    robustness_model = create_model('msea_net', Config.NUM_CLASSES, pretrained=False)
    ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
    robustness_model.load_state_dict(ckpt['model_state_dict'])
    robustness_model = robustness_model.to(DEVICE)
    robustness_model.eval()
    
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    blur_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    bright_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    print("  Testing Gaussian noise robustness...")
    noise_results = evaluate_robustness(robustness_model, TEST_DATASET, DEVICE, gaussian_noise, noise_levels)
    
    print("  Testing Gaussian blur robustness...")
    blur_results = evaluate_robustness(robustness_model, TEST_DATASET, DEVICE, gaussian_blur, blur_levels)
    
    print("  Testing brightness shift robustness...")
    bright_results = evaluate_robustness(robustness_model, TEST_DATASET, DEVICE, brightness_shift, bright_levels)
    
    print("\n  --- Gaussian Noise Robustness ---")
    for r in noise_results:
        print(f"    σ={r['severity']:.2f}: Acc={r['accuracy']:.4f}, Unc={r['mean_uncertainty']:.4f}")
    
    print("\n  --- Gaussian Blur Robustness ---")
    for r in blur_results:
        print(f"    σ={r['severity']:.2f}: Acc={r['accuracy']:.4f}, Unc={r['mean_uncertainty']:.4f}")
    
    # Save robustness results
    robustness_data = {
        'noise': noise_results,
        'blur': blur_results,
        'brightness': bright_results,
    }
    
    del robustness_model
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("  ⚠️ Model checkpoint not found, skipping robustness analysis")
    robustness_data = None


# ============================================================
# SECTION 6: XAI - Grad-CAM & Grad-CAM++ Visualizations
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 6: Grad-CAM & Grad-CAM++ Visualizations")
print("=" * 60)

def generate_gradcam_figures(model, test_dataset, device, save_dir, num_samples=2):
    """Generate Grad-CAM and Grad-CAM++ visualizations per class."""
    model.eval()
    
    # Target layer: last convolutional layer of backbone
    # For EfficientNetV2-S with features_only, we target the last block
    try:
        # For timm features_only models, get the last block
        # Method 1: Direct block access
        target_layers = [model.backbone.blocks[-1][-1]]
    except (AttributeError, IndexError, TypeError):
        try:
            # Method 2: Use feature_info to get module name, then access it
            last_module_name = model.backbone.feature_info[-1]['module']
            target_layers = [dict(model.backbone.named_modules())[last_module_name]]
        except Exception:
            try:
                # Method 3: Last child of backbone
                children = list(model.backbone.children())
                target_layers = [children[-1]]
            except Exception:
                print("  ⚠️ Could not identify target layer for Grad-CAM")
                return
    
    # Wrapper to make model output logits-like for Grad-CAM
    class GradCAMWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
        
        def forward(self, x):
            output = self.model(x)
            if 'evidence' in output:
                return output['evidence']
            elif 'logits' in output:
                return output['logits']
            else:
                return output['probs']
    
    wrapper = GradCAMWrapper(model)
    
    # Mean/std for un-normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(Config.NUM_CLASSES, 4, figsize=(16, Config.NUM_CLASSES * 3.5))
    fig.suptitle('Grad-CAM & Grad-CAM++ Interpretability Analysis', fontsize=16, fontweight='bold', y=1.01)
    
    col_titles = ['Original Image', 'Grad-CAM', 'Grad-CAM++', 'Overlay']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Collect samples per class
    class_samples = defaultdict(list)
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        if len(class_samples[label]) < num_samples:
            class_samples[label].append(idx)
        if all(len(v) >= num_samples for v in class_samples.values()):
            break
    
    try:
        cam_gc = GradCAM(model=wrapper, target_layers=target_layers)
        cam_gcpp = GradCAMPlusPlus(model=wrapper, target_layers=target_layers)
        
        for class_idx in range(Config.NUM_CLASSES):
            if class_idx not in class_samples or len(class_samples[class_idx]) == 0:
                continue
            
            sample_idx = class_samples[class_idx][0]
            img_tensor, label = test_dataset[sample_idx]
            
            # Un-normalize for display
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            
            input_tensor = img_tensor.unsqueeze(0).to(device)
            targets_gc = [ClassifierOutputTarget(class_idx)]
            
            # Generate CAMs
            grayscale_cam_gc = cam_gc(input_tensor=input_tensor, targets=targets_gc)[0]
            grayscale_cam_gcpp = cam_gcpp(input_tensor=input_tensor, targets=targets_gc)[0]
            
            # Overlay
            overlay_gc = show_cam_on_image(img_np.astype(np.float32), grayscale_cam_gc, use_rgb=True)
            overlay_gcpp = show_cam_on_image(img_np.astype(np.float32), grayscale_cam_gcpp, use_rgb=True)
            
            # Plot
            row = class_idx
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_ylabel(Config.CLASS_NAMES[class_idx].replace('-', '\n'), 
                                    fontsize=9, fontweight='bold', rotation=0, labelpad=60, ha='right')
            
            axes[row, 1].imshow(grayscale_cam_gc, cmap='jet')
            axes[row, 2].imshow(grayscale_cam_gcpp, cmap='jet')
            axes[row, 3].imshow(overlay_gcpp)
        
        del cam_gc, cam_gcpp
        
    except Exception as e:
        print(f"  ⚠️ Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
    
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fig_gradcam_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {save_path}")


# Load best model and generate Grad-CAM
if os.path.exists(best_ckpt_path):
    gc_model = create_model('msea_net', Config.NUM_CLASSES, pretrained=False)
    ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
    gc_model.load_state_dict(ckpt['model_state_dict'])
    gc_model = gc_model.to(DEVICE)
    gc_model.eval()
    
    generate_gradcam_figures(gc_model, TEST_DATASET, DEVICE, FIG_DIR)
    
    del gc_model
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# SECTION 7: XAI - LIME Visualizations
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 7: LIME Visualizations")
print("=" * 60)

if LIME_AVAILABLE and os.path.exists(best_ckpt_path):
    lime_model = create_model('msea_net', Config.NUM_CLASSES, pretrained=False)
    ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
    lime_model.load_state_dict(ckpt['model_state_dict'])
    lime_model = lime_model.to(DEVICE)
    lime_model.eval()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    def predict_fn_lime(images_np):
        """Prediction function for LIME (expects numpy HWC images 0-1)."""
        batch = []
        transform = get_transforms(False)
        for img_np in images_np:
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_tensor = transform(img_pil)
            batch.append(img_tensor)
        batch_tensor = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            if Config.USE_AMP:
                with autocast(dtype=torch.float16):
                    output = lime_model(batch_tensor)
            else:
                output = lime_model(batch_tensor)
        return output['probs'].float().cpu().numpy()
    
    explainer = lime_image.LimeImageExplainer()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('LIME Interpretability Analysis (Top-8 Classes)', fontsize=16, fontweight='bold')
    
    for class_idx in range(min(8, Config.NUM_CLASSES)):
        row, col = class_idx // 4, class_idx % 4
        
        # Find a sample for this class
        sample_idx = None
        for i in range(len(TEST_DATASET)):
            _, label = TEST_DATASET[i]
            if label == class_idx:
                sample_idx = i
                break
        
        if sample_idx is None:
            continue
        
        img_tensor, label = TEST_DATASET[sample_idx]
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        try:
            explanation = explainer.explain_instance(
                img_np, predict_fn_lime,
                top_labels=1, hide_color=0,
                num_samples=200,  # Reduced for speed on Kaggle
            )
            
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=10,
                hide_rest=False,
            )
            
            # Create overlay
            try:
                from skimage.segmentation import mark_boundaries
            except ImportError:
                print("  ⚠️ skimage not available, installing...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-image'])
                from skimage.segmentation import mark_boundaries
            lime_img = mark_boundaries(temp, mask)
            
            axes[row, col].imshow(lime_img)
            axes[row, col].set_title(Config.CLASS_NAMES[class_idx], fontsize=10)
        except Exception as e:
            axes[row, col].imshow(img_np)
            axes[row, col].set_title(f"{Config.CLASS_NAMES[class_idx]}\n(LIME failed)", fontsize=9)
        
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'fig_lime_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {save_path}")
    
    del lime_model
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("  ⚠️ Skipping LIME (not available or model not found)")


# ============================================================
# SECTION 8: XAI - SHAP Visualizations
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 8: SHAP Visualizations")
print("=" * 60)

if SHAP_AVAILABLE and os.path.exists(best_ckpt_path):
    try:
        shap_model = create_model('msea_net', Config.NUM_CLASSES, pretrained=False)
        ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
        shap_model.load_state_dict(ckpt['model_state_dict'])
        shap_model = shap_model.to(DEVICE)
        shap_model.eval()
        
        # Wrapper for SHAP
        class SHAPWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                output = self.model(x)
                return output['probs']
        
        shap_wrapper = SHAPWrapper(shap_model)
        
        # Get background samples (small batch)
        bg_images = []
        for i, (img, _) in enumerate(TEST_DATASET):
            if i >= 20:
                break
            bg_images.append(img)
        background = torch.stack(bg_images).to(DEVICE)
        
        # Get test samples (one per class if possible)
        test_images = []
        test_labels_shap = []
        seen_classes = set()
        for i in range(len(TEST_DATASET)):
            img, label = TEST_DATASET[i]
            if label not in seen_classes:
                test_images.append(img)
                test_labels_shap.append(label)
                seen_classes.add(label)
            if len(seen_classes) >= min(4, Config.NUM_CLASSES):
                break
        test_batch = torch.stack(test_images).to(DEVICE)
        
        # GradientExplainer is fastest for neural networks
        explainer_shap = shap.GradientExplainer(shap_wrapper, background)
        shap_values = explainer_shap.shap_values(test_batch)
        
        # Plot SHAP values
        fig, axes = plt.subplots(len(test_images), 2, figsize=(10, len(test_images) * 4))
        fig.suptitle('SHAP Feature Attribution Analysis', fontsize=16, fontweight='bold')
        
        mean = np.array([0.485, 0.456, 0.406])
        std_arr = np.array([0.229, 0.224, 0.225])
        
        for i in range(len(test_images)):
            # Original
            img_np = test_images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * std_arr + mean
            img_np = np.clip(img_np, 0, 1)
            
            ax_orig = axes[i, 0] if len(test_images) > 1 else axes[0]
            ax_shap = axes[i, 1] if len(test_images) > 1 else axes[1]
            
            ax_orig.imshow(img_np)
            ax_orig.set_title(f"Original: {Config.CLASS_NAMES[test_labels_shap[i]]}", fontsize=10)
            ax_orig.axis('off')
            
            # SHAP heatmap (sum across channels)
            if isinstance(shap_values, list):
                sv = shap_values[test_labels_shap[i]][i]
            else:
                sv = shap_values[i]
            
            if sv.ndim == 3:
                shap_heatmap = np.abs(sv).sum(axis=0)
            else:
                shap_heatmap = np.abs(sv).sum(axis=0)
            
            ax_shap.imshow(img_np)
            ax_shap.imshow(shap_heatmap, cmap='hot', alpha=0.5)
            ax_shap.set_title(f"SHAP Attribution", fontsize=10)
            ax_shap.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, 'fig_shap_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: {save_path}")
        
        del shap_model, shap_wrapper, background, test_batch
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"  ⚠️ SHAP failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ⚠️ Skipping SHAP (not available)")


# ============================================================
# SECTION 9: XAI - UMAP Visualization (3-Panel)
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 9: UMAP Visualization (3-Panel)")
print("=" * 60)

if UMAP_AVAILABLE and proposed_results is not None:
    features = proposed_results['features']
    targets = proposed_results['targets']
    preds = proposed_results['preds']
    uncerts = proposed_results.get('uncertainties', None)
    
    # Fit UMAP
    print("  Computing UMAP embedding...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, metric='cosine')
    embedding = reducer.fit_transform(features)
    
    # 3-Panel UMAP figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('UMAP Feature Space Visualization', fontsize=16, fontweight='bold')
    
    # Color palette for 8 classes
    colors_class = plt.cm.Set2(np.linspace(0, 1, Config.NUM_CLASSES))
    
    # Panel 1: Color by ground truth class
    for c in range(Config.NUM_CLASSES):
        mask = targets == c
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors_class[c]], s=8, alpha=0.6,
                       label=Config.CLASS_NAMES[c])
    axes[0].set_title('(a) Ground Truth Labels', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=7, markerscale=3, ncol=2, loc='best',
                   framealpha=0.9, edgecolor='gray')
    axes[0].set_xlabel('UMAP-1')
    axes[0].set_ylabel('UMAP-2')
    
    # Panel 2: Color by epistemic uncertainty
    if uncerts is not None:
        sc = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                            c=uncerts, cmap='RdYlBu_r', s=8, alpha=0.6,
                            vmin=np.percentile(uncerts, 1),
                            vmax=np.percentile(uncerts, 99))
        plt.colorbar(sc, ax=axes[1], label='Epistemic Uncertainty', shrink=0.8)
        axes[1].set_title('(b) Epistemic Uncertainty', fontsize=13, fontweight='bold')
    else:
        # Use max probability as confidence proxy
        max_probs = proposed_results['probs'].max(axis=1)
        sc = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                            c=1-max_probs, cmap='RdYlBu_r', s=8, alpha=0.6)
        plt.colorbar(sc, ax=axes[1], label='1 - Max Probability', shrink=0.8)
        axes[1].set_title('(b) Prediction Confidence', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('UMAP-1')
    axes[1].set_ylabel('UMAP-2')
    
    # Panel 3: Color by prediction correctness
    correct_mask = preds == targets
    axes[2].scatter(embedding[correct_mask, 0], embedding[correct_mask, 1],
                   c='#2196F3', s=8, alpha=0.5, label=f'Correct ({correct_mask.sum()})')
    axes[2].scatter(embedding[~correct_mask, 0], embedding[~correct_mask, 1],
                   c='#F44336', s=15, alpha=0.8, label=f'Incorrect ({(~correct_mask).sum()})',
                   marker='x', linewidths=1)
    axes[2].set_title('(c) Prediction Correctness', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10, markerscale=2, loc='best', framealpha=0.9)
    axes[2].set_xlabel('UMAP-1')
    axes[2].set_ylabel('UMAP-2')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'fig_umap_3panel.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {save_path}")
else:
    print("  ⚠️ Skipping UMAP (not available or no results)")


# ============================================================
# SECTION 10: Publication Figures
# ============================================================

print("\n" + "=" * 60)
print("  SECTION 10: Publication-Ready Figures")
print("=" * 60)

# Display name mappings
model_names_display = {
    'vgg16': 'VGG-16', 'resnet50': 'ResNet-50', 'densenet121': 'DenseNet-121',
    'inception_v3': 'InceptionV3', 'mobilenetv3': 'MobileNetV3',
    'efficientnetv2s_plain': 'EfficientNetV2-S (Plain)', 'msea_net': 'MSEA-Net (Ours)',
    'msea_no_multiscale': 'w/o Multi-Scale', 'msea_no_attention': 'w/o ACSA',
    'msea_no_gff': 'w/o GFF', 'msea_no_edl': 'w/o EDL',
}

# Load all histories
histories_path = os.path.join(Config.OUTPUT_DIR, 'all_histories.json')
all_histories = {}
if os.path.exists(histories_path):
    with open(histories_path, 'r') as f:
        all_histories = json.load(f)

# Create subdirs for organized output
INDIV_DIR = os.path.join(FIG_DIR, 'individual_curves')
CM_DIR = os.path.join(FIG_DIR, 'confusion_matrices')
os.makedirs(INDIV_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)


# ---------------------------------------------------------------
# Fig 1: INDIVIDUAL Training Curves (Loss + Accuracy) per model
# ---------------------------------------------------------------
print("  📊 Generating INDIVIDUAL training curves per model...")

for model_key, hist in all_histories.items():
    display_name = model_names_display.get(model_key.split('_seed')[0] if '_seed' in model_key else model_key, model_key)
    
    epochs_range = range(1, len(hist['train_loss']) + 1)
    has_val_loss = 'val_loss' in hist and len(hist.get('val_loss', [])) > 0
    
    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, hist['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if has_val_loss:
        ax.plot(epochs_range, hist['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{display_name} — Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_path = os.path.join(INDIV_DIR, f'loss_{model_key}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Accuracy curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, hist['train_acc'], 'b-', linewidth=2, label='Train Accuracy')
    if 'val_acc' in hist:
        ax.plot(epochs_range, hist['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{display_name} — Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_path = os.path.join(INDIV_DIR, f'accuracy_{model_key}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"  💾 Individual curves saved to: {INDIV_DIR}/")


# ---------------------------------------------------------------
# Fig 2: COMBINED Training Curves (all models overlay)
# ---------------------------------------------------------------
print("  📊 Generating combined training curves...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Training Dynamics — All Models', fontsize=16, fontweight='bold')

all_model_colors = {
    'vgg16': '#9E9E9E', 'resnet50': '#2196F3', 'densenet121': '#4CAF50',
    'inception_v3': '#FF9800', 'mobilenetv3': '#9C27B0', 'efficientnetv2s_plain': '#00BCD4',
}

# Plot proposed model (all seeds — shaded)
msea_keys = [k for k in all_histories if 'msea_net_seed' in k]
for sk in msea_keys:
    hist = all_histories[sk]
    ep = range(1, len(hist['train_loss']) + 1)
    axes[0, 0].plot(ep, hist['train_loss'], color='#F44336', alpha=0.3)
    axes[0, 1].plot(ep, hist['train_acc'], color='#F44336', alpha=0.3)
    axes[1, 0].plot(ep, hist['val_f1'], color='#F44336', alpha=0.3)
    axes[1, 1].plot(ep, hist['val_auc'], color='#F44336', alpha=0.3)
axes[0, 0].plot([], [], color='#F44336', linewidth=2, label='MSEA-Net (Ours)')

# Plot baselines
for name, hist in all_histories.items():
    if name in all_model_colors:
        color = all_model_colors[name]
        dname = model_names_display.get(name, name)
        ep = range(1, len(hist['train_loss']) + 1)
        axes[0, 0].plot(ep, hist['train_loss'], color=color, alpha=0.8, label=dname, linewidth=1.5)
        axes[0, 1].plot(ep, hist['train_acc'], color=color, alpha=0.8, linewidth=1.5)
        axes[1, 0].plot(ep, hist['val_f1'], color=color, alpha=0.8, linewidth=1.5)
        axes[1, 1].plot(ep, hist['val_auc'], color=color, alpha=0.8, linewidth=1.5)

axes[0, 0].set_ylabel('Training Loss'); axes[0, 0].set_title('(a) Training Loss')
axes[0, 0].legend(fontsize=7, ncol=2)
axes[0, 1].set_ylabel('Training Accuracy'); axes[0, 1].set_title('(b) Training Accuracy')
axes[1, 0].set_ylabel('Val F1 (Macro)'); axes[1, 0].set_title('(c) Validation F1')
axes[1, 0].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Val AUC'); axes[1, 1].set_title('(d) Validation AUC')
axes[1, 1].set_xlabel('Epoch')
for ax in axes.flat:
    ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_training_curves_combined.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: fig_training_curves_combined.png")


# ---------------------------------------------------------------
# Fig 3: INDIVIDUAL Confusion Matrices per model
# ---------------------------------------------------------------
print("  📊 Generating confusion matrices for ALL models...")

short_names = [n.replace('dyed-lifted-', 'DL-').replace('dyed-resection-', 'DR-')
               .replace('normal-', 'N-').replace('ulcerative-', 'U-')
               .replace('polyps', 'Polyps').replace('esophagitis', 'Esoph.')
               for n in Config.CLASS_NAMES]

# Proposed model
cm_p = confusion_matrix(proposed_results['targets'], proposed_results['preds'])
cm_p_norm = cm_p.astype('float') / cm_p.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('MSEA-Net (Proposed) — Confusion Matrix', fontsize=16, fontweight='bold')
sns.heatmap(cm_p, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=short_names, yticklabels=short_names)
axes[0].set_title('(a) Raw Counts', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
sns.heatmap(cm_p_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
            xticklabels=short_names, yticklabels=short_names, vmin=0, vmax=1)
axes[1].set_title('(b) Normalized', fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(CM_DIR, 'cm_msea_net.png'), dpi=300, bbox_inches='tight')
plt.close()

# Baselines + ablation
all_model_names = BASELINE_MODELS + ABLATION_MODELS
for mname in all_model_names:
    res = load_results(mname)
    if res is None:
        continue
    
    dname = model_names_display.get(mname, mname)
    cm_m = confusion_matrix(res['targets'], res['preds'])
    cm_m_norm = cm_m.astype('float') / cm_m.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{dname} — Confusion Matrix', fontsize=16, fontweight='bold')
    sns.heatmap(cm_m, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
                xticklabels=short_names, yticklabels=short_names)
    axes[0].set_title('(a) Raw Counts', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    sns.heatmap(cm_m_norm, annot=True, fmt='.3f', cmap='Oranges', ax=axes[1],
                xticklabels=short_names, yticklabels=short_names, vmin=0, vmax=1)
    axes[1].set_title('(b) Normalized', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f'cm_{mname}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"  💾 Confusion matrices saved to: {CM_DIR}/")


# ---------------------------------------------------------------
# Fig 4: Baseline Comparison Bar Chart
# ---------------------------------------------------------------
print("  📊 Generating baseline comparison figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Baseline Comparison', fontsize=16, fontweight='bold')
metrics_list = ['accuracy', 'f1_macro', 'auc_macro']
metric_labels = ['Accuracy', 'F1 Score (Macro)', 'AUC-ROC (Macro)']

for ax, metric, label in zip(axes, metrics_list, metric_labels):
    names = []
    values = []
    bar_colors = []
    
    for bname in BASELINE_MODELS:
        if bname in baseline_results and 'error' not in baseline_results[bname]:
            names.append(model_names_display.get(bname, bname).replace(' ', '\n'))
            values.append(baseline_results[bname]['test_metrics'][metric])
            bar_colors.append('#90CAF9')
    
    prop_vals = [r['test_metrics'][metric] for r in msea_results]
    names.append('MSEA-Net\n(Ours)')
    values.append(np.mean(prop_vals))
    bar_colors.append('#F44336')
    
    bars = ax.bar(range(len(names)), values, color=bar_colors, edgecolor='white', linewidth=0.5)
    if len(prop_vals) > 1:
        ax.errorbar(len(names)-1, np.mean(prop_vals), yerr=np.std(prop_vals),
                    fmt='none', ecolor='black', capsize=5, capthick=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, ha='center')
    ax.set_ylabel(label)
    ax.set_ylim(min(values) - 0.05, 1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_baseline_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: fig_baseline_comparison.png")


# ---------------------------------------------------------------
# Fig 5: Per-Class Metrics (Precision / Recall / F1) Bar Chart
# ---------------------------------------------------------------
print("  📊 Generating per-class metrics figure...")
cr = classification_report(proposed_results['targets'], proposed_results['preds'],
                           target_names=Config.CLASS_NAMES, output_dict=True, zero_division=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('MSEA-Net — Per-Class Performance', fontsize=16, fontweight='bold')

per_class_metrics = ['precision', 'recall', 'f1-score']
per_class_labels = ['Precision', 'Recall', 'F1-Score']
bar_cmap = plt.cm.Set2(np.linspace(0, 1, Config.NUM_CLASSES))

for ax, pm, plabel in zip(axes, per_class_metrics, per_class_labels):
    vals = [cr[cn][pm] for cn in Config.CLASS_NAMES]
    bars = ax.barh(range(Config.NUM_CLASSES), vals, color=bar_cmap, edgecolor='white')
    ax.set_yticks(range(Config.NUM_CLASSES))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel(plabel)
    ax.set_title(plabel, fontweight='bold')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_per_class_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: fig_per_class_metrics.png")


# ---------------------------------------------------------------
# Fig 6: ROC & PR Curves
# ---------------------------------------------------------------
print("  📊 Generating ROC and PR curves...")
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

y_test_bin = label_binarize(proposed_results['targets'], classes=range(Config.NUM_CLASSES))
y_probs = proposed_results['probs']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.tab10(np.linspace(0, 1, Config.NUM_CLASSES))

for i in range(Config.NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=colors[i], lw=1.5, alpha=0.8,
                label=f'{short_names[i]} (AUC={roc_auc:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('(a) ROC Curves (One-vs-Rest)', fontweight='bold')
axes[0].legend(fontsize=8, loc='lower right', framealpha=0.9)
axes[0].grid(alpha=0.3)

for i in range(Config.NUM_CLASSES):
    prec_c, rec_c, _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_probs[:, i])
    axes[1].plot(rec_c, prec_c, color=colors[i], lw=1.5, alpha=0.8,
                label=f'{short_names[i]} (AP={ap:.3f})')

axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('(b) Precision-Recall Curves', fontweight='bold')
axes[1].legend(fontsize=8, loc='lower left', framealpha=0.9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_roc_pr_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: fig_roc_pr_curves.png")


# ---------------------------------------------------------------
# Fig 7: Ablation Study
# ---------------------------------------------------------------
print("  📊 Generating ablation study figure...")
abl_names_display = {
    'msea_net': 'Full Model', 'msea_no_multiscale': 'w/o Multi-Scale',
    'msea_no_attention': 'w/o ACSA', 'msea_no_gff': 'w/o GFF', 'msea_no_edl': 'w/o EDL',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Ablation Study', fontsize=16, fontweight='bold')

abl_data = {'names': [], 'acc': [], 'f1': [], 'auc': [], 'colors': []}
abl_data['names'].append('Full Model')
abl_data['acc'].append(np.mean([r['test_metrics']['accuracy'] for r in msea_results]))
abl_data['f1'].append(np.mean([r['test_metrics']['f1_macro'] for r in msea_results]))
abl_data['auc'].append(np.mean([r['test_metrics']['auc_macro'] for r in msea_results]))
abl_data['colors'].append('#F44336')

for abl_name in ABLATION_MODELS:
    if abl_name in ablation_results and 'error' not in ablation_results[abl_name]:
        m = ablation_results[abl_name]['test_metrics']
        abl_data['names'].append(abl_names_display.get(abl_name, abl_name))
        abl_data['acc'].append(m['accuracy'])
        abl_data['f1'].append(m['f1_macro'])
        abl_data['auc'].append(m['auc_macro'])
        abl_data['colors'].append('#90CAF9')

for ax, vals, ylabel in zip(axes, [abl_data['acc'], abl_data['f1'], abl_data['auc']],
                              ['Accuracy', 'F1 (Macro)', 'AUC (Macro)']):
    bars = ax.bar(range(len(abl_data['names'])), vals, color=abl_data['colors'], edgecolor='white')
    ax.set_xticks(range(len(abl_data['names'])))
    ax.set_xticklabels(abl_data['names'], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontweight='bold')
    ax.set_ylim(min(vals) - 0.05, 1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_ablation_study.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: fig_ablation_study.png")


# ---------------------------------------------------------------
# Fig 8: Selective Prediction Curve
# ---------------------------------------------------------------
if selective_acc is not None:
    print("  📊 Generating selective prediction curve...")
    fig, ax = plt.subplots(figsize=(8, 5))
    cov_pct = [c * 100 for c in coverages]
    ax.plot(cov_pct, [a * 100 for a in selective_acc], 'o-', color='#F44336',
            linewidth=2, markersize=8, label='MSEA-Net')
    ax.axhline(y=selective_acc[0] * 100, color='gray', linestyle='--', alpha=0.5,
              label=f'Full Coverage ({selective_acc[0]*100:.1f}%)')
    ax.set_xlabel('Coverage (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Uncertainty-Guided Selective Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if len(selective_acc) > 6:
        ax.annotate(f'{selective_acc[6]*100:.1f}%\n@ {coverages[6]*100:.0f}% cov.',
                   xy=(cov_pct[6], selective_acc[6]*100),
                   xytext=(cov_pct[6]-15, selective_acc[6]*100+2),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black'))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_selective_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: fig_selective_prediction.png")


# ---------------------------------------------------------------
# Fig 9: Calibration Reliability Diagram
# ---------------------------------------------------------------
print("  📊 Generating calibration reliability diagram...")
def plot_calibration(probs, targets, n_bins=15, save_path=None):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies_arr = (predictions == targets).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []; bin_accs = []; bin_confs = []; bin_counts = []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_accs.append(accuracies_arr[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    ax1.bar(bin_centers, bin_accs, width=1/n_bins * 0.8, alpha=0.7, color='#2196F3',
           edgecolor='white', label='Model')
    ax1.plot(bin_confs, bin_accs, 'o-', color='#F44336', markersize=4)
    ece_val = compute_ece(probs, targets, n_bins)
    ax1.set_title(f'Calibration Reliability Diagram (ECE = {ece_val:.4f})',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy'); ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.grid(alpha=0.3)
    ax2.bar(bin_centers, bin_counts, width=1/n_bins * 0.8, color='#90CAF9', edgecolor='white')
    ax2.set_xlabel('Confidence'); ax2.set_ylabel('Count'); ax2.set_xlim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return ece_val

ece = plot_calibration(proposed_results['probs'], proposed_results['targets'],
                       save_path=os.path.join(FIG_DIR, 'fig_calibration.png'))
print(f"  💾 Saved: fig_calibration.png (ECE = {ece:.4f})")


# ---------------------------------------------------------------
# Fig 10: Uncertainty Distribution
# ---------------------------------------------------------------
if _has_uncertainties(proposed_results):
    print("  📊 Generating uncertainty distribution figure...")
    uncerts = proposed_results['uncertainties']
    correct = proposed_preds == proposed_targets
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(uncerts[correct], bins=50, alpha=0.7, color='#4CAF50', label=f'Correct ({correct.sum()})',
           density=True)
    ax.hist(uncerts[~correct], bins=50, alpha=0.7, color='#F44336', label=f'Incorrect ({(~correct).sum()})',
           density=True)
    ax.set_xlabel('Epistemic Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Uncertainty Distribution: Correct vs Incorrect Predictions',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_uncertainty_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: fig_uncertainty_dist.png")


# ---------------------------------------------------------------
# Fig 11: Robustness Analysis
# ---------------------------------------------------------------
if robustness_data is not None:
    print("  📊 Generating robustness analysis figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Robustness Analysis Under Image Perturbations', fontsize=16, fontweight='bold')
    
    perturbation_types = [
        ('noise', 'Gaussian Noise (σ)', noise_levels),
        ('blur', 'Gaussian Blur (σ)', blur_levels),
        ('brightness', 'Brightness Shift', bright_levels),
    ]
    
    for ax, (ptype, xlabel, levels) in zip(axes, perturbation_types):
        data = robustness_data[ptype]
        accs_r = [d['accuracy'] for d in data]
        uncs_r = [d['mean_uncertainty'] for d in data]
        ax2 = ax.twinx()
        l1 = ax.plot(levels, accs_r, 'o-', color='#2196F3', linewidth=2, markersize=6, label='Accuracy')
        l2 = ax2.plot(levels, uncs_r, 's--', color='#F44336', linewidth=2, markersize=6, label='Uncertainty')
        ax.set_xlabel(xlabel); ax.set_ylabel('Accuracy', color='#2196F3')
        ax2.set_ylabel('Mean Uncertainty', color='#F44336')
        lines = l1 + l2
        ax.legend(lines, [l.get_label() for l in lines], loc='center left', fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_robustness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: fig_robustness.png")


# ---------------------------------------------------------------
# Fig 12: Comprehensive Results Table (saved as image)
# ---------------------------------------------------------------
print("  📊 Generating comprehensive results table...")
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

table_data = []
table_colors = []

# Header
header = ['Model', 'Accuracy', 'F1 (Macro)', 'AUC (Macro)', 'Precision', 'Recall', 'ECE ↓', 'Brier ↓']

# Proposed model row
row = ['MSEA-Net (Ours)',
       f"{np.mean(accs):.4f}±{np.std(accs):.4f}",
       f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}",
       f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}",
       f"{np.mean([r['test_metrics']['precision'] for r in msea_results]):.4f}",
       f"{np.mean([r['test_metrics']['recall'] for r in msea_results]):.4f}",
       f"{np.mean(eces):.4f}",
       f"{np.mean([r['test_metrics']['brier'] for r in msea_results]):.4f}"]
table_data.append(row)
table_colors.append(['#FFCDD2'] * len(header))

# Baselines
for bname in BASELINE_MODELS:
    if bname in baseline_results and 'error' not in baseline_results[bname]:
        m = baseline_results[bname]['test_metrics']
        row = [model_names_display.get(bname, bname),
               f"{m['accuracy']:.4f}", f"{m['f1_macro']:.4f}", f"{m['auc_macro']:.4f}",
               f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['ece']:.4f}", f"{m['brier']:.4f}"]
        table_data.append(row)
        table_colors.append(['#E3F2FD'] * len(header))

# Ablation
for aname in ABLATION_MODELS:
    if aname in ablation_results and 'error' not in ablation_results[aname]:
        m = ablation_results[aname]['test_metrics']
        row = [model_names_display.get(aname, aname),
               f"{m['accuracy']:.4f}", f"{m['f1_macro']:.4f}", f"{m['auc_macro']:.4f}",
               f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['ece']:.4f}", f"{m['brier']:.4f}"]
        table_data.append(row)
        table_colors.append(['#FFF3E0'] * len(header))

table = ax.table(cellText=table_data, colLabels=header, cellColours=table_colors,
                loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)
for (row_idx, col_idx), cell in table.get_celld().items():
    if row_idx == 0:
        cell.set_fontsize(11)
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', weight='bold')

ax.set_title('Comprehensive Results Comparison', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_results_table.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: fig_results_table.png")


# ============================================================
# SECTION 11: Final Summary Report
# ============================================================

print("\n" + "=" * 70)
print("  ✅ PHASE 5 COMPLETE — ALL EVALUATION & VISUALIZATION DONE!")
print("=" * 70)

print(f"\n  📁 Output directory: {Config.OUTPUT_DIR}")
print(f"\n  📁 Figure directories:")
print(f"     📁 {FIG_DIR}/  (main figures)")
print(f"     📁 {INDIV_DIR}/  (individual training curves)")
print(f"     📁 {CM_DIR}/  (confusion matrices per model)")

# Count all figures
total_figs = 0
for root, dirs, files in os.walk(FIG_DIR):
    total_figs += len([f for f in files if f.endswith('.png')])

print(f"\n  📊 Total figures generated: {total_figs}")

print(f"\n  📊 Proposed Model (MSEA-Net) — Final Results:")
print(f"     Accuracy:   {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"     F1 (macro): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"     AUC (macro): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"     ECE:        {np.mean(eces):.4f} ± {np.std(eces):.4f}")

if selective_acc is not None and len(selective_acc) > 6:
    print(f"\n  🎯 Selective Prediction (70% coverage):")
    print(f"     Accuracy:   {selective_acc[6]*100:.1f}% (vs {selective_acc[0]*100:.1f}% full coverage)")
    print(f"     Improvement: +{(selective_acc[6]-selective_acc[0])*100:.1f} percentage points")

print(f"\n  🏁 Ready for paper writing!")
print("=" * 70)

