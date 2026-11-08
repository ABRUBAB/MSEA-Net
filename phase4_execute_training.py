"""
=============================================================================
MSEA-Net: Multi-Scale Evidential Attention Network
Phase 4: Execute All Training — Main Model + Baselines + Ablation
=============================================================================
PASTE THIS AS THE FOURTH CELL in your Kaggle notebook.
Requires Phase 1, 2, and 3 to be executed first.

⚠️ THIS CELL RUNS ALL TRAINING. Estimated time: ~2-2.5 hours on T4/P100.
=============================================================================
"""

print("\n" + "=" * 60)
print("  MSEA-Net: Phase 4 — Execute All Training")
print("=" * 60)

# ============================================================
# SECTION 1: Train Proposed Model (MSEA-Net) — 3 Seeds
# ============================================================

print("\n" + "=" * 60)
print("  STAGE 1: Training MSEA-Net (Proposed Model) — 3 Seeds")
print("=" * 60)

all_results = {}
msea_results = []

for seed in Config.SEEDS:
    result = train_model(
        model_type='msea_net',
        seed=seed,
        device=DEVICE,
        train_loader=TRAIN_LOADER,
        val_loader=VAL_LOADER,
        test_loader=TEST_LOADER,
        epochs=Config.EPOCHS,
        save_dir=Config.OUTPUT_DIR,
    )
    msea_results.append(result)
    all_results[f'msea_net_seed{seed}'] = result

# Print multi-seed summary for proposed model
print("\n" + "-" * 60)
print("  📊 MSEA-Net Multi-Seed Summary (3 seeds)")
print("-" * 60)
accs = [r['test_metrics']['accuracy'] for r in msea_results]
f1s = [r['test_metrics']['f1_macro'] for r in msea_results]
aucs = [r['test_metrics']['auc_macro'] for r in msea_results]
eces = [r['test_metrics']['ece'] for r in msea_results]

print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"  F1 (macro): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"  AUC (macro): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  ECE:       {np.mean(eces):.4f} ± {np.std(eces):.4f}")


# ============================================================
# SECTION 2: Train Baseline Models — 1 Seed Each
# ============================================================

print("\n" + "=" * 60)
print("  STAGE 2: Training Baseline Models")
print("=" * 60)

BASELINE_MODELS = [
    'vgg16',
    'resnet50',
    'densenet121',
    'inception_v3',
    'mobilenetv3',
    'efficientnetv2s_plain',
]

baseline_results = {}

for model_name in BASELINE_MODELS:
    try:
        result = train_model(
            model_type=model_name,
            seed=Config.BASELINE_SEED,
            device=DEVICE,
            train_loader=TRAIN_LOADER,
            val_loader=VAL_LOADER,
            test_loader=TEST_LOADER,
            epochs=Config.EPOCHS,
            save_dir=Config.OUTPUT_DIR,
        )
        baseline_results[model_name] = result
        all_results[model_name] = result
    except Exception as e:
        print(f"  ❌ ERROR training {model_name}: {e}")
        print(f"     Skipping {model_name} and continuing...")
        # Save error info but continue
        baseline_results[model_name] = {'error': str(e)}
        continue

# Print baseline summary
print("\n" + "-" * 60)
print("  📊 Baseline Comparison Summary")
print("-" * 60)
print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
print(f"  {'-'*55}")

# First print proposed model (mean of 3 seeds)
print(f"  {'MSEA-Net (Ours)':<25} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f} {np.mean(aucs):>10.4f}")

for name, res in baseline_results.items():
    if 'error' in res:
        print(f"  {name:<25} {'ERROR':>10} {'ERROR':>10} {'ERROR':>10}")
    else:
        m = res['test_metrics']
        print(f"  {name:<25} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} {m['auc_macro']:>10.4f}")


# ============================================================
# SECTION 3: Ablation Study — 1 Seed Each
# ============================================================

print("\n" + "=" * 60)
print("  STAGE 3: Ablation Study")
print("=" * 60)

ABLATION_MODELS = [
    'msea_no_multiscale',
    'msea_no_attention',
    'msea_no_gff',
    'msea_no_edl',
]

ablation_results = {}

for model_name in ABLATION_MODELS:
    try:
        result = train_model(
            model_type=model_name,
            seed=Config.BASELINE_SEED,
            device=DEVICE,
            train_loader=TRAIN_LOADER,
            val_loader=VAL_LOADER,
            test_loader=TEST_LOADER,
            epochs=Config.EPOCHS,
            save_dir=Config.OUTPUT_DIR,
        )
        ablation_results[model_name] = result
        all_results[model_name] = result
    except Exception as e:
        print(f"  ❌ ERROR training {model_name}: {e}")
        print(f"     Skipping and continuing...")
        ablation_results[model_name] = {'error': str(e)}
        continue

# Print ablation summary
print("\n" + "-" * 60)
print("  📊 Ablation Study Summary")
print("-" * 60)
print(f"  {'Variant':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'ECE':>10}")
print(f"  {'-'*65}")

# Full model (mean of seeds)
print(f"  {'MSEA-Net (Full)':<25} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f} {np.mean(aucs):>10.4f} {np.mean(eces):>10.4f}")

for name, res in ablation_results.items():
    if 'error' in res:
        print(f"  {name:<25} {'ERROR':>10} {'ERROR':>10} {'ERROR':>10} {'ERROR':>10}")
    else:
        m = res['test_metrics']
        print(f"  {name:<25} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} {m['auc_macro']:>10.4f} {m['ece']:>10.4f}")


# ============================================================
# SECTION 4: Save Comprehensive Results Summary
# ============================================================

print("\n💾 Saving comprehensive results...")

# Build summary table
summary = {
    'proposed_model': {
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'ece_mean': float(np.mean(eces)),
        'ece_std': float(np.std(eces)),
        'seeds': Config.SEEDS,
        'per_seed_accuracy': [float(a) for a in accs],
        'per_seed_f1': [float(f) for f in f1s],
        'per_seed_auc': [float(a) for a in aucs],
    },
    'baselines': {},
    'ablation': {},
}

for name, res in baseline_results.items():
    if 'error' not in res:
        m = res['test_metrics']
        summary['baselines'][name] = {
            'accuracy': float(m['accuracy']),
            'f1_macro': float(m['f1_macro']),
            'auc_macro': float(m['auc_macro']),
            'ece': float(m['ece']),
            'brier': float(m['brier']),
            'precision': float(m['precision']),
            'recall': float(m['recall']),
            'training_time': float(res.get('training_time', 0)),
        }

for name, res in ablation_results.items():
    if 'error' not in res:
        m = res['test_metrics']
        summary['ablation'][name] = {
            'accuracy': float(m['accuracy']),
            'f1_macro': float(m['f1_macro']),
            'auc_macro': float(m['auc_macro']),
            'ece': float(m['ece']),
            'brier': float(m['brier']),
            'precision': float(m['precision']),
            'recall': float(m['recall']),
            'training_time': float(res.get('training_time', 0)),
        }

# Save as JSON
summary_path = os.path.join(Config.OUTPUT_DIR, 'training_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  💾 Summary saved: {summary_path}")

# Save all training histories
histories = {}
for name, res in all_results.items():
    if 'error' not in res and 'history' in res:
        histories[name] = res['history']

histories_path = os.path.join(Config.OUTPUT_DIR, 'all_histories.json')
with open(histories_path, 'w') as f:
    json.dump(histories, f, indent=2)
print(f"  💾 Histories saved: {histories_path}")


# ============================================================
# SECTION 5: Total Training Time
# ============================================================

total_training_time = sum(
    r.get('training_time', 0) for r in all_results.values() 
    if isinstance(r, dict) and 'error' not in r
)

print(f"\n{'='*60}")
print(f"  ✅ Phase 4 Complete! All Training Finished!")
print(f"{'='*60}")
print(f"  Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
print(f"  Models trained: {len([r for r in all_results.values() if isinstance(r, dict) and 'error' not in r])}")
print(f"  Output directory: {Config.OUTPUT_DIR}")
print(f"\n  Saved files:")
for f in sorted(os.listdir(Config.OUTPUT_DIR)):
    fpath = os.path.join(Config.OUTPUT_DIR, f)
    if os.path.isfile(fpath):
        fsize = os.path.getsize(fpath) / 1e6
        print(f"    📄 {f} ({fsize:.1f} MB)")
    else:
        print(f"    📁 {f}/")
