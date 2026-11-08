"""
=============================================================================
MSEA-Net: Multi-Scale Evidential Attention Network
Phase 3: Training Utilities — Losses, Training Loop, Evaluation
=============================================================================
PASTE THIS AS THE THIRD CELL in your Kaggle notebook.
Requires Phase 1 and Phase 2 to be executed first.
=============================================================================
"""

# ============================================================
# SECTION 1: Evidential Deep Learning Loss
# ============================================================

def kl_divergence_dirichlet(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two Dirichlet distributions.
    KL(Dir(alpha) || Dir(beta))
    """
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)
    
    kl = (
        torch.lgamma(S_alpha) - torch.lgamma(S_beta)
        - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=1, keepdim=True)
        + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True)
    )
    return kl.squeeze(1)


def edl_loss(
    output: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    num_classes: int,
    epoch: int,
    total_epochs: int,
    lambda_max: float = 0.1,
    annealing: str = 'linear',
) -> torch.Tensor:
    """
    Evidential Deep Learning loss (Sensoy et al., 2018).
    
    Combines:
    1. Bayes risk (MSE + variance of Dirichlet)
    2. KL divergence regularization (annealed)
    
    Args:
        output: Model output dict with 'alpha', 'probs'
        targets: Class labels [B]
        num_classes: Number of classes
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        lambda_max: Maximum KL weight
        annealing: 'linear' or 'cosine'
    """
    alpha = output['alpha']
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    
    # One-hot encode targets
    y_onehot = F.one_hot(targets, num_classes).float()
    
    # --- Bayes Risk ---
    # MSE part
    mse = (probs - y_onehot) ** 2
    # Variance part (Dirichlet variance)
    var = alpha * (S - alpha) / (S ** 2 * (S + 1))
    bayes_risk = (mse + var).sum(dim=1).mean()
    
    # --- KL Regularization ---
    # Remove evidence for the correct class before KL
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    beta = torch.ones_like(alpha)  # Uniform Dirichlet prior
    kl = kl_divergence_dirichlet(alpha_tilde, beta).mean()
    
    # Annealing coefficient
    if annealing == 'linear':
        lambda_t = lambda_max * min(1.0, epoch / max(total_epochs * 0.5, 1))
    elif annealing == 'cosine':
        import math
        lambda_t = lambda_max * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    else:
        lambda_t = lambda_max
    
    loss = bayes_risk + lambda_t * kl
    return loss


def standard_loss(
    output: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Standard cross-entropy loss with label smoothing."""
    logits = output['logits']
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


# ============================================================
# SECTION 2: Optimizer Creation
# ============================================================

def create_optimizer(model: nn.Module, lr: float, lr_backbone: float, 
                     weight_decay: float) -> optim.Optimizer:
    """
    Create AdamW optimizer with differential learning rates.
    Backbone gets lower lr (pretrained), head gets higher lr.
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr, 'weight_decay': weight_decay},
    ]
    
    optimizer = optim.AdamW(param_groups)
    return optimizer


# ============================================================
# SECTION 3: Training & Evaluation Functions
# ============================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_edl: bool = True,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Train for one epoch. Returns metrics dict."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        if scaler is not None and Config.USE_AMP:
            with autocast(dtype=torch.float16):
                output = model(images)
                
                if use_edl:
                    loss = edl_loss(
                        output, targets, Config.NUM_CLASSES,
                        epoch, total_epochs,
                        Config.EDL_LAMBDA_MAX, Config.EDL_ANNEALING
                    )
                else:
                    loss = standard_loss(output, targets, Config.LABEL_SMOOTHING)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            
            if use_edl:
                loss = edl_loss(
                    output, targets, Config.NUM_CLASSES,
                    epoch, total_epochs,
                    Config.EDL_LAMBDA_MAX, Config.EDL_ANNEALING
                )
            else:
                loss = standard_loss(output, targets, Config.LABEL_SMOOTHING)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        preds = output['probs'].argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    if scheduler is not None:
        scheduler.step()
    
    # Compute metrics
    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1,
        'lr': optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_edl: bool = True,
    epoch: int = 0,
    total_epochs: int = 30,
) -> Dict[str, any]:
    """
    Evaluate model on a dataset.
    Returns comprehensive metrics + predictions for further analysis.
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_targets = []
    all_features = []
    all_uncertainties = []
    total_loss = 0.0
    total_samples = 0
    
    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets_dev = targets.to(device, non_blocking=True)
        
        if Config.USE_AMP:
            with autocast(dtype=torch.float16):
                output = model(images)
        else:
            output = model(images)
        
        probs = output['probs'].float().cpu().numpy()
        preds = probs.argmax(axis=1)
        features = output['features'].float().cpu().numpy()
        
        # Compute loss
        with torch.no_grad():
            if use_edl and 'alpha' in output:
                batch_loss = edl_loss(
                    output, targets_dev, Config.NUM_CLASSES,
                    epoch, total_epochs,
                    Config.EDL_LAMBDA_MAX, Config.EDL_ANNEALING
                )
            elif 'logits' in output:
                batch_loss = standard_loss(output, targets_dev, Config.LABEL_SMOOTHING)
            else:
                batch_loss = torch.tensor(0.0)
        total_loss += batch_loss.item() * images.size(0)
        total_samples += images.size(0)
        
        all_probs.append(probs)
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        all_features.append(features)
        
        if 'uncertainty' in output:
            all_uncertainties.extend(output['uncertainty'].float().cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_features = np.concatenate(all_features, axis=0)
    all_uncertainties = np.array(all_uncertainties) if all_uncertainties else None
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # AUC (one-vs-rest, macro-averaged)
    try:
        auc_macro = roc_auc_score(
            all_targets, all_probs, multi_class='ovr', average='macro'
        )
    except ValueError:
        auc_macro = 0.0
    
    # ECE (Expected Calibration Error)
    ece = compute_ece(all_probs, all_targets, n_bins=15)
    
    # Brier score (multi-class)
    y_onehot = np.eye(Config.NUM_CLASSES)[all_targets]
    brier = np.mean(np.sum((all_probs - y_onehot) ** 2, axis=1))
    
    return {
        'loss': total_loss / max(total_samples, 1),
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'auc_macro': auc_macro,
        'ece': ece,
        'brier': brier,
        'probs': all_probs,
        'preds': all_preds,
        'targets': all_targets,
        'features': all_features,
        'uncertainties': all_uncertainties,
        'classification_report': classification_report(
            all_targets, all_preds, 
            target_names=Config.CLASS_NAMES, 
            output_dict=True, zero_division=0
        ),
    }


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error.
    Measures how well predicted confidence matches actual accuracy.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


# ============================================================
# SECTION 4: Full Training Pipeline
# ============================================================

def train_model(
    model_type: str,
    seed: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = None,
    save_dir: str = None,
) -> Dict[str, any]:
    """
    Complete training pipeline for one model + one seed.
    
    Returns:
        Dict with training history + final test metrics + predictions
    """
    if epochs is None:
        epochs = Config.EPOCHS
    if save_dir is None:
        save_dir = Config.OUTPUT_DIR
    
    print(f"\n{'='*60}")
    print(f"  Training: {model_type} | Seed: {seed}")
    print(f"{'='*60}")
    
    # Set seed
    set_seed(seed)
    
    # Create model
    model = create_model(model_type, Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Determine if using EDL
    use_edl = hasattr(model, 'use_edl') and model.use_edl if hasattr(model, 'use_edl') else False
    
    # For baseline models, check for EDL differently
    if model_type.startswith('msea') and 'no_edl' not in model_type:
        use_edl = True
    
    # Create optimizer
    if hasattr(model, 'backbone'):
        optimizer = create_optimizer(model, Config.LR, Config.LR_BACKBONE, Config.WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Mixed precision scaler (compatible with PyTorch 2.0+ and 2.4+)
    try:
        scaler = GradScaler('cuda') if Config.USE_AMP else None
    except TypeError:
        scaler = GradScaler() if Config.USE_AMP else None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [], 'val_ece': [],
        'lr': [],
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, epochs, use_edl, scaler
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, use_edl, epoch, epochs)
        
        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_auc'].append(val_metrics['auc_macro'])
        history['val_ece'].append(val_metrics['ece'])
        history['lr'].append(train_metrics['lr'])
        
        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - epoch_start
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f} | "
                  f"Val AUC: {val_metrics['auc_macro']:.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n  ⏱️  Training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  🏆 Best epoch: {best_epoch+1} (Val F1: {best_val_f1:.4f})")
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model = model.to(device)
    test_metrics = evaluate(model, test_loader, device, use_edl, best_epoch, epochs)
    
    print(f"\n  📊 Test Results:")
    print(f"     Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"     F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"     AUC (macro): {test_metrics['auc_macro']:.4f}")
    print(f"     ECE:       {test_metrics['ece']:.4f}")
    print(f"     Brier:     {test_metrics['brier']:.4f}")
    
    # Save model checkpoint
    checkpoint_path = os.path.join(save_dir, f"{model_type}_seed{seed}_best.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'model_type': model_type,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_metrics': {k: v for k, v in test_metrics.items() 
                        if k not in ['probs', 'preds', 'targets', 'features', 'uncertainties', 'classification_report']},
        'history': history,
        'config': {
            'epochs': epochs,
            'batch_size': Config.BATCH_SIZE,
            'lr': Config.LR,
            'lr_backbone': Config.LR_BACKBONE,
        },
    }, checkpoint_path)
    print(f"  💾 Model saved: {checkpoint_path}")
    
    # Save predictions (for later evaluation)
    results_path = os.path.join(save_dir, f"{model_type}_seed{seed}_results.npz")
    save_dict = {
        'probs': test_metrics['probs'],
        'preds': test_metrics['preds'],
        'targets': test_metrics['targets'],
        'features': test_metrics['features'],
    }
    if test_metrics['uncertainties'] is not None:
        save_dict['uncertainties'] = test_metrics['uncertainties']
    np.savez_compressed(results_path, **save_dict)
    print(f"  💾 Results saved: {results_path}")
    
    # Cleanup GPU memory
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'model_type': model_type,
        'seed': seed,
        'history': history,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'training_time': total_time,
    }


print("\n" + "=" * 60)
print("  MSEA-Net: Phase 3 — Training Utilities")
print("=" * 60)
print("✅ EDL loss function ready")
print("✅ Standard CE loss ready")
print("✅ Training loop ready")
print("✅ Evaluation pipeline ready")
print("✅ ECE computation ready")
print("✅ Checkpoint saving ready")
print(f"\n✅ Phase 3 Complete!")
