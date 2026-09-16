import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..losses.edl import edl_loss, standard_loss
from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.calibration import compute_ece

# AMP setup
try:
    from torch.amp import autocast as _autocast_base, GradScaler
    def autocast(dtype=torch.float16):
        return _autocast_base('cuda', dtype=dtype)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


class Trainer:
    """
    Modular training and validation manager for MSEA-Net and baseline models.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 8,
        lr: float = 3e-4,
        lr_backbone: float = 3e-5,
        weight_decay: float = 1e-4,
        use_edl: bool = True,
        edl_lambda_max: float = 0.1,
        edl_annealing: str = 'linear',
        use_amp: bool = True,
        save_dir: str = './msea_net_results',
        model_tag: str = 'msea_net_seed42',
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.use_edl = use_edl
        self.edl_lambda_max = edl_lambda_max
        self.edl_annealing = edl_annealing
        self.use_amp = use_amp and (device.type == 'cuda')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_tag = model_tag

        # Differential optimizer
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
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
        self.optimizer = optim.AdamW(param_groups)
        self.scaler = GradScaler(enabled=self.use_amp)

    def train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets, _ in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if self.use_edl and 'alpha' in outputs:
                    loss = edl_loss(
                        outputs, targets,
                        num_classes=self.num_classes,
                        epoch=epoch,
                        total_epochs=total_epochs,
                        lambda_max=self.edl_lambda_max,
                        annealing=self.edl_annealing
                    )
                else:
                    loss = standard_loss(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(targets)
            preds = outputs['probs'].argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        return {
            'loss': total_loss / max(total, 1),
            'acc': correct / max(total, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        all_uncertainties = []

        for images, targets, _ in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if self.use_edl and 'alpha' in outputs:
                    loss = edl_loss(
                        outputs, targets,
                        num_classes=self.num_classes,
                        epoch=30, total_epochs=30,
                        lambda_max=self.edl_lambda_max
                    )
                else:
                    loss = standard_loss(outputs, targets)

            total_loss += loss.item() * len(targets)
            probs = outputs['probs'].cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(np.argmax(probs, axis=1))

            if 'uncertainty' in outputs:
                all_uncertainties.append(outputs['uncertainty'].cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0) if all_uncertainties else None

        metrics = compute_classification_metrics(all_targets, all_preds, all_probs, self.num_classes)
        metrics['loss'] = total_loss / max(len(all_targets), 1)
        metrics['ece'] = compute_ece(all_probs, all_targets)

        return metrics, all_targets, all_probs, all_uncertainties

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 30,
        patience: int = 10,
    ) -> Dict[str, Any]:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        
        best_val_f1 = -1.0
        best_epoch = 0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        checkpoint_path = self.save_dir / f"{self.model_tag}_best.pth"

        print(f"🚀 Starting training: {self.model_tag} ({epochs} epochs) on {self.device}")
        start_time = time.time()

        for epoch in range(epochs):
            train_res = self.train_epoch(train_loader, epoch, epochs)
            val_metrics, _, _, _ = self.evaluate(val_loader)
            scheduler.step()

            history['train_loss'].append(train_res['loss'])
            history['train_acc'].append(train_res['acc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_macro'])

            # Early stopping and checkpointing
            current_f1 = val_metrics['f1_macro']
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                improved = "★ (Saved Best)"
            else:
                patience_counter += 1
                improved = ""

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
                      f"Loss: {train_res['loss']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                      f"Val F1: {val_metrics['f1_macro']*100:.2f}% {improved}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (Best epoch: {best_epoch+1})")
                break

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed/60:.2f} minutes.")

        # Load best weights for test evaluation
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = None
        if test_loader is not None:
            test_metrics, targets, probs, unc = self.evaluate(test_loader)
            print(f"📊 Final Test Results ({self.model_tag}): "
                  f"Acc: {test_metrics['accuracy']*100:.2f}% | "
                  f"F1: {test_metrics['f1_macro']*100:.2f}% | "
                  f"AUC: {test_metrics['auc_macro']:.4f} | "
                  f"ECE: {test_metrics['ece']:.4f}")

            # Save predictions
            np.savez_compressed(
                self.save_dir / f"{self.model_tag}_results.npz",
                targets=targets,
                probs=probs,
                uncertainties=unc if unc is not None else np.zeros_like(targets),
                metrics=test_metrics
            )

        # Save history
        with open(self.save_dir / f"{self.model_tag}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return {
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'test_metrics': test_metrics,
            'history': history,
            'checkpoint_path': str(checkpoint_path),
        }


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 3e-4,
    lr_backbone: float = 3e-5,
    weight_decay: float = 1e-4,
    use_edl: bool = True,
    save_dir: str = './msea_net_results',
    model_tag: str = 'model',
) -> Dict[str, Any]:
    """Helper functional wrapper for single model training."""
    trainer = Trainer(
        model=model,
        device=device,
        lr=lr,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
        use_edl=use_edl,
        save_dir=save_dir,
        model_tag=model_tag
    )
    return trainer.fit(train_loader, val_loader, test_loader, epochs=epochs)
