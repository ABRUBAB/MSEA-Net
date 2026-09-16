import math
from typing import Dict
import torch
import torch.nn.functional as F


def kl_divergence_dirichlet(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Closed-form Kullback-Leibler (KL) divergence between two Dirichlet distributions:
    KL(Dir(alpha) || Dir(beta))
    
    Args:
        alpha: Predicted Dirichlet parameters [B, K]
        beta: Prior Dirichlet parameters [B, K] (typically uniform 1.0)
    Returns:
        Tensor of shape [B]
    """
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)

    kl = (
        torch.lgamma(S_alpha)
        - torch.lgamma(S_beta)
        - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=1, keepdim=True)
        + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True)
    )
    return kl.squeeze(1)


def edl_loss(
    output: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    num_classes: int = 8,
    epoch: int = 0,
    total_epochs: int = 30,
    lambda_max: float = 0.1,
    annealing: str = 'linear',
) -> torch.Tensor:
    """
    Evidential Deep Learning loss (Sensoy et al., NeurIPS 2018).
    
    Combines:
    1. Expected Mean Squared Error (Bayes Risk) under the Dirichlet distribution.
    2. Variance term quantifying uncertainty penalty.
    3. Annealed KL divergence penalizing evidence on incorrect classes against a uniform prior.
    """
    alpha = output['alpha']
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S

    # One-hot representation of ground-truth targets
    y_onehot = F.one_hot(targets, num_classes).float()

    # 1. Bayes Risk (MSE + Dirichlet Variance)
    mse = (probs - y_onehot) ** 2
    variance = alpha * (S - alpha) / (S ** 2 * (S + 1))
    bayes_risk = (mse + variance).sum(dim=1).mean()

    # 2. KL Divergence Regularization
    # Retain evidence only on counterfactual/incorrect classes
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    beta = torch.ones_like(alpha)  # Flat uniform prior Dir(1, 1, ..., 1)
    kl = kl_divergence_dirichlet(alpha_tilde, beta).mean()

    # Annealing coefficient calculation
    if annealing == 'linear':
        lambda_t = lambda_max * min(1.0, epoch / max(total_epochs * 0.5, 1.0))
    elif annealing == 'cosine':
        lambda_t = lambda_max * 0.5 * (1.0 - math.cos(math.pi * epoch / max(total_epochs, 1)))
    else:
        lambda_t = lambda_max

    total_loss = bayes_risk + lambda_t * kl
    return total_loss


def standard_loss(
    output: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Cross-entropy loss with optional label smoothing for softmax baselines.
    """
    if 'logits' in output:
        logits = output['logits']
    else:
        # If model outputs probabilities without logits, take log
        probs = output['probs'].clamp(min=1e-7)
        logits = torch.log(probs)

    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
