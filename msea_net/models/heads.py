from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Evidential Deep Learning (EDL) Classification Head.
    
    Parameterizes the classification output as the parameters of a
    Dirichlet distribution over categorical probabilities, separating
    data-inherent ambiguity from model epistemic uncertainty.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Softplus(),  # Ensures non-negative evidence e_k >= 0
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = self.head(x)
        alpha = evidence + 1.0  # Dirichlet concentration parameters alpha_k = e_k + 1
        S = alpha.sum(dim=1, keepdim=True)  # Dirichlet strength S = sum(alpha_k)
        probs = alpha / S  # Expected categorical probabilities
        uncertainty = self.num_classes / S.squeeze(1)  # Epistemic uncertainty u = K / S
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty,
            'strength': S.squeeze(1),
        }


class StandardHead(nn.Module):
    """
    Standard linear classification head with Softmax output.
    Used for baseline models and ablation study variants.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.head(x)
        probs = F.softmax(logits, dim=1)
        return {
            'logits': logits,
            'probs': probs,
        }
