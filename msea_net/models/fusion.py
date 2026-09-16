from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFeatureFusion(nn.Module):
    """
    Gated Feature Fusion (GFF) module.
    
    Dynamically weighs and aggregates multi-scale hierarchical feature maps
    via a lightweight gating network rather than static concatenation.
    """
    def __init__(self, in_dims: List[int], out_dim: int = 256):
        super().__init__()
        self.num_scales = len(in_dims)
        total_dim = sum(in_dims)
        
        # Project each scale feature to identical dimensionality
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ) for dim in in_dims
        ])
        
        # Scale importance gating network
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, self.num_scales * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_scales * 4, self.num_scales),
        )
        
        # Final output projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of tensors [B, D_i] from different pyramid scales.
        Returns:
            fused: [B, out_dim] aggregated representation.
            gate_weights: [B, num_scales] learned scale attention distribution.
        """
        projected = [proj(f) for proj, f in zip(self.scale_projections, features)]
        
        # Compute dynamic scale importance weights
        concat_raw = torch.cat(features, dim=1)
        gate_logits = self.gate_network(concat_raw)
        gate_weights = F.softmax(gate_logits, dim=1)  # [B, num_scales]
        
        # Softmax-weighted aggregation
        stacked = torch.stack(projected, dim=1)  # [B, num_scales, out_dim]
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, out_dim]
        
        out = self.fusion_proj(fused)
        return out, gate_weights
