from typing import Dict, List, Optional
import torch
import torch.nn as nn
import timm

from .attention import ACSA
from .fusion import GatedFeatureFusion
from .heads import EvidentialHead, StandardHead


class MSEANet(nn.Module):
    """
    MSEA-Net: Multi-Scale Evidential Attention Network
    
    Combines:
    1. EfficientNetV2-S hierarchical backbone.
    2. Adaptive Channel-Spatial Attention (ACSA) applied to multi-scale features.
    3. Gated Feature Fusion (GFF) for dynamic multi-scale representation aggregation.
    4. Evidential Deep Learning (EDL) head for principled uncertainty quantification.
    
    Supports ablation configurations:
    - use_multiscale: Toggle multi-scale extraction vs single top-level feature.
    - use_attention: Toggle ACSA modules.
    - use_gff: Toggle GFF vs static concatenation.
    - use_edl: Toggle evidential head vs standard cross-entropy head.
    """
    def __init__(
        self,
        num_classes: int = 8,
        backbone_name: str = 'tf_efficientnetv2_s',
        use_multiscale: bool = True,
        use_attention: bool = True,
        use_gff: bool = True,
        use_edl: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        self.use_gff = use_gff
        self.use_edl = use_edl

        # 1. Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        # Dynamic channel extraction
        feature_info = self.backbone.feature_info.channels()
        self.ms_dims = feature_info[-3:]  # Last 3 pyramid stages
        self.last_dim = feature_info[-1]

        # 2. ACSA Attention Modules
        if self.use_attention:
            self.acsa_modules = nn.ModuleList([
                ACSA(dim) for dim in self.ms_dims
            ])

        # 3. Global Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 4. Feature Aggregation
        if self.use_multiscale:
            if self.use_gff:
                self.fusion = GatedFeatureFusion(
                    in_dims=self.ms_dims,
                    out_dim=256
                )
                feat_dim = 256
            else:
                total_concat_dim = sum(self.ms_dims)
                self.simple_proj = nn.Sequential(
                    nn.Linear(total_concat_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                )
                feat_dim = 256
        else:
            self.single_proj = nn.Sequential(
                nn.Linear(self.last_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            feat_dim = 256

        # 5. Output Head
        if self.use_edl:
            self.classifier = EvidentialHead(feat_dim, num_classes)
        else:
            self.classifier = StandardHead(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)  # List of hierarchical feature maps
        
        gate_weights = None
        if self.use_multiscale:
            selected_features = features[-3:]
            
            # Apply ACSA attention
            if self.use_attention:
                attended_features = [
                    acsa(f) for acsa, f in zip(self.acsa_modules, selected_features)
                ]
            else:
                attended_features = selected_features
            
            # Pool each scale
            pooled = [self.gap(f).flatten(1) for f in attended_features]
            
            # Fuse features
            if self.use_gff:
                fused_features, gate_weights = self.fusion(pooled)
            else:
                concat = torch.cat(pooled, dim=1)
                fused_features = self.simple_proj(concat)
        else:
            # Single-scale top-level feature map
            top_feat = features[-1]
            if self.use_attention:
                # Use ACSA for the top stage
                top_acsa = self.acsa_modules[-1] if hasattr(self, 'acsa_modules') else ACSA(self.last_dim).to(x.device)
                top_feat = top_acsa(top_feat)
            
            pooled = self.gap(top_feat).flatten(1)
            fused_features = self.single_proj(pooled)

        # Output predictions
        output = self.classifier(fused_features)
        output['features'] = fused_features
        if gate_weights is not None:
            output['gate_weights'] = gate_weights
            
        return output
