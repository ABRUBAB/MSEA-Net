"""
=============================================================================
MSEA-Net: Multi-Scale Evidential Attention Network
Phase 2: Model Architecture
=============================================================================
PASTE THIS AS THE SECOND CELL in your Kaggle notebook.
Requires Phase 1 to be executed first.
=============================================================================
"""

# ============================================================
# SECTION 1: Adaptive Channel-Spatial Attention (ACSA) Module
# ============================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention using both average-pool and max-pool.
    Learns WHICH feature channels are important.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)  # Ensure minimum width
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Average pool
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        avg_out = self.shared_mlp(avg_pool)
        # Max pool
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        max_out = self.shared_mlp(max_pool)
        # Combine
        attention = torch.sigmoid(avg_out + max_out)
        return attention.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention using channel-wise statistics.
    Learns WHERE to focus in the image.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))
        return attention


class ACSA(nn.Module):
    """
    Adaptive Channel-Spatial Attention (ACSA) — NOVEL MODULE.
    
    Unlike standard CBAM (serial: channel → spatial), ACSA uses:
    1. PARALLEL channel + spatial attention paths
    2. A LEARNABLE GATE (α) to adaptively balance both paths
    3. The gate is learned per-scale, allowing different scales to 
       emphasize different attention types
    
    This is the first novel contribution of MSEA-Net.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Learnable gate parameter (initialized to 0.5 = equal weight)
        # This is what makes ACSA novel: the gate learns the optimal
        # balance between channel and spatial attention per scale
        self.gate = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel attention computation
        ca = self.channel_attention(x)  # [B, C, 1, 1]
        sa = self.spatial_attention(x)  # [B, 1, H, W]
        
        # Learnable gated fusion
        gate = torch.sigmoid(self.gate)
        
        # Apply: x * (gate * channel_attn + (1-gate) * spatial_attn)
        out = x * (gate * ca + (1.0 - gate) * sa)
        
        return out


# ============================================================
# SECTION 2: Gated Feature Fusion (GFF) Module
# ============================================================

class GatedFeatureFusion(nn.Module):
    """
    Gated Feature Fusion (GFF) — NOVEL MODULE.
    
    Instead of simple concatenation or averaging of multi-scale features,
    GFF learns importance weights for each scale via a gating network.
    
    This is the second novel contribution of MSEA-Net.
    """
    def __init__(self, in_dims: List[int], out_dim: int = 256):
        super().__init__()
        self.num_scales = len(in_dims)
        total_dim = sum(in_dims)
        
        # Project each scale to same dimension
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ) for dim in in_dims
        ])
        
        # Gating network: learns which scale is most important
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, self.num_scales * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_scales * 4, self.num_scales),
        )
        
        # Final projection after weighted fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, D_i] tensors from different scales
        Returns:
            [B, out_dim] fused feature vector
        """
        # Project each scale to same dimension
        projected = [proj(f) for proj, f in zip(self.scale_projections, features)]
        
        # Compute gate weights from concatenated raw features
        concat = torch.cat(features, dim=1)
        gate_logits = self.gate_network(concat)
        gate_weights = F.softmax(gate_logits, dim=1)  # [B, num_scales]
        
        # Weighted sum of projected features
        stacked = torch.stack(projected, dim=1)  # [B, num_scales, out_dim]
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [B, num_scales, 1]
        fused = (stacked * gate_weights_expanded).sum(dim=1)  # [B, out_dim]
        
        # Final projection
        out = self.fusion_proj(fused)
        
        return out, gate_weights


# ============================================================
# SECTION 3: Evidential Classification Head
# ============================================================

class EvidentialHead(nn.Module):
    """
    Evidential Deep Learning classification head.
    Outputs Dirichlet parameters instead of logits.
    
    Based on Sensoy et al. (2018) "Evidential Deep Learning to Quantify
    Classification Uncertainty"
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Softplus(),  # Ensures non-negative evidence
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = self.head(x)
        alpha = evidence + 1.0  # Dirichlet concentration parameters
        S = alpha.sum(dim=1, keepdim=True)  # Dirichlet strength
        probs = alpha / S  # Expected probabilities
        uncertainty = self.num_classes / S.squeeze(1)  # Epistemic uncertainty
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty,
        }


class StandardHead(nn.Module):
    """Standard softmax classification head (for baselines and ablation)."""
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


# ============================================================
# SECTION 4: MSEA-Net (Main Model)
# ============================================================

class MSEANet(nn.Module):
    """
    MSEA-Net: Multi-Scale Evidential Attention Network.
    
    Architecture:
    1. EfficientNetV2-S backbone (pretrained, fine-tuned)
    2. Multi-scale feature extraction (3 stages)
    3. ACSA attention at each scale
    4. Gated Feature Fusion
    5. Evidential classification head
    
    Supports ablation via flags:
    - use_multiscale: Enable/disable multi-scale features
    - use_attention: Enable/disable ACSA modules
    - use_gff: Enable/disable gated fusion
    - use_edl: Enable/disable evidential head
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
        
        # ---------- Backbone ----------
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True,  # Return intermediate features
        )
        
        # Get feature dimensions dynamically (robust across timm versions)
        feature_info = self.backbone.feature_info.channels()
        print(f"   📐 Backbone feature dims: {feature_info}")
        
        # We use the last 3 stages for multi-scale
        self.ms_dims = feature_info[-3:]  # e.g., [64, 160, 256]
        self.last_dim = feature_info[-1]  # e.g., 256
        
        # ---------- ACSA Attention Modules ----------
        if use_attention:
            self.acsa_modules = nn.ModuleList([
                ACSA(dim) for dim in self.ms_dims
            ])
        
        # ---------- Global Average Pooling ----------
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # ---------- Feature Fusion ----------
        if use_multiscale:
            total_concat_dim = sum(self.ms_dims)
            
            if use_gff:
                self.fusion = GatedFeatureFusion(
                    in_dims=self.ms_dims, 
                    out_dim=256
                )
                feat_dim = 256
            else:
                # Simple concatenation + projection (ablation variant)
                self.fusion = nn.Sequential(
                    nn.Linear(total_concat_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                )
                feat_dim = 256
        else:
            # Single-scale: only last stage
            self.fusion = nn.Sequential(
                nn.Linear(self.last_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            feat_dim = 256
        
        # ---------- Classification Head ----------
        if use_edl:
            self.classifier = EvidentialHead(feat_dim, num_classes)
        else:
            self.classifier = StandardHead(feat_dim, num_classes)
        
        # ---------- Print model info ----------
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   📊 Total params: {total_params:,}")
        print(f"   📊 Trainable params: {trainable_params:,}")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the fused feature vector (for XAI/UMAP)."""
        features = self.backbone(x)
        ms_features = features[-3:]  # Last 3 stages
        
        if self.use_attention:
            ms_features = [acsa(f) for acsa, f in zip(self.acsa_modules, ms_features)]
        
        pooled = [self.gap(f).flatten(1) for f in ms_features]
        
        if self.use_multiscale:
            if self.use_gff:
                fused, _ = self.fusion(pooled)
            else:
                concat = torch.cat(pooled, dim=1)
                fused = self.fusion(concat)
        else:
            fused = self.fusion(pooled[-1])
        
        return fused
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns dict with:
        - 'probs': predicted probabilities [B, C]
        - 'features': fused features [B, 256]
        - If EDL: 'evidence', 'alpha', 'uncertainty'
        - If standard: 'logits'
        - If GFF: 'gate_weights'
        """
        # Extract multi-scale features from backbone
        features = self.backbone(x)
        ms_features = features[-3:]  # Last 3 stages
        
        # Apply ACSA attention at each scale
        if self.use_attention:
            ms_features = [acsa(f) for acsa, f in zip(self.acsa_modules, ms_features)]
        
        # Global average pooling
        pooled = [self.gap(f).flatten(1) for f in ms_features]
        
        # Feature fusion
        gate_weights = None
        if self.use_multiscale:
            if self.use_gff:
                fused, gate_weights = self.fusion(pooled)
            else:
                concat = torch.cat(pooled, dim=1)
                fused = self.fusion(concat)
        else:
            fused = self.fusion(pooled[-1])
        
        # Classification
        output = self.classifier(fused)
        output['features'] = fused
        
        if gate_weights is not None:
            output['gate_weights'] = gate_weights
        
        return output


# ============================================================
# SECTION 5: Baseline Models
# ============================================================

class BaselineModel(nn.Module):
    """
    Generic baseline model using different backbones.
    Uses standard softmax classification (no EDL, no attention, no multi-scale).
    """
    
    BACKBONE_CONFIGS = {
        'vgg16': 'vgg16',
        'resnet50': 'resnet50',
        'densenet121': 'densenet121',
        'inception_v3': 'inception_v3',
        'mobilenetv3': 'mobilenetv3_large_100',
        'efficientnetv2s_plain': 'tf_efficientnetv2_s',
    }
    
    def __init__(self, backbone_name: str, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        
        self.backbone_name = backbone_name
        timm_name = self.BACKBONE_CONFIGS.get(backbone_name, backbone_name)
        
        # Create backbone with pooling + classification removed
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg',  # Add global average pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            feat_dim = self.backbone(dummy).shape[1]
        
        print(f"   📐 {backbone_name} feature dim: {feat_dim}")
        
        # Standard classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   📊 {backbone_name} total params: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        return {
            'logits': logits,
            'probs': probs,
            'features': features,
        }


# ============================================================
# SECTION 6: Model Factory Function
# ============================================================

def create_model(
    model_type: str = 'msea_net',
    num_classes: int = 8,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a model by name. Used for systematic training of all variants.
    
    Args:
        model_type: One of:
            - 'msea_net': Full proposed model (all components)
            - 'msea_no_multiscale': Ablation without multi-scale
            - 'msea_no_attention': Ablation without ACSA
            - 'msea_no_gff': Ablation without gated fusion
            - 'msea_no_edl': Ablation without evidential head
            - 'vgg16', 'resnet50', etc.: Baseline models
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
    
    Returns:
        PyTorch model
    """
    print(f"\n🔨 Creating model: {model_type}")
    
    if model_type == 'msea_net':
        return MSEANet(
            num_classes=num_classes,
            use_multiscale=True,
            use_attention=True,
            use_gff=True,
            use_edl=True,
            pretrained=pretrained,
        )
    
    elif model_type == 'msea_no_multiscale':
        return MSEANet(
            num_classes=num_classes,
            use_multiscale=False,
            use_attention=True,
            use_gff=False,
            use_edl=True,
            pretrained=pretrained,
        )
    
    elif model_type == 'msea_no_attention':
        return MSEANet(
            num_classes=num_classes,
            use_multiscale=True,
            use_attention=False,
            use_gff=True,
            use_edl=True,
            pretrained=pretrained,
        )
    
    elif model_type == 'msea_no_gff':
        return MSEANet(
            num_classes=num_classes,
            use_multiscale=True,
            use_attention=True,
            use_gff=False,
            use_edl=True,
            pretrained=pretrained,
        )
    
    elif model_type == 'msea_no_edl':
        return MSEANet(
            num_classes=num_classes,
            use_multiscale=True,
            use_attention=True,
            use_gff=True,
            use_edl=False,
            pretrained=pretrained,
        )
    
    elif model_type in BaselineModel.BACKBONE_CONFIGS:
        return BaselineModel(
            backbone_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# SECTION 7: Verify Model Creation
# ============================================================

print("\n" + "=" * 60)
print("  MSEA-Net: Phase 2 — Model Architecture")
print("=" * 60)

# Quick verification that the model builds correctly
print("\n🧪 Verifying MSEA-Net architecture...")
with torch.no_grad():
    _test_model = MSEANet(
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # Don't download weights for verify
    )
    _test_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    _test_output = _test_model(_test_input)
    
    print(f"\n   ✅ Input shape:  {_test_input.shape}")
    print(f"   ✅ Output keys:  {list(_test_output.keys())}")
    print(f"   ✅ Probs shape:  {_test_output['probs'].shape}")
    print(f"   ✅ Feature shape: {_test_output['features'].shape}")
    
    if 'uncertainty' in _test_output:
        print(f"   ✅ Uncertainty:  {_test_output['uncertainty'].shape}")
    if 'gate_weights' in _test_output:
        print(f"   ✅ Gate weights: {_test_output['gate_weights'].shape}")
        print(f"   ✅ Gate values:  {_test_output['gate_weights'][0].detach().numpy()}")
    
    # Check ACSA gate values
    for i, acsa in enumerate(_test_model.acsa_modules):
        gate_val = torch.sigmoid(acsa.gate).item()
        print(f"   ✅ ACSA gate {i}: {gate_val:.4f} (0.5 = equal channel/spatial)")
    
    del _test_model, _test_input, _test_output
    gc.collect()

print(f"\n✅ Phase 2 Complete! MSEA-Net architecture verified.")
print(f"   Available models: msea_net, msea_no_multiscale, msea_no_attention,")
print(f"   msea_no_gff, msea_no_edl, vgg16, resnet50, densenet121,")
print(f"   inception_v3, mobilenetv3, efficientnetv2s_plain")
