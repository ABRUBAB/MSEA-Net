import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention using both average-pooling and max-pooling.
    Identifies 'WHAT' semantic feature channels are informative.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        avg_out = self.shared_mlp(avg_pool)
        
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        max_out = self.shared_mlp(max_pool)
        
        attention = torch.sigmoid(avg_out + max_out)
        return attention.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention using channel-wise statistics.
    Identifies 'WHERE' spatial regions (e.g. lesion boundaries) require focus.
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
    Adaptive Channel-Spatial Attention (ACSA) module.
    
    Novel properties:
    1. Parallel computation of channel and spatial attention maps.
    2. Dynamically balances both mechanisms using a learnable gating parameter (alpha).
    3. Learned independently at each feature hierarchy scale.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        # Learnable gate parameter initialized to 0.5 (logits = 0.0)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        gate = torch.sigmoid(self.gate)
        
        # Element-wise modulation by gated combination
        attended = x * (gate * ca + (1.0 - gate) * sa)
        return attended
