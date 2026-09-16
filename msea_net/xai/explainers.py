from typing import Optional, List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class ModelProbWrapper(nn.Module):
    """Wraps model to return probability tensor directly for Grad-CAM."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out['probs']


def generate_gradcam_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    target_class: Optional[int] = None,
    method: str = 'gradcam++',
) -> np.ndarray:
    """
    Generate Class Activation Map overlay on original RGB image.
    
    Args:
        model: Trained MSEA-Net model
        image_tensor: [1, 3, H, W] normalized input tensor
        original_rgb: [H, W, 3] float32 image in range [0, 1]
        target_class: Target integer class index (if None, highest probability class is used)
        method: 'gradcam' or 'gradcam++'
    """
    if not GRADCAM_AVAILABLE:
        raise ImportError("pytorch-grad-cam is not installed. Run: pip install grad-cam")

    wrapper = ModelProbWrapper(model).eval()

    # Identify target layer: ACSA top module or backbone conv
    if hasattr(model, 'acsa_modules') and len(model.acsa_modules) > 0:
        target_layer = [model.acsa_modules[-1].spatial_attention.conv]
    elif hasattr(model, 'backbone'):
        target_layer = [list(model.backbone.children())[-1]]
    else:
        target_layer = [list(model.children())[0]]

    cam_cls = GradCAMPlusPlus if method.lower() == 'gradcam++' else GradCAM

    with cam_cls(model=wrapper, target_layers=target_layer) as cam:
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)

    return visualization


def generate_umap_projection(
    features: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Fit UMAP 2D dimensional reduction on extracted deep latent representations.
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is not installed. Run: pip install umap-learn")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric='cosine'
    )
    embedding = reducer.fit_transform(features)
    return embedding, reducer
