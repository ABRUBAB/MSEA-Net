"""
MSEA-Net: Multi-Scale Evidential Attention Network
Author: Abdullah Rubab (rubab2712@gmail.com)
"""

__version__ = "1.0.0"
__author__ = "Abdullah Rubab"

from .config import load_config, find_dataset_path

# Optional/lazy exports for core DL models when torch/timm are installed
try:
    from .models.msea_net import MSEANet
    from .losses.edl import edl_loss
    __all__ = ["MSEANet", "edl_loss", "load_config", "find_dataset_path"]
except ImportError:
    __all__ = ["load_config", "find_dataset_path"]
