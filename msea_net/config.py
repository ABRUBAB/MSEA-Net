import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    Falls back to configs/default.yaml if no path is provided.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def find_dataset_path(preferred_path: Optional[str] = None) -> str:
    """
    Find the Kvasir-v2 dataset directory across common local and cloud paths.
    """
    candidates = [
        preferred_path,
        "./data/kvasir-dataset-v2",
        "../data/kvasir-dataset-v2",
        "/kaggle/input/kvasir-v2-a-gastrointestinal-tract-dataset",
        "/kaggle/input/kvasir-dataset/kvasir-dataset-v2",
        "/kaggle/input/kvasir-dataset",
    ]
    
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.path.isdir(candidate):
            # Check for subdirectories (classes)
            subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            if len(subdirs) >= 7:
                return candidate
    
    return preferred_path or "./data/kvasir-dataset-v2"
