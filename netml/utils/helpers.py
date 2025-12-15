"""
Helper Utilities
================

Common utility functions for the NetML package.
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'device': 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if info['cuda_available']:
        info['device'] = 'cuda'
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    elif info['mps_available']:
        info['device'] = 'mps'
    
    return info


def save_json(data: Any, path: Path, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    print(f"  ✓ Saved: {path}")


def load_json(path: Path) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def format_number(n: int) -> str:
    """Format number with commas for readability."""
    return f"{n:,}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def print_section(title: str, char: str = '=', width: int = 60):
    """Print a section header."""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def print_metrics(metrics: Dict[str, float], title: Optional[str] = None):
    """Pretty print metrics dictionary."""
    if title:
        print(f"\n{title}")
        print("-" * 40)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'acc' in key.lower() or 'accuracy' in key.lower():
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


class ProgressTracker:
    """Simple progress tracker for training."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, key: str, value: float):
        """Add value to metric history."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def get_best(self, key: str, mode: str = 'max') -> float:
        """Get best value for a metric."""
        if key not in self.metrics:
            return None
        
        if mode == 'max':
            return max(self.metrics[key])
        return min(self.metrics[key])
    
    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return None
        return self.metrics[key][-1]
    
    def to_dict(self) -> Dict[str, list]:
        """Export as dictionary."""
        return dict(self.metrics)
