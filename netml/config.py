"""
Configuration Module
====================

Centralized configuration for the NetML classification system.
All hyperparameters, paths, and settings are defined here.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Path Configuration
# =============================================================================

# Project root (parent of netml package)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_DATA_FILE = DATA_DIR / "Master.json"

# Output paths
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_FIGURES_DIR = RESULTS_DIR / "paper_figures"

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Device Configuration
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # File path
    data_file: Path = DEFAULT_DATA_FILE
    
    # Sample limits
    limit_samples: int = 350_000
    val_split: float = 0.20  # 20% validation
    
    # Sequence fields from NetML-2020 dataset
    seq_fields: List[str] = field(default_factory=lambda: [
        'intervals_ccnt', 'pld_ccnt', 'hdr_ccnt',
        'rev_intervals_ccnt', 'rev_pld_ccnt', 'rev_hdr_ccnt',
        'ack_psh_rst_syn_fin_cnt', 'rev_ack_psh_rst_syn_fin_cnt'
    ])
    
    # Scalar fields
    scalar_fields: List[str] = field(default_factory=lambda: [
        'pr', 'rev_pld_max', 'rev_pld_mean', 'pld_mean',
        'pld_median', 'pld_distinct', 'time_length',
        'bytes_out', 'bytes_in', 'num_pkts_out', 'num_pkts_in',
        'src_port', 'dst_port'
    ])
    
    # Fields requiring log transformation
    log_transform_fields: List[str] = field(default_factory=lambda: [
        'src_port', 'dst_port', 'bytes_in', 'bytes_out', 
        'num_pkts_out', 'num_pkts_in'
    ])
    
    # Sequence padding length
    seq_length: int = 16
    
    # DataLoader settings
    batch_size: int = 512
    num_workers: int = 0  # Set to 0 for MPS compatibility


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class EncoderConfig:
    """Configuration for LSTM-Attention encoder."""
    
    # Input dimensions (derived from DataConfig)
    num_scalars: int = 13
    num_seq_features: int = 8
    seq_length: int = 16
    
    # LSTM settings
    lstm_hidden: int = 128
    lstm_layers: int = 3
    lstm_dropout: float = 0.3
    
    # Attention settings
    attention_heads: int = 4
    attention_dropout: float = 0.3
    
    # Projection head
    embedding_dim: int = 64  # Final embedding dimension
    projection_dropout: float = 0.3
    
    # Number of classes (set dynamically from data)
    num_classes: Optional[int] = None


@dataclass
class HybridConfig:
    """Configuration for hybrid classifiers (XGBoost, LightGBM, etc.)."""
    
    # Classifier type: 'xgboost', 'lightgbm', 'randomforest', 'ensemble'
    classifier_type: str = 'ensemble'
    
    # XGBoost parameters
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.1
    xgb_early_stopping: int = 30
    
    # LightGBM parameters
    lgb_n_estimators: int = 300
    lgb_max_depth: int = 10
    lgb_learning_rate: float = 0.05
    lgb_early_stopping: int = 30
    
    # Random Forest parameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    
    # Ensemble weights (auto-tuned if None)
    ensemble_weights: Optional[dict] = None
    
    # Class weight boosting for imbalanced classes
    boost_zero_recall: float = 10.0  # Weight multiplier for zero-recall classes
    boost_low_f1: float = 5.0        # Weight multiplier for low-F1 classes


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Optimizer
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    
    # Training loop
    epochs: int = 25
    early_stopping_patience: int = 5
    
    # Focal loss
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_metric: str = 'macro_f1'  # 'accuracy' or 'macro_f1'


# =============================================================================
# Class Categories (for analysis and visualization)
# =============================================================================

# Classes with F1 >= 0.90 in baseline
HIGH_F1_CLASSES = [
    "Artemis", "BitCoinMiner", "Dridex", 
    "MinerTrojan", "Ramnit", "benign"
]

# Classes with F1 = 0 in baseline (zero recall)
ZERO_RECALL_CLASSES = [
    "CCleaner", "MagicHound", "WebCompanion", "Trickster"
]

# Classes with F1 < 0.50 in baseline
LOW_F1_CLASSES = [
    "Cobalt", "TrickBot", "HTBot", "PUA", 
    "Ursnif", "Downware", "Tinba"
]


# =============================================================================
# Default Configuration Instances
# =============================================================================

def get_default_config():
    """Return default configuration objects."""
    return {
        'data': DataConfig(),
        'encoder': EncoderConfig(),
        'hybrid': HybridConfig(),
        'training': TrainingConfig(),
    }


# For backward compatibility
DATA_CONFIG = DataConfig()
ENCODER_CONFIG = EncoderConfig()
HYBRID_CONFIG = HybridConfig()
TRAINING_CONFIG = TrainingConfig()
