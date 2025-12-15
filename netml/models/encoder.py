"""
LSTM-Attention Encoder
======================

Neural network encoder for network traffic classification.
Uses bidirectional LSTM with multi-head attention for sequence processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..config import EncoderConfig, ENCODER_CONFIG


class NetMLEncoder(nn.Module):
    """
    LSTM + Multi-head Attention encoder with projection head.
    
    Architecture:
        Sequence Branch: LSTM → Multi-head Attention → Pooling
        Scalar Branch: MLP → Self-Attention
        Fusion: Concatenate → Projection MLP → L2-normalized embeddings
        
    Args:
        config: EncoderConfig with model hyperparameters
        num_classes: Number of output classes (overrides config)
    """
    
    def __init__(
        self, 
        config: Optional[EncoderConfig] = None,
        num_classes: Optional[int] = None
    ):
        super().__init__()
        
        cfg = config or ENCODER_CONFIG
        self.config = cfg
        
        # Override num_classes if provided
        n_classes = num_classes or cfg.num_classes
        if n_classes is None:
            raise ValueError("num_classes must be specified")
        
        # === Sequence Branch (LSTM + Attention) ===
        self.lstm = nn.LSTM(
            input_size=cfg.num_seq_features,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.lstm_hidden,
            num_heads=cfg.attention_heads,
            dropout=cfg.attention_dropout,
            batch_first=True
        )
        
        # === Scalar Branch (MLP + Self-Attention) ===
        self.scalar_net = nn.Sequential(
            nn.Linear(cfg.num_scalars, cfg.lstm_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(cfg.lstm_hidden),
            nn.Dropout(cfg.projection_dropout),
            nn.Linear(cfg.lstm_hidden, cfg.lstm_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(cfg.lstm_hidden)
        )
        
        self.scalar_attention = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # === Fusion & Projection Head ===
        fusion_dim = cfg.lstm_hidden + cfg.lstm_hidden  # seq + scalar
        
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(cfg.projection_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(cfg.projection_dropout / 2),
            nn.Linear(128, cfg.embedding_dim)
        )
        
        # === Classification Head ===
        self.classifier = nn.Linear(cfg.embedding_dim, n_classes)
        
        # Store dimensions for external use
        self.embedding_dim = cfg.embedding_dim
        self.num_classes = n_classes
    
    def forward(
        self, 
        x_seq: torch.Tensor, 
        x_scalar: torch.Tensor,
        return_embeddings: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x_seq: Sequence features [batch, seq_len, num_seq_features]
            x_scalar: Scalar features [batch, num_scalars]
            return_embeddings: If True, return (embeddings, logits); else just logits
            
        Returns:
            embeddings: L2-normalized embeddings [batch, embedding_dim]
            logits: Classification logits [batch, num_classes]
        """
        # === Sequence Branch ===
        lstm_out, _ = self.lstm(x_seq)  # [batch, seq_len, hidden]
        
        # Self-attention over LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Weighted pooling
        attn_weights = torch.softmax(attn_out.mean(dim=-1, keepdim=True), dim=1)
        seq_feat = (attn_out * attn_weights).sum(dim=1)  # [batch, hidden]
        
        # === Scalar Branch ===
        scalar_feat = self.scalar_net(x_scalar)  # [batch, hidden]
        
        # Self-attention weighting
        scalar_attn = self.scalar_attention(scalar_feat.unsqueeze(1))
        scalar_feat = scalar_feat * scalar_attn.squeeze(-1)
        
        # === Fusion ===
        combined = torch.cat([seq_feat, scalar_feat], dim=1)
        
        # === Projection ===
        embeddings = self.projection(combined)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
        
        # === Classification ===
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return embeddings, logits
        return logits
    
    def extract_embeddings(
        self, 
        x_seq: torch.Tensor, 
        x_scalar: torch.Tensor
    ) -> torch.Tensor:
        """Extract only embeddings (for hybrid classifiers)."""
        embeddings, _ = self.forward(x_seq, x_scalar)
        return embeddings
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: torch.device = None):
        """
        Load encoder from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to load model to
            
        Returns:
            Loaded NetMLEncoder instance
        """
        if device is None:
            from ..config import DEVICE
            device = DEVICE
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Infer num_classes from classifier weight shape
        classifier_weight = state_dict.get('classifier.weight')
        if classifier_weight is not None:
            num_classes = classifier_weight.shape[0]
        else:
            raise ValueError("Cannot infer num_classes from checkpoint")
        
        model = cls(num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
