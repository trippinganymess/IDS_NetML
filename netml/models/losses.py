"""
Loss Functions
==============

Custom loss functions for handling class imbalance in traffic classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights easy examples and focuses on hard negatives.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        
    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch, num_classes]
            targets: Class indices [batch]
            
        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ClassWeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class weights.
    
    Combines focal loss with class-specific weighting to handle
    both hard examples and class imbalance simultaneously.
    
    Args:
        class_weights: Tensor of per-class weights [num_classes]
        gamma: Focusing parameter (default: 2.0)
        label_smoothing: Smoothing factor (default: 0.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class-weighted focal loss.
        
        Args:
            inputs: Logits [batch, num_classes]
            targets: Class indices [batch]
            
        Returns:
            Scalar loss value
        """
        num_classes = inputs.size(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(inputs)
                targets_smooth.fill_(self.label_smoothing / (num_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets_smooth * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal modulation
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights.to(inputs.device)
            weight_vector = weights[targets]
            focal_loss = focal_loss * weight_vector
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Regularizes the model by preventing overconfident predictions.
    
    Args:
        smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = uniform)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            targets_smooth = torch.zeros_like(log_probs)
            targets_smooth.fill_(self.smoothing / (num_classes - 1))
            targets_smooth.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        loss = -(targets_smooth * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()
