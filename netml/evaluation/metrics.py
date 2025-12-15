"""
Evaluation Metrics
==================

Classification metrics for model evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class ('macro', 'weighted', 'micro')
        
    Returns:
        Dictionary with accuracy, precision, recall, f1 scores
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def per_class_f1(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    label_map: Optional[Dict[str, int]] = None
) -> Dict[str, float]:
    """
    Compute per-class F1 scores.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_map: Optional mapping from class names to indices
        
    Returns:
        Dictionary mapping class names/indices to F1 scores
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    
    if label_map:
        inv_map = {v: k for k, v in label_map.items()}
        return {inv_map.get(int(c), str(c)): float(f) for c, f in zip(classes, f1_scores)}
    
    return {str(int(c)): float(f) for c, f in zip(classes, f1_scores)}


def classification_report_dict(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    label_map: Optional[Dict[str, int]] = None
) -> Dict:
    """
    Generate classification report as dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_map: Optional mapping from class names to indices
        
    Returns:
        Classification report dictionary
    """
    target_names = None
    if label_map:
        inv_map = {v: k for k, v in label_map.items()}
        classes = sorted(set(y_true) | set(y_pred))
        target_names = [inv_map.get(c, str(c)) for c in classes]
    
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )


def compute_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    normalize: bool = False
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: If True, normalize by row (true labels)
        
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    
    return cm


def compare_models(
    baseline_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    label_map: Optional[Dict[str, int]] = None
) -> Dict:
    """
    Compare metrics between baseline and new model.
    
    Args:
        baseline_metrics: Per-class F1 from baseline
        new_metrics: Per-class F1 from new model
        label_map: Optional class name mapping
        
    Returns:
        Comparison dictionary with improvements/degradations
    """
    from ..config import HIGH_F1_CLASSES, ZERO_RECALL_CLASSES, LOW_F1_CLASSES
    
    comparison = {
        'improved': [],
        'degraded': [],
        'stable': [],
        'zero_recall_recovered': [],
        'high_f1_maintained': []
    }
    
    for cls in baseline_metrics.keys():
        if cls not in new_metrics:
            continue
            
        old_f1 = baseline_metrics[cls]
        new_f1 = new_metrics[cls]
        diff = new_f1 - old_f1
        
        entry = {
            'class': cls,
            'baseline': old_f1,
            'new': new_f1,
            'change': diff
        }
        
        if diff > 0.01:
            comparison['improved'].append(entry)
        elif diff < -0.01:
            comparison['degraded'].append(entry)
        else:
            comparison['stable'].append(entry)
        
        # Check special cases
        if cls in ZERO_RECALL_CLASSES and old_f1 == 0 and new_f1 > 0:
            comparison['zero_recall_recovered'].append(entry)
        
        if cls in HIGH_F1_CLASSES and new_f1 >= 0.90:
            comparison['high_f1_maintained'].append(entry)
    
    # Sort by change magnitude
    comparison['improved'].sort(key=lambda x: x['change'], reverse=True)
    comparison['degraded'].sort(key=lambda x: x['change'])
    
    return comparison
