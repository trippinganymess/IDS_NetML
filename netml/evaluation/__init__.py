"""Evaluation module - metrics and visualization."""

from .metrics import compute_metrics, classification_report_dict, per_class_f1
from .visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_comparison,
    plot_architecture_diagram,
    plot_per_class_f1,
    generate_paper_figures,
    generate_latex_table,
)

__all__ = [
    'compute_metrics',
    'classification_report_dict', 
    'per_class_f1',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_per_class_comparison',
    'plot_architecture_diagram',
    'plot_per_class_f1',
    'generate_paper_figures',
    'generate_latex_table',
]
