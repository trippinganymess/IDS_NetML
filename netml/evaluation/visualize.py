"""
Visualization Functions
=======================

Plotting functions for training curves, confusion matrices, and paper figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from ..config import (
    RESULTS_DIR, PAPER_FIGURES_DIR,
    HIGH_F1_CLASSES, ZERO_RECALL_CLASSES, LOW_F1_CLASSES
)


# Publication style settings
def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Acc', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path | str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    show: bool = False
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
        normalize: Whether to normalize by row
        show: Whether to display the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=1 if normalize else None)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Frequency' if normalize else 'Count')
    
    # Labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title, fontweight='bold')
    
    # Add text annotations for diagonal
    for i in range(len(class_names)):
        val = cm[i, i]
        color = 'white' if val > 0.5 else 'black'
        ax.text(i, i, f'{val:.2f}' if normalize else str(int(val)), 
                ha='center', va='center', color=color, fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_per_class_comparison(
    model_results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot per-class F1 comparison across models.
    
    Args:
        model_results: Dict mapping model names to per-class F1 dicts
        save_path: Path to save figure
        show: Whether to display the plot
    """
    set_publication_style()
    
    # Get all classes
    all_classes = sorted(list(list(model_results.values())[0].keys()))
    model_names = list(model_results.keys())
    
    # Sort by baseline F1
    first_model = model_names[0]
    all_classes = sorted(all_classes, key=lambda c: model_results[first_model].get(c, 0), reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(all_classes))
    width = 0.8 / len(model_names)
    colors = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726']
    
    for i, (model_name, results) in enumerate(model_results.items()):
        values = [results.get(c, 0) for c in all_classes]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=colors[i % len(colors)], 
               alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Traffic Class')
    ax.set_title('Per-Class F1 Score Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.08)
    
    # Threshold lines
    ax.axhline(y=0.90, color='#4CAF50', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.50, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7)
    
    # Color-code x-axis labels
    for i, label in enumerate(ax.get_xticklabels()):
        cls = all_classes[i]
        if cls in HIGH_F1_CLASSES:
            label.set_color('#2E7D32')
            label.set_fontweight('bold')
        elif cls in ZERO_RECALL_CLASSES:
            label.set_color('#C62828')
            label.set_fontweight('bold')
        elif cls in LOW_F1_CLASSES:
            label.set_color('#E65100')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_paper_figures(
    baseline_f1: Dict[str, float],
    hybrid_f1: Dict[str, float],
    ensemble_f1: Dict[str, float],
    overall_metrics: Dict[str, Dict[str, float]],
    output_dir: Optional[Path] = None
):
    """
    Generate all publication-ready figures.
    
    Args:
        baseline_f1: Per-class F1 for baseline model
        hybrid_f1: Per-class F1 for XGBoost hybrid
        ensemble_f1: Per-class F1 for ensemble model
        overall_metrics: Dict with 'accuracy', 'macro_f1' for each model
        output_dir: Output directory for figures
    """
    set_publication_style()
    output_dir = output_dir or PAPER_FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating publication figures in {output_dir}...")
    
    # Figure 1: Overall Metrics
    _plot_overall_metrics(overall_metrics, output_dir)
    
    # Figure 2: Per-class F1 Comparison
    model_results = {
        'LSTM-Attn (Softmax)': baseline_f1,
        'LSTM-Attn + XGBoost': hybrid_f1,
        'LSTM-Attn + Ensemble': ensemble_f1
    }
    plot_per_class_comparison(model_results, output_dir / "fig2_per_class_f1.pdf")
    
    # Figure 3: Zero-recall Recovery
    _plot_zero_recall_recovery(baseline_f1, hybrid_f1, ensemble_f1, output_dir)
    
    print(f"\n✓ All figures saved to {output_dir}")


def _plot_overall_metrics(
    overall_metrics: Dict[str, Dict[str, float]],
    output_dir: Path
):
    """Plot overall metrics comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    models = list(overall_metrics.keys())
    colors = ['#EF5350', '#42A5F5', '#66BB6A']
    
    # Macro F1
    macro_f1 = [overall_metrics[m]['macro_f1'] for m in models]
    bars1 = axes[0].bar(models, macro_f1, color=colors, edgecolor='black')
    axes[0].set_ylabel('Macro F1 Score')
    axes[0].set_title('(a) Macro F1 Score', fontweight='bold')
    axes[0].set_ylim(0.45, 0.58)
    
    for bar, val in zip(bars1, macro_f1):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    # Accuracy
    accuracy = [overall_metrics[m]['accuracy'] for m in models]
    bars2 = axes[1].bar(models, accuracy, color=colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('(b) Classification Accuracy', fontweight='bold')
    axes[1].set_ylim(70, 82)
    
    for bar, val in zip(bars2, accuracy):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_overall_metrics.pdf", format='pdf')
    plt.savefig(output_dir / "fig1_overall_metrics.png", format='png')
    plt.close()
    print("  ✓ fig1_overall_metrics.pdf/png")


def _plot_zero_recall_recovery(
    baseline_f1: Dict[str, float],
    hybrid_f1: Dict[str, float],
    ensemble_f1: Dict[str, float],
    output_dir: Path
):
    """Plot zero-recall class recovery."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    zero_classes = ZERO_RECALL_CLASSES
    x = np.arange(len(zero_classes))
    width = 0.25
    
    baseline_vals = [baseline_f1.get(c, 0) for c in zero_classes]
    hybrid_vals = [hybrid_f1.get(c, 0) for c in zero_classes]
    ensemble_vals = [ensemble_f1.get(c, 0) for c in zero_classes]
    
    ax.bar(x - width, baseline_vals, width, label='LSTM-Attn (Softmax)', color='#EF5350')
    ax.bar(x, hybrid_vals, width, label='LSTM-Attn + XGBoost', color='#42A5F5')
    bars3 = ax.bar(x + width, ensemble_vals, width, label='LSTM-Attn + Ensemble', color='#66BB6A')
    
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Traffic Class')
    ax.set_title('Recovery of Zero-Recall Classes', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(zero_classes)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.75)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        if height > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.015,
                    f'{height:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_zero_recall_recovery.pdf", format='pdf')
    plt.savefig(output_dir / "fig3_zero_recall_recovery.png", format='png')
    plt.close()
    print("  ✓ fig3_zero_recall_recovery.pdf/png")


def plot_architecture_diagram(save_path: str):
    """
    Create a model architecture diagram.
    
    Args:
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors
    input_color = '#E3F2FD'
    lstm_color = '#BBDEFB'
    attention_color = '#90CAF9'
    embed_color = '#64B5F6'
    classifier_color = '#42A5F5'
    output_color = '#1E88E5'
    
    # Helper function for boxes
    def draw_box(x, y, w, h, color, text, fontsize=10):
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    
    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Input Layer
    draw_box(0.5, 3.5, 2, 1, input_color, 'Input\n(76 features)', 9)
    
    # Bidirectional LSTM
    draw_box(3.5, 2.5, 2.5, 3, lstm_color, 'Bi-LSTM\n(3 layers)\nhidden=128', 9)
    
    # Multi-Head Attention
    draw_box(7, 2.5, 2.5, 3, attention_color, 'Multi-Head\nAttention\n(4 heads)', 9)
    
    # Embedding Layer
    draw_box(10.5, 3.5, 2, 1, embed_color, 'Embedding\n(64-dim, L2)', 9)
    
    # Classifiers (below main flow)
    draw_box(3, 0.5, 1.8, 1, classifier_color, 'LightGBM', 8)
    draw_box(5.5, 0.5, 2.2, 1, classifier_color, 'Random Forest', 8)
    draw_box(8.5, 0.5, 1.8, 1, classifier_color, 'XGBoost', 8)
    
    # Ensemble Voting
    draw_box(5.5, -1, 2.5, 0.8, '#FFA726', 'Soft Voting', 9)
    
    # Output
    draw_box(10.5, 0.5, 2, 1, output_color, 'Output\n(21 classes)', 9)
    
    # Arrows - main flow
    draw_arrow(2.5, 4, 3.5, 4)
    draw_arrow(6, 4, 7, 4)
    draw_arrow(9.5, 4, 10.5, 4)
    
    # Arrows - to classifiers
    draw_arrow(11.5, 3.5, 11.5, 2.5)
    ax.annotate('', xy=(3.9, 1.5), xytext=(11.5, 2.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1, connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(6.6, 1.5), xytext=(11.5, 2.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.annotate('', xy=(9.4, 1.5), xytext=(11.5, 2.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1, connectionstyle='arc3,rad=-0.2'))
    
    # Arrows - to voting
    draw_arrow(3.9, 0.5, 5.7, -0.2)
    draw_arrow(6.6, 0.5, 6.7, -0.2)
    draw_arrow(9.4, 0.5, 7.8, -0.2)
    
    # Arrow - to output
    draw_arrow(8, -0.6, 10.5, 0.8)
    
    # Title
    ax.text(7, 7.2, 'NetML: LSTM-Attention + Hybrid Ensemble Architecture', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Legend
    ax.text(0.5, 7, 'Network Traffic Classification', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_f1(
    per_class_f1: Dict[str, float],
    save_path: str,
    title: str = "Per-Class F1 Scores"
):
    """
    Plot per-class F1 scores as a horizontal bar chart.
    
    Args:
        per_class_f1: Dictionary mapping class names to F1 scores
        save_path: Path to save the figure
        title: Plot title
    """
    # Sort by F1 score
    sorted_items = sorted(per_class_f1.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Color based on performance
    colors = []
    for cls in classes:
        if cls in HIGH_F1_CLASSES:
            colors.append('#4CAF50')  # Green
        elif cls in ZERO_RECALL_CLASSES:
            colors.append('#f44336')  # Red
        elif cls in LOW_F1_CLASSES:
            colors.append('#FF9800')  # Orange
        else:
            colors.append('#2196F3')  # Blue
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(classes))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel('F1 Score')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, 1.1)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=8)
    
    # Add threshold lines
    ax.axvline(x=0.9, color='#4CAF50', linestyle='--', alpha=0.7, label='High (≥90%)')
    ax.axvline(x=0.5, color='#FF9800', linestyle='--', alpha=0.7, label='Medium (≥50%)')
    
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_latex_table(
    results: Dict[str, Dict[str, float]],
    save_path: str
):
    """
    Generate a LaTeX table from results.
    
    Args:
        results: Dictionary mapping model names to metrics dictionaries
        save_path: Path to save the .tex file
    """
    latex = r"""\begin{table}[htbp]
\centering
\caption{Performance Comparison on NetML-2020 Dataset}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Macro F1} & \textbf{Accuracy} & \textbf{$\Delta$ F1} \\
\midrule
"""
    
    baseline_f1 = None
    for model_name, metrics in results.items():
        macro_f1 = metrics.get('macro_f1', 0)
        accuracy = metrics.get('accuracy', 0)
        
        if baseline_f1 is None:
            baseline_f1 = macro_f1
            delta = "--"
        else:
            delta_val = (macro_f1 - baseline_f1) / baseline_f1 * 100
            delta = f"+{delta_val:.2f}\\%"
        
        latex += f"{model_name} & {macro_f1:.4f} & {accuracy:.4f} & {delta} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)

