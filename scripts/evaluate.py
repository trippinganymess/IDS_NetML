#!/usr/bin/env python3
"""
Evaluate Models
===============

Evaluate trained models and generate metrics/visualizations.

Usage:
    python scripts/evaluate.py --encoder checkpoints/encoder_best.pth
    python scripts/evaluate.py --encoder checkpoints/encoder_best.pth --hybrid checkpoints/ensemble_config.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm

from netml.config import DEVICE, RESULTS_DIR
from netml.data import NetMLDataset, create_data_loaders
from netml.models import NetMLEncoder
from netml.evaluation import (
    compute_metrics, 
    per_class_f1, 
    plot_confusion_matrix,
    plot_per_class_comparison
)
from netml.evaluation.metrics import compute_confusion_matrix
from netml.utils import set_seed, print_section, save_json, load_json


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models')
    
    parser.add_argument('--encoder', type=str, required=True,
                        help='Path to encoder checkpoint')
    parser.add_argument('--hybrid', type=str, default=None,
                        help='Path to hybrid classifier config')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to evaluation data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--name', type=str, default='evaluation',
                        help='Output name prefix')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def evaluate_encoder(model, val_loader, device):
    """Evaluate encoder with softmax classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating encoder"):
            seq = batch['seq'].to(device)
            scalar = batch['scalar'].to(device)
            label = batch['label'].to(device)
            
            _, logits = model(seq, scalar)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)


def evaluate_hybrid(model, classifier, val_loader, device):
    """Evaluate hybrid classifier on encoder embeddings."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting embeddings"):
            seq = batch['seq'].to(device)
            scalar = batch['scalar'].to(device)
            
            embeddings = model.extract_embeddings(seq, scalar)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(batch['label'].numpy())
    
    X = np.vstack(all_embeddings)
    y_true = np.concatenate(all_labels)
    
    y_pred = classifier.predict(X)
    
    return y_true, y_pred


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print_section("MODEL EVALUATION")
    
    # Load dataset
    print("Loading dataset...")
    dataset = NetMLDataset(
        data_file=args.data,
        limit=args.limit
    )
    
    _, val_loader = create_data_loaders(
        dataset,
        val_split=0.20,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Load encoder
    print(f"\nLoading encoder from {args.encoder}...")
    encoder = NetMLEncoder.from_pretrained(args.encoder, device=DEVICE)
    
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Evaluate encoder (softmax)
    print("\n--- Evaluating LSTM-Attn (Softmax) ---")
    y_true, y_pred_softmax = evaluate_encoder(encoder, val_loader, DEVICE)
    
    metrics_softmax = compute_metrics(y_true, y_pred_softmax)
    f1_softmax = per_class_f1(y_true, y_pred_softmax, dataset.label_map)
    
    results['softmax'] = {
        'overall': metrics_softmax,
        'per_class_f1': f1_softmax
    }
    
    print(f"  Accuracy: {metrics_softmax['accuracy']:.2f}%")
    print(f"  Macro F1: {metrics_softmax['macro_f1']:.4f}")
    
    # Evaluate hybrid if provided
    if args.hybrid:
        print(f"\n--- Evaluating Hybrid Classifier ---")
        
        hybrid_config = load_json(args.hybrid)
        
        # Load classifiers based on config
        # This is simplified - full implementation would load all classifiers
        import joblib
        import xgboost as xgb
        
        classifier_path = Path(args.hybrid).parent
        
        if 'xgboost' in hybrid_config.get('classifiers', []):
            xgb_clf = xgb.XGBClassifier()
            xgb_clf.load_model(classifier_path / "xgboost_classifier.json")
            
            y_true, y_pred_xgb = evaluate_hybrid(encoder, xgb_clf, val_loader, DEVICE)
            metrics_xgb = compute_metrics(y_true, y_pred_xgb)
            f1_xgb = per_class_f1(y_true, y_pred_xgb, dataset.label_map)
            
            results['xgboost'] = {
                'overall': metrics_xgb,
                'per_class_f1': f1_xgb
            }
            
            print(f"  XGBoost Macro F1: {metrics_xgb['macro_f1']:.4f}")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    class_names = sorted(dataset.label_map.keys())
    cm = compute_confusion_matrix(y_true, y_pred_softmax)
    
    plot_confusion_matrix(
        cm, class_names,
        save_path=output_dir / f"{args.name}_confusion_matrix.png",
        title="Confusion Matrix"
    )
    
    # Save results
    save_json(results, output_dir / f"{args.name}_results.json")
    
    print_section("EVALUATION COMPLETE")
    print(f"  Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
