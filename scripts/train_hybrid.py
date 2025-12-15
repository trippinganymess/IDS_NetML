#!/usr/bin/env python3
"""
Train Hybrid Classifier
========================

Train tree-based classifiers (XGBoost, LightGBM, RF) on LSTM-Attention embeddings.

Usage:
    python scripts/train_hybrid.py --encoder checkpoints/encoder_best.pth --classifier ensemble
    python scripts/train_hybrid.py --encoder checkpoints/encoder_best.pth --classifier xgboost
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from netml.config import DEVICE, CHECKPOINT_DIR, RESULTS_DIR
from netml.data import NetMLDataset, create_data_loaders
from netml.models import NetMLEncoder
from netml.training import HybridTrainer
from netml.utils import set_seed, print_section, save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Train hybrid classifier')
    
    # Required arguments
    parser.add_argument('--encoder', type=str, required=True,
                        help='Path to pre-trained encoder checkpoint')
    
    # Classifier arguments
    parser.add_argument('--classifier', type=str, default='ensemble',
                        choices=['xgboost', 'lightgbm', 'randomforest', 'ensemble'],
                        help='Classifier type to train')
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data JSON file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for embedding extraction')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for classifiers')
    parser.add_argument('--name', type=str, default='hybrid',
                        help='Model name prefix')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print_section("HYBRID CLASSIFIER TRAINING")
    
    # Load dataset
    print("Loading dataset...")
    dataset = NetMLDataset(
        data_file=args.data,
        limit=args.limit
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset,
        val_split=0.20,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Load pre-trained encoder
    print(f"\nLoading encoder from {args.encoder}...")
    encoder = NetMLEncoder.from_pretrained(args.encoder, device=DEVICE)
    print(f"  Encoder loaded on {DEVICE}")
    print(f"  Embedding dim: {encoder.embedding_dim}")
    print(f"  Classes: {encoder.num_classes}")
    
    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        print("\nComputing class weights...")
        weight_tensor = dataset.get_class_weights({
            'zero_recall': 10.0,
            'low_f1': 5.0,
            'high_f1': 0.8
        })
        # Convert to dict for sklearn/xgboost
        class_weights = {i: float(weight_tensor[i]) for i in range(len(weight_tensor))}
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create hybrid trainer
    trainer = HybridTrainer(
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        classifier_type=args.classifier,
        class_weights=class_weights,
        device=DEVICE
    )
    
    # Train
    results = trainer.train(save_dir=output_dir)
    
    # Print results
    print_section("RESULTS")
    
    for classifier_name, metrics in results.items():
        print(f"\n{classifier_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Save results
    save_json(results, RESULTS_DIR / f"{args.name}_results.json")
    
    print_section("TRAINING COMPLETE")
    print(f"  Classifiers saved to: {output_dir}")
    print(f"  Results: {RESULTS_DIR / f'{args.name}_results.json'}")


if __name__ == '__main__':
    main()
