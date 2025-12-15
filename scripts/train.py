#!/usr/bin/env python3
"""
Train LSTM-Attention Encoder
=============================

Main training script for the NetML traffic classification encoder.

Usage:
    python scripts/train.py --data Data/Master.json --epochs 25
    python scripts/train.py --config custom_config.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from netml.config import (
    DATA_CONFIG, ENCODER_CONFIG, TRAINING_CONFIG,
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR
)
from netml.data import NetMLDataset, create_data_loaders
from netml.models import NetMLEncoder
from netml.training import Trainer
from netml.evaluation import plot_training_curves, plot_confusion_matrix, compute_metrics
from netml.utils import set_seed, print_section, save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM-Attention encoder')
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data JSON file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of training samples')
    parser.add_argument('--val-split', type=float, default=0.20,
                        help='Validation split fraction')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--lstm-hidden', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--lstm-layers', type=int, default=3,
                        help='Number of LSTM layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--name', type=str, default='encoder',
                        help='Model name prefix')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print_section("NETML ENCODER TRAINING")
    
    # Load dataset
    print("Loading dataset...")
    dataset = NetMLDataset(
        data_file=args.data,
        limit=args.limit
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Get class weights if requested
    class_weights = None
    if args.use_class_weights:
        print("Computing class weights...")
        class_weights = dataset.get_class_weights({
            'zero_recall': 10.0,
            'low_f1': 5.0,
            'high_f1': 0.8
        })
    
    # Create model
    print(f"\nCreating model on {DEVICE}...")
    
    # Update config with command line args
    ENCODER_CONFIG.embedding_dim = args.embedding_dim
    ENCODER_CONFIG.lstm_hidden = args.lstm_hidden
    ENCODER_CONFIG.lstm_layers = args.lstm_layers
    
    model = NetMLEncoder(
        config=ENCODER_CONFIG,
        num_classes=dataset.num_classes
    )
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Classes: {dataset.num_classes}")
    
    # Update training config
    TRAINING_CONFIG.epochs = args.epochs
    TRAINING_CONFIG.learning_rate = args.lr
    TRAINING_CONFIG.early_stopping_patience = args.patience
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TRAINING_CONFIG,
        class_weights=class_weights,
        device=DEVICE
    )
    
    # Setup output paths
    output_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / f"{args.name}_best.pth"
    
    # Train
    history = trainer.train(
        checkpoint_path=checkpoint_path,
        label_map=dataset.label_map
    )
    
    # Save training curves
    plot_training_curves(
        history,
        save_path=RESULTS_DIR / f"{args.name}_training_curves.png"
    )
    
    # Save label map
    dataset.save_label_map(output_dir / "label_map.json")
    
    # Save training history
    save_json(history, RESULTS_DIR / f"{args.name}_history.json")
    
    # Final evaluation
    print_section("FINAL EVALUATION")
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    # Compute final metrics
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            seq = batch['seq'].to(DEVICE)
            scalar = batch['scalar'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            
            _, logits = model(seq, scalar)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    from netml.evaluation.metrics import per_class_f1, classification_report_dict
    
    metrics = compute_metrics(all_labels, all_preds)
    class_f1 = per_class_f1(all_labels, all_preds, dataset.label_map)
    
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Save final results
    results = {
        'overall': metrics,
        'per_class_f1': class_f1,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'embedding_dim': args.embedding_dim
        }
    }
    save_json(results, RESULTS_DIR / f"{args.name}_results.json")
    
    print_section("TRAINING COMPLETE")
    print(f"  Best model: {checkpoint_path}")
    print(f"  Results: {RESULTS_DIR / f'{args.name}_results.json'}")


if __name__ == '__main__':
    import torch
    main()
