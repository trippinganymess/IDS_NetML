#!/usr/bin/env python3
"""
Inference Script for NetML Network Traffic Classification

This script performs inference on new network traffic data using the trained
LSTM-Attention encoder with hybrid ensemble classifiers.

Usage:
    python scripts/inference.py --data path/to/data.csv --output predictions.csv
    python scripts/inference.py --data path/to/data.csv --classifier ensemble
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from netml.config import EncoderConfig, ENCODER_CONFIG
from netml.models.encoder import NetMLEncoder
from netml.utils.helpers import get_device_info, print_section


def load_encoder(checkpoint_path: str, device: torch.device, num_classes: int = 21) -> NetMLEncoder:
    """Load the trained encoder model."""
    encoder = NetMLEncoder(config=ENCODER_CONFIG, num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    
    return encoder


def load_classifiers(checkpoint_dir: Path, classifier_type: str = "ensemble"):
    """Load the trained classifiers."""
    classifiers = {}
    
    if classifier_type in ["xgboost", "all"]:
        xgb_path = checkpoint_dir / "xgboost_classifier.json"
        if xgb_path.exists():
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(str(xgb_path))
            classifiers["xgboost"] = xgb_model
    
    if classifier_type in ["lightgbm", "ensemble", "all"]:
        lgb_path = checkpoint_dir / "lightgbm_classifier.txt"
        if lgb_path.exists():
            import lightgbm as lgb
            lgb_model = lgb.Booster(model_file=str(lgb_path))
            classifiers["lightgbm"] = lgb_model
    
    if classifier_type in ["randomforest", "ensemble", "all"]:
        rf_path = checkpoint_dir / "randomforest_classifier.joblib"
        if rf_path.exists():
            rf_model = joblib.load(rf_path)
            classifiers["randomforest"] = rf_model
    
    return classifiers


def load_label_map(checkpoint_dir: Path) -> dict:
    """Load the label mapping."""
    label_map_path = checkpoint_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r") as f:
            return json.load(f)
    return {}


def extract_embeddings(
    encoder: NetMLEncoder,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract embeddings from data using the encoder."""
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.FloatTensor(data[i:i+batch_size]).to(device)
            emb = encoder(batch)
            embeddings.append(emb.cpu().numpy())
    
    return np.vstack(embeddings)


def predict_ensemble(
    embeddings: np.ndarray,
    classifiers: dict,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions using ensemble voting."""
    predictions = []
    probabilities = []
    
    for name, clf in classifiers.items():
        if name == "lightgbm":
            proba = clf.predict(embeddings)
            pred = np.argmax(proba, axis=1)
        else:
            pred = clf.predict(embeddings)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(embeddings)
            else:
                # One-hot encode predictions as probabilities
                proba = np.zeros((len(pred), num_classes))
                proba[np.arange(len(pred)), pred] = 1.0
        
        predictions.append(pred)
        probabilities.append(proba)
    
    # Soft voting: average probabilities
    avg_proba = np.mean(probabilities, axis=0)
    final_pred = np.argmax(avg_proba, axis=1)
    
    return final_pred, avg_proba


def predict_single(
    embeddings: np.ndarray,
    classifier,
    classifier_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions using a single classifier."""
    if classifier_name == "lightgbm":
        proba = classifier.predict(embeddings)
        pred = np.argmax(proba, axis=1)
    else:
        pred = classifier.predict(embeddings)
        proba = classifier.predict_proba(embeddings) if hasattr(classifier, "predict_proba") else None
    
    return pred, proba


def load_data(data_path: str) -> tuple[np.ndarray, list | None]:
    """Load data from CSV or NPZ file."""
    path = Path(data_path)
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        # Assume last column might be label if it exists
        if "label" in df.columns.str.lower():
            label_col = [c for c in df.columns if c.lower() == "label"][0]
            labels = df[label_col].tolist()
            features = df.drop(columns=[label_col]).values
        else:
            labels = None
            features = df.values
        return features.astype(np.float32), labels
    
    elif path.suffix == ".npz":
        data = np.load(path)
        features = data["features"] if "features" in data else data["X"]
        labels = data["labels"].tolist() if "labels" in data else None
        return features.astype(np.float32), labels
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="NetML Inference - Network Traffic Classification"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input data (CSV or NPZ file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to output predictions file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="ensemble",
        choices=["xgboost", "lightgbm", "randomforest", "ensemble"],
        help="Classifier to use for prediction",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    
    args = parser.parse_args()
    
    # Setup
    device_info = get_device_info()
    device = device_info["device"]
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.verbose:
        print_section("NetML Inference")
        print(f"Device: {device}")
        print(f"Input: {args.data}")
        print(f"Classifier: {args.classifier}")
    
    # Load label map first to get num_classes
    label_map = load_label_map(checkpoint_dir)
    num_classes = len(label_map) if label_map else 21
    
    # Load encoder
    if args.verbose:
        print("\nLoading encoder...")
    encoder_path = checkpoint_dir / "encoder_best.pth"
    if not encoder_path.exists():
        print(f"Error: Encoder checkpoint not found at {encoder_path}")
        sys.exit(1)
    
    encoder = load_encoder(str(encoder_path), device, num_classes)
    
    # Load classifiers
    if args.verbose:
        print(f"Loading {args.classifier} classifier(s)...")
    classifiers = load_classifiers(checkpoint_dir, args.classifier)
    
    if not classifiers:
        print(f"Error: No classifiers found in {checkpoint_dir}")
        sys.exit(1)
    
    # Get idx to label mapping
    idx_to_label = {int(v): k for k, v in label_map.items()} if label_map else {}
    num_classes = len(label_map) if label_map else 21
    
    # Load data
    if args.verbose:
        print(f"\nLoading data from {args.data}...")
    features, true_labels = load_data(args.data)
    if args.verbose:
        print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    
    # Extract embeddings
    if args.verbose:
        print("\nExtracting embeddings...")
    embeddings = extract_embeddings(encoder, features, device, args.batch_size)
    
    # Make predictions
    if args.verbose:
        print("Making predictions...")
    
    if args.classifier == "ensemble":
        predictions, probabilities = predict_ensemble(embeddings, classifiers, num_classes)
    else:
        clf = classifiers[args.classifier]
        predictions, probabilities = predict_single(embeddings, clf, args.classifier)
    
    # Convert to class names
    if idx_to_label:
        pred_labels = [idx_to_label.get(p, f"class_{p}") for p in predictions]
    else:
        pred_labels = predictions.tolist()
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        "prediction": pred_labels,
        "prediction_idx": predictions,
    })
    
    # Add confidence scores if available
    if probabilities is not None:
        output_df["confidence"] = np.max(probabilities, axis=1)
    
    # Add true labels if available
    if true_labels is not None:
        output_df["true_label"] = true_labels
        correct = (output_df["prediction"] == output_df["true_label"]).sum()
        accuracy = correct / len(output_df) * 100
        if args.verbose:
            print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(output_df)})")
    
    # Save predictions
    output_df.to_csv(args.output, index=False)
    if args.verbose:
        print(f"\nPredictions saved to {args.output}")
    
    # Print summary
    print_section("Prediction Summary")
    print(f"Total samples: {len(predictions)}")
    print(f"Unique classes predicted: {len(np.unique(predictions))}")
    
    if args.verbose:
        print("\nClass distribution:")
        for cls, count in zip(*np.unique(predictions, return_counts=True)):
            label = idx_to_label.get(cls, f"class_{cls}")
            pct = count / len(predictions) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
