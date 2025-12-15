"""
Trainer Classes
===============

Training logic for LSTM-Attention encoder and hybrid classifiers.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from ..config import (
    TrainingConfig, TRAINING_CONFIG, DEVICE,
    CHECKPOINT_DIR, RESULTS_DIR
)
from ..models import NetMLEncoder, ClassWeightedFocalLoss
from ..evaluation.metrics import compute_metrics


class Trainer:
    """
    Trainer for LSTM-Attention encoder with focal loss.
    
    Handles the training loop, validation, early stopping,
    and checkpoint management.
    
    Args:
        model: NetMLEncoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: TrainingConfig with hyperparameters
        class_weights: Optional class weights for loss function
        device: Torch device
    """
    
    def __init__(
        self,
        model: NetMLEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TRAINING_CONFIG
        self.device = device or DEVICE
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = ClassWeightedFocalLoss(
            class_weights=class_weights,
            gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max' if self.config.checkpoint_metric == 'macro_f1' else 'min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_macro_f1': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_metric = 0.0 if self.config.checkpoint_metric == 'macro_f1' else float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            seq = batch['seq'].to(self.device)
            scalar = batch['scalar'].to(self.device)
            label = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            _, logits = self.model(seq, scalar)
            loss = self.criterion(logits, label)
            
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.1f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, List[int], List[int]]:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            seq = batch['seq'].to(self.device)
            scalar = batch['scalar'].to(self.device)
            label = batch['label'].to(self.device)
            
            _, logits = self.model(seq, scalar)
            loss = self.criterion(logits, label)
            
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, metrics['accuracy'], metrics['macro_f1'], all_preds, all_labels
    
    def train(
        self,
        checkpoint_path: Optional[Path] = None,
        label_map: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            checkpoint_path: Path to save best model
            label_map: Class name to index mapping (for logging)
            
        Returns:
            Training history dictionary
        """
        checkpoint_path = checkpoint_path or CHECKPOINT_DIR / "encoder_best.pth"
        
        print(f"\n{'='*60}")
        print(f"Starting Training on {self.device}")
        print(f"{'='*60}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Early Stopping Patience: {self.config.early_stopping_patience}")
        print(f"  Checkpoint Metric: {self.config.checkpoint_metric}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_f1, _, _ = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_macro_f1'].append(val_f1)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Val Macro F1: {val_f1:.4f} | LR: {current_lr:.6f}")
            
            # Check if best model
            current_metric = val_f1 if self.config.checkpoint_metric == 'macro_f1' else val_acc
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Save checkpoint
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved ({self.config.checkpoint_metric}: {current_metric:.4f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement for {self.patience_counter}/{self.config.early_stopping_patience} epochs")
            
            # Update scheduler
            self.scheduler.step(current_metric)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                print(f"  Best {self.config.checkpoint_metric}: {self.best_metric:.4f} (epoch {self.best_epoch})")
                break
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best {self.config.checkpoint_metric}: {self.best_metric:.4f}")
        print(f"  Best Epoch: {self.best_epoch}")
        print(f"{'='*60}\n")
        
        return self.history


class HybridTrainer:
    """
    Trainer for hybrid classifiers (XGBoost, LightGBM, RF) on LSTM embeddings.
    
    Extracts embeddings from a frozen encoder and trains tree-based classifiers.
    """
    
    def __init__(
        self,
        encoder: NetMLEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        classifier_type: str = 'ensemble',
        class_weights: Optional[Dict[int, float]] = None,
        device: Optional[torch.device] = None
    ):
        self.encoder = encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classifier_type = classifier_type
        self.class_weights = class_weights
        self.device = device or DEVICE
        
        # Freeze encoder
        self.encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Classifiers (initialized during training)
        self.classifiers = {}
    
    @torch.no_grad()
    def extract_embeddings(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings from data loader."""
        embeddings = []
        labels = []
        
        for batch in tqdm(loader, desc="Extracting embeddings"):
            seq = batch['seq'].to(self.device)
            scalar = batch['scalar'].to(self.device)
            
            emb = self.encoder.extract_embeddings(seq, scalar)
            embeddings.append(emb.cpu().numpy())
            labels.append(batch['label'].numpy())
        
        return np.vstack(embeddings), np.concatenate(labels)
    
    def train(self, save_dir: Optional[Path] = None) -> Dict:
        """
        Train hybrid classifier on encoder embeddings.
        
        Returns:
            Results dictionary with metrics
        """
        save_dir = save_dir or CHECKPOINT_DIR
        
        print(f"\n{'='*60}")
        print(f"Training Hybrid Classifier: {self.classifier_type.upper()}")
        print(f"{'='*60}\n")
        
        # Extract embeddings
        X_train, y_train = self.extract_embeddings(self.train_loader)
        X_val, y_val = self.extract_embeddings(self.val_loader)
        
        print(f"  Train embeddings: {X_train.shape}")
        print(f"  Val embeddings: {X_val.shape}")
        
        results = {}
        
        if self.classifier_type in ['xgboost', 'ensemble']:
            results['xgboost'] = self._train_xgboost(X_train, y_train, X_val, y_val, save_dir)
        
        if self.classifier_type in ['lightgbm', 'ensemble']:
            results['lightgbm'] = self._train_lightgbm(X_train, y_train, X_val, y_val, save_dir)
        
        if self.classifier_type in ['randomforest', 'ensemble']:
            results['randomforest'] = self._train_randomforest(X_train, y_train, X_val, y_val, save_dir)
        
        if self.classifier_type == 'ensemble':
            results['ensemble'] = self._create_ensemble(X_val, y_val, save_dir)
        
        return results
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, save_dir) -> Dict:
        """Train XGBoost classifier."""
        import xgboost as xgb
        
        print("\nTraining XGBoost...")
        
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            objective='multi:softprob',
            tree_method='hist',
            early_stopping_rounds=30,
            eval_metric='mlogloss'
        )
        
        # Apply sample weights if provided
        sample_weights = None
        if self.class_weights:
            sample_weights = np.array([self.class_weights.get(y, 1.0) for y in y_train])
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=50
        )
        
        # Evaluate
        y_pred = clf.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        
        # Save
        clf.save_model(save_dir / "xgboost_classifier.json")
        self.classifiers['xgboost'] = clf
        
        print(f"  XGBoost Macro F1: {metrics['macro_f1']:.4f}")
        return metrics
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, save_dir) -> Dict:
        """Train LightGBM classifier."""
        import lightgbm as lgb
        
        print("\nTraining LightGBM...")
        
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            class_weight='balanced' if not self.class_weights else self.class_weights,
            verbose=-1
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=True)]
        )
        
        # Evaluate
        y_pred = clf.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        
        # Save
        clf.booster_.save_model(str(save_dir / "lightgbm_classifier.txt"))
        self.classifiers['lightgbm'] = clf
        
        print(f"  LightGBM Macro F1: {metrics['macro_f1']:.4f}")
        return metrics
    
    def _train_randomforest(self, X_train, y_train, X_val, y_val, save_dir) -> Dict:
        """Train Random Forest classifier."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        print("\nTraining Random Forest...")
        
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced' if not self.class_weights else self.class_weights,
            n_jobs=-1,
            verbose=1
        )
        
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        
        # Save
        joblib.dump(clf, save_dir / "randomforest_classifier.joblib")
        self.classifiers['randomforest'] = clf
        
        print(f"  Random Forest Macro F1: {metrics['macro_f1']:.4f}")
        return metrics
    
    def _create_ensemble(self, X_val, y_val, save_dir) -> Dict:
        """Create voting ensemble from trained classifiers."""
        print("\nCreating Ensemble...")
        
        # Get predictions from all classifiers
        predictions = {}
        probabilities = {}
        
        for name, clf in self.classifiers.items():
            predictions[name] = clf.predict(X_val)
            if hasattr(clf, 'predict_proba'):
                probabilities[name] = clf.predict_proba(X_val)
        
        # Simple averaging of probabilities
        if probabilities:
            avg_probs = np.mean(list(probabilities.values()), axis=0)
            ensemble_pred = np.argmax(avg_probs, axis=1)
        else:
            # Majority voting
            from scipy import stats
            votes = np.stack(list(predictions.values()), axis=0)
            ensemble_pred = stats.mode(votes, axis=0)[0].flatten()
        
        metrics = compute_metrics(y_val, ensemble_pred)
        
        # Save ensemble config
        config = {
            'classifiers': list(self.classifiers.keys()),
            'method': 'probability_averaging',
            'macro_f1': metrics['macro_f1']
        }
        with open(save_dir / "ensemble_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  Ensemble Macro F1: {metrics['macro_f1']:.4f}")
        return metrics
