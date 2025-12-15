# NetML Traffic Classification

A hybrid deep learning system for network traffic classification using LSTM-Attention encoders with tree-based ensemble classifiers. Achieves state-of-the-art performance on the NetML-2020 benchmark.

## Project Structure

```
IDS_NetML/
├── netml/                      # Main package
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py          # NetML dataset class
│   ├── models/                 # Neural network models
│   │   ├── __init__.py
│   │   ├── encoder.py          # LSTM-Attention encoder
│   │   └── losses.py           # Focal loss and other losses
│   ├── training/               # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py          # Trainer class
│   ├── evaluation/             # Metrics and visualization
│   │   ├── __init__.py
│   │   ├── metrics.py          # Classification metrics
│   │   └── visualize.py        # Plotting functions
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── helpers.py          # Common utilities
├── scripts/                    # Entry point scripts
│   ├── train.py                # Train LSTM-Attention encoder
│   ├── train_hybrid.py         # Train hybrid classifier
│   ├── evaluate.py             # Evaluate models
│   ├── inference.py            # Run inference on new data
│   └── generate_figures.py     # Generate paper figures
├── checkpoints/                # Saved model weights
│   ├── encoder_best.pth        # Best encoder checkpoint
│   ├── lightgbm_classifier.txt # LightGBM model
│   ├── randomforest_classifier.joblib
│   ├── xgboost_classifier.json
│   └── label_map.json          # Class label mapping
├── results/                    # Outputs
│   ├── paper_figures/          # Publication-ready figures
│   ├── visualizations/         # Training curves, charts
│   └── *.json                  # Metric results
├── docs/                       # Documentation
│   ├── TRAINING_METRICS_REPORT.md
│   ├── F1_SCORES_QUICK_SUMMARY.md
│   └── FOCAL_LOSS_GUIDE.md
├── Data/                       # Dataset (not in git)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/IDS_NetML.git
cd IDS_NetML

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the LSTM-Attention Encoder

```bash
python scripts/train.py --data Data/Master.json --epochs 25 --device mps
```

### 2. Train the Hybrid Classifier (Best Performance)

```bash
# Ensemble (LightGBM + Random Forest) - Recommended
python scripts/train_hybrid.py --encoder checkpoints/encoder_best.pth --classifier ensemble

# XGBoost only
python scripts/train_hybrid.py --encoder checkpoints/encoder_best.pth --classifier xgboost

# LightGBM only
python scripts/train_hybrid.py --encoder checkpoints/encoder_best.pth --classifier lightgbm
```

### 3. Evaluate Models

```bash
python scripts/evaluate.py --checkpoint-dir checkpoints --classifier ensemble
```

### 4. Inference on New Data

```bash
python scripts/inference.py --data path/to/data.csv --output predictions.csv --classifier ensemble
```

### 5. Generate Publication Figures

```bash
python scripts/generate_figures.py --results-dir results --output-dir results/paper_figures
```

## Model Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    NetML Hybrid Classification System                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input (76 features)                                                   │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────────┐                                               │
│   │  Bidirectional LSTM │  (3 layers, hidden=128)                       │
│   │  with Dropout       │                                               │
│   └─────────────────────┘                                               │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────────┐                                               │
│   │  Multi-Head         │  (4 attention heads)                          │
│   │  Self-Attention     │                                               │
│   └─────────────────────┘                                               │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────────┐                                               │
│   │  L2-Normalized      │  (64 dimensions)                              │
│   │  Embeddings         │                                               │
│   └─────────────────────┘                                               │
│         │                                                               │
│         ├──────────────┬───────────────┐                                │
│         ▼              ▼               ▼                                │
│   ┌──────────┐  ┌──────────────┐  ┌──────────┐                          │
│   │ LightGBM │  │RandomForest  │  │ XGBoost  │  (Alternative)           │
│   └──────────┘  └──────────────┘  └──────────┘                          │
│         │              │                                                │
│         └──────┬───────┘                                                │
│                ▼                                                        │
│        ┌───────────────┐                                                │
│        │  Soft Voting  │                                                │
│        └───────────────┘                                                │
│                │                                                        │
│                ▼                                                        │
│          21 Classes                                                     │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Performance Results

### Overall Metrics

| Model | Macro F1 | Accuracy | Improvement |
|-------|----------|----------|-------------|
| LSTM-Attn (Softmax) | 0.5103 | 78.68% | Baseline |
| LSTM-Attn + XGBoost | 0.5243 | 76.07% | +2.74% |
| **LSTM-Attn + Ensemble** | **0.5332** | **76.10%** | **+4.49%** |

### Zero-Recall Class Recovery

| Class | Baseline | Ensemble | Recovery |
|-------|----------|----------|----------|
| CCleaner | 0.00% | **64.29%** | ✅ |
| Trickster | 0.00% | **35.29%** | ✅ |
| MagicHound | 0.00% | **10.35%** | ✅ |
| WebCompanion | 0.00% | **9.10%** | ✅ |

### High-F1 Classes (Maintained ≥90%)

- Artemis: 99.46%
- BitCoinMiner: 99.50%
- Dridex: 100.00%
- MinerTrojan: 92.50%
- Ramnit: 98.05%
- benign: 92.04%

## Configuration

Edit `netml/config.py` to adjust:

```python
# Model hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 3
EMBEDDING_DIM = 64
NUM_HEADS = 4

# Training settings
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 25

# Data paths
DATA_DIR = "Data"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
```

## API Usage

```python
from netml.data.dataset import NetMLDataset
from netml.models.encoder import NetMLEncoder
from netml.training.trainer import Trainer, HybridTrainer
from netml.evaluation.metrics import compute_metrics

# Load data
dataset = NetMLDataset("Data/Master.json")
train_loader, val_loader = dataset.get_loaders(batch_size=512)

# Create and train encoder
encoder = NetMLEncoder(input_dim=76, hidden_dim=128, embedding_dim=64)
trainer = Trainer(encoder, train_loader, val_loader)
trainer.train(epochs=25)

# Train hybrid classifier
hybrid_trainer = HybridTrainer(encoder, classifier_type="ensemble")
hybrid_trainer.fit(train_embeddings, train_labels)

# Evaluate
metrics = compute_metrics(predictions, labels)
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

## Files Generated

- **Checkpoints**: `checkpoints/*.pth`, `checkpoints/*.json`
- **Results**: `results/*.json`, `results/visualizations/*.png`
- **Figures**: `results/paper_figures/*.pdf` (publication-ready)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourname2024netml,
  title={Hybrid Deep Learning for Network Traffic Classification},
  author={Your Name},
  booktitle={Conference Name},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- NetML-2020 dataset: [Link to dataset]
- PyTorch, XGBoost, LightGBM, scikit-learn
