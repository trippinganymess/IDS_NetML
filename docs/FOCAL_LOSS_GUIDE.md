# Focal Loss Implementation for Class Imbalance

## Overview

Focal Loss has been integrated into your training pipeline to better handle **class imbalance** in your 21-class malware classification dataset.

---

## Problem: Class Imbalance

Your dataset has imbalanced classes (from the earlier analysis):
- **benign**: ~1,011 samples
- **Trickster**: ~736 samples  
- **Emotet**: ~592 samples
- ...
- **Dridex**: ~137 samples (and other rare classes)

**Standard CrossEntropyLoss** treats all classes equally:
- Easy examples (high confidence) still contribute significantly to loss
- Rare classes are overwhelmed by frequent classes
- Model learns to ignore hard-to-classify minority classes

---

## Solution: Focal Loss

Focal Loss was introduced in 2017 (Lin et al., "Focal Loss for Dense Object Detection") to address exactly this problem.

### How It Works

**Standard CrossEntropyLoss:**
$$L_{ce} = -\log(p_t)$$

where $p_t$ is the probability of the correct class.

**Focal Loss:**
$$L_{focal} = -\alpha (1 - p_t)^\gamma \log(p_t)$$

### Key Components

| Parameter | Default | Purpose |
|-----------|---------|---------|
| **alpha** | 1.0 | Weighting factor to balance class importance |
| **gamma** | 2.0 | Focusing parameter (exponent) |
| **reduction** | 'mean' | How to aggregate loss ('mean', 'sum', 'none') |

### The Magic: $(1 - p_t)^\gamma$

This term **down-weights easy examples** and **up-weights hard examples**:

- When model is **confident and correct** ($p_t = 0.9$):
  - Modifier: $(1 - 0.9)^2 = 0.01$ → Loss reduced to 1% of original
  - Less focus on already-learned examples

- When model is **uncertain or wrong** ($p_t = 0.5$):
  - Modifier: $(1 - 0.5)^2 = 0.25$ → Loss reduced to 25% of original
  - Still focuses on hard examples

- When model is **very wrong** ($p_t = 0.1$):
  - Modifier: $(1 - 0.1)^2 = 0.81$ → Loss reduced to 81% of original
  - Maximum focus on misclassified samples

---

## Usage in Your Script

```python
# Initialize Focal Loss
criterion = FocalLoss(alpha=1.0, gamma=2.0)

# Use in training (same as CrossEntropyLoss)
loss = criterion(logits, targets)  # logits: (batch, 21), targets: (batch,)
loss.backward()
optimizer.step()
```

### Adjusting Hyperparameters

```python
# More aggressive focusing on hard examples
criterion = FocalLoss(alpha=1.0, gamma=3.0)  # Higher gamma = more focus

# Balance class weights (if one class is much rarer)
criterion = FocalLoss(alpha=2.0, gamma=2.0)  # Higher alpha = stronger weighting
```

---

## Benefits for Your Dataset

✅ **Better minority class performance:** Rare malware families (Dridex, etc.) get more attention

✅ **Faster convergence:** Hard examples are learned first, easy ones later

✅ **More balanced F1-score:** Reduces the class imbalance bias toward frequent classes

✅ **Better generalization:** Forces the model to learn distinctive features of minority classes

---

## Expected Training Changes

| Aspect | CrossEntropyLoss | FocalLoss |
|--------|------------------|-----------|
| Loss values | Larger | Often smaller (because hard examples weighted more) |
| Convergence | May favor majority classes early | Balanced learning from epoch 1 |
| Minority class recall | Lower (10-30% for rare classes) | Higher (40-60% for rare classes) |
| Overall accuracy | May be slightly higher | Often lower, but more balanced across classes |

---

## Tuning Gamma ($\gamma$)

### $\gamma = 0$ (equivalent to CrossEntropyLoss)
```
Loss = - log(p_t)  [no focusing effect]
```
Use when: Classes are balanced

### $\gamma = 1$ (moderate focusing)
```
Loss = -(1 - p_t) * log(p_t)
```
Use when: Mild class imbalance

### $\gamma = 2$ (strong focusing) ← **Current setting**
```
Loss = -(1 - p_t)^2 * log(p_t)
```
Use when: Severe class imbalance (your case!)

### $\gamma = 3$ (very strong focusing)
```
Loss = -(1 - p_t)^3 * log(p_t)
```
Use when: Extreme class imbalance with many rare classes

---

## Monitoring Training

Your training script now reports:
```
Epoch 1 Results:
  Train Loss: 2.1453, Train Acc: 68.23%
  Val Loss:   1.9876, Val Acc:   72.15%
```

**What to look for:**
- Loss should decrease (Focal Loss often starts lower than CrossEntropyLoss)
- Accuracy should increase steadily
- Validation accuracy should plateau after 5-10 epochs (if data is sufficient)
- If val accuracy plateaus too early → increase EPOCHS or reduce patience

---

## If Training Results Are Poor

### Loss not decreasing:
1. Increase `gamma` (e.g., 2.0 → 3.0) to focus more on hard examples
2. Reduce learning rate (LR: 0.001 → 0.0005)
3. Check data normalization (verify scalars are z-normalized)

### Validation accuracy plateaus early:
1. Increase `EPOCHS` (10 → 30)
2. Reduce `patience` (7 → 5) to catch best model earlier
3. Try data augmentation or mixup

### Minority classes still have low recall:
1. Increase `alpha` (1.0 → 2.0)
2. Increase `gamma` (2.0 → 3.0)
3. Consider weighted sampler (oversample rare classes)

---

## Literature Reference

**Paper:** "Focal Loss for Dense Object Detection"
- **Authors:** Lin, Goyal, Girshick, He, Dollár (Facebook AI Research)
- **Year:** 2017 (ICCV)
- **Citation:** Used in state-of-the-art object detection and class imbalance tasks

The loss has been proven effective for:
- Imbalanced datasets
- Multi-class classification
- Hard example mining
- One-stage detection models

---

## Quick Reference

```python
# In your training script:

# OLD (CrossEntropyLoss):
# criterion = nn.CrossEntropyLoss()

# NEW (Focal Loss):
criterion = FocalLoss(alpha=1.0, gamma=2.0)

# Training remains the same:
for epoch in range(EPOCHS):
    # ... training code ...
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
```

No other changes needed! Focal Loss is a drop-in replacement for CrossEntropyLoss.
