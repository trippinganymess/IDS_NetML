# 📊 F1 Scores Summary - Quick Reference

## ✅ What You Asked For

You requested **the last training F1 scores for all classes** and mentioned **the confusion matrix might be wrong**.

### Answer: Yes, the confusion matrix reveals real problems!

---

## 🎯 The Bottom Line

**Your model has a serious problem**: Several malware families are **completely undetectable** (0% recall):
- ❌ CCleaner - 0/1268 detected
- ❌ MagicHound - 0/615 detected  
- ❌ WebCompanion - 0/656 detected
- ❌ Trickster - 0/7 detected

**This is NOT a visualization issue** - your confusion matrix is showing the real problem correctly.

---

## 📈 Performance Tiers

### 🏆 Excellent (>90% F1)
- BitCoinMiner: **F1 = 0.9792**
- Dridex: **F1 = 0.9837**
- Artemis: **F1 = 0.9598**
- Ramnit: **F1 = 0.9636**
- benign: **F1 = 0.9593**

### ✅ Good (70-90% F1)
- Adload: **F1 = 0.7060**
- Emotet: **F1 = 0.7359** (but only 62.57% recall)

### ⚠️ Poor (50-70% F1)
- Sality: **F1 = 0.8916**
- MinerTrojan: **F1 = 0.9358**
- TrojanDownloader: **F1 = 0.6813** (99.56% recall, but only 51.78% precision)
- Tinba: **F1 = 0.5064**

### 🔴 Critical (<50% F1)
- Emotet: **F1 = 0.7359** recall only 62.57%
- Cobalt: **F1 = 0.3325** (only 37.75% detected)
- TrickBot: **F1 = 0.3043** (only 32.22% detected)
- HTBot: **F1 = 0.2552** (only 29.52% detected)
- Ursnif: **F1 = 0.2509** (only 15.89% detected)
- PUA: **F1 = 0.2272** (only 14.90% detected)
- Downware: **F1 = 0.0430** (only 2.59% detected)
- **CCleaner: F1 = 0.0000** (0% detected)
- **MagicHound: F1 = 0.0000** (0% detected)
- **WebCompanion: F1 = 0.0000** (0% detected)
- **Trickster: F1 = 0.0000** (0% detected)

---

## 🔍 The Confusion Matrix Problem

### What's Actually Wrong (Not a display issue):

1. **CCleaner completely misclassified**:
   - 1,259 out of 1,268 CCleaner samples (99.3%) → Predicted as **TrojanDownloader**
   - **Cause**: CCleaner and TrojanDownloader features are too similar, or CCleaner is mislabeled

2. **Cobalt is a "black hole"**:
   - Many classes end up classified as Cobalt:
     - 3,267 samples from Cobalt itself
     - 378 from HTBot
     - 228 from MagicHound
     - 209 from PUA
   - **Cause**: Model learned Cobalt features too broadly

3. **Bidirectional confusion**:
   - Cobalt ↔ Adload: Both confused with each other (55.5% in one direction, 20.3% the other)
   - **Cause**: These families have very similar feature distributions

---

## 💡 Why Confusion Matrix Might "Seem Wrong"

**Your confusion matrix is CORRECT**, but here's what may seem surprising:

1. **Zero diagonal elements** (e.g., CCleaner row shows 0 detections):
   - This is real - not a bug
   - Model genuinely can't identify these classes

2. **Off-diagonal spikes**:
   - CCleaner column has 1,259 in TrojanDownloader row
   - This means: "Predicted as TrojanDownloader when actually CCleaner"
   - This is exactly what the confusion matrix should show!

3. **High precision but low recall** (e.g., Emotet):
   - Emotet: 89.32% precision, 62.57% recall
   - Means: When model says "Emotet", it's usually right (89%), but it only finds 62% of true Emotets
   - This is correct behavior to display!

**Conclusion**: Your confusion matrix is displaying the reality correctly. The problem is with the MODEL, not the visualization.

---

## 📊 Per-Class F1 Scores (All 21)

```
✓ BitCoinMiner:    0.9792   (Excellent)
✓ Dridex:          0.9837   (Excellent)
✓ Artemis:         0.9598   (Excellent)
✓ Ramnit:          0.9636   (Excellent)
✓ benign:          0.9593   (Excellent)
✓ Sality:          0.8916   (Very Good)
✓ MinerTrojan:     0.9358   (Very Good)
✓ Adload:          0.7060   (Good)
✓ Emotet:          0.7359   (Good)
⚠️ TrojanDownloader: 0.6813  (Okay but high false positives)
⚠️ Tinba:          0.5064   (Poor)
🔴 Cobalt:         0.3325   (Critical)
🔴 TrickBot:       0.3043   (Critical)
🔴 HTBot:          0.2552   (Critical)
🔴 Ursnif:         0.2509   (Critical)
🔴 PUA:            0.2272   (Critical)
🔴 Downware:       0.0430   (Critical)
🔴 CCleaner:       0.0000   (Not detected)
🔴 MagicHound:     0.0000   (Not detected)
🔴 WebCompanion:   0.0000   (Not detected)
🔴 Trickster:      0.0000   (Not detected)
```

---

## 🎯 Top 3 Immediate Issues to Fix

### Issue #1: CCleaner Disappears (F1=0.0)
- **Problem**: All 1,268 CCleaner samples predicted as TrojanDownloader
- **Fix**: 
  - Check if CCleaner in training data is actually TrojanDownloader
  - Or: Reduce model's confidence in TrojanDownloader predictions
  - Test: Lower TrojanDownloader output threshold

### Issue #2: Cobalt is Too Generalized
- **Problem**: Model predicts Cobalt for everything (3,267 Cobalt + 378 HTBot + 228 MagicHound + ...)
- **Fix**:
  - Increase Cobalt decision threshold
  - Check if Cobalt features overlap with other families
  - Consider oversampling other classes

### Issue #3: Small Classes Disappear (MagicHound, WebCompanion, Trickster)
- **Problem**: Very few training samples → Model can't learn them
- **Fix**:
  - Oversample these classes
  - Use SMOTE for synthetic oversampling
  - Increase class weights in training loss

---

## 📋 Summary Statistics

| Metric | Value |
|--------|-------|
| Overall Accuracy | **78.68%** |
| Macro F1 Score | **0.5103** (unweighted) |
| Weighted F1 Score | **0.7752** (realistic) |
| Classes Perfect (>95%) | 5 out of 21 |
| Classes Critical (<50%) | 11 out of 21 |
| Classes with 0% Recall | 4 out of 21 |

---

## 📁 Files Generated

1. **TRAINING_METRICS_REPORT.md** - Full detailed analysis
2. **f1_scores_latest.json** - Programmatic access to all metrics
3. **get_f1_scores.py** - Script to regenerate these metrics

---

## ✍️ Next Steps

1. **Review the full report**: `TRAINING_METRICS_REPORT.md`
2. **Check if confusion matrix numbers match expectations**:
   - Row = true label, Column = predicted label
   - Diagonal = correct predictions
   - Off-diagonal = misclassifications
3. **Decide on fixes**:
   - Data quality issues (mislabeling)?
   - Model capacity issues?
   - Class imbalance?
4. **Test fixes** by re-running: `python get_f1_scores.py`

---

**Generated**: December 15, 2025  
**Model**: encoder_best.pth (Embedding Dim: 64)  
**Dataset**: 350,000 training + 70,000 validation samples
