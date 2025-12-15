# Training Metrics Analysis Report

**Generated**: December 15, 2025  
**Model**: encoder_best.pth (Embedding Dim: 64)  
**Dataset**: 350,000 training samples, 70,000 validation samples  
**Classes**: 21 malware families + benign

---

## Executive Summary

### Overall Performance
- **Overall Accuracy**: 78.68%
- **Macro F1 Score**: 0.5103 (unweighted average - treats all classes equally)
- **Weighted F1 Score**: 0.7752 (weighted by class size - more realistic)

### Key Findings
✅ **Excellent Performance**: BitCoinMiner (99.21%), Dridex (99.96%), TrojanDownloader (99.56%)
⚠️ **Critical Issues**: Several classes have 0% recall (not detected at all)
🔴 **Major Imbalance**: Significant confusion between certain class pairs

---

## Per-Class F1 Scores (All 21 Classes)

| Class | F1-Score | Recall | Precision | Support | Status |
|-------|----------|--------|-----------|---------|--------|
| Adload | 0.7060 | 75.01% | 66.68% | 8,985 | ✓ Good |
| Artemis | 0.9598 | 96.29% | 95.67% | 2,962 | ✓ Excellent |
| BitCoinMiner | 0.9792 | 99.21% | 96.66% | 21,649 | ✓ Excellent |
| **CCleaner** | **0.0000** | **0.00%** | 0% | 1,268 | 🔴 **Not Detected** |
| **Cobalt** | **0.3325** | **37.75%** | 29.70% | 5,888 | 🔴 **Critical** |
| **Downware** | **0.0430** | **2.59%** | 12.60% | 617 | 🔴 **Critical** |
| Dridex | 0.9837 | 99.96% | 96.82% | 7,074 | ✓ Excellent |
| **Emotet** | **0.7359** | **62.57%** | 89.32% | 2,367 | ⚠️ Poor Recall |
| **HTBot** | **0.2552** | **29.52%** | 22.48% | 996 | 🔴 **Critical** |
| **MagicHound** | **0.0000** | **0.00%** | 0% | 615 | 🔴 **Not Detected** |
| MinerTrojan | 0.9358 | 88.06% | 99.84% | 1,415 | ✓ Very Good |
| **PUA** | **0.2272** | **14.90%** | 47.77% | 718 | 🔴 **Critical** |
| Ramnit | 0.9636 | 97.20% | 95.54% | 4,714 | ✓ Excellent |
| Sality | 0.8916 | 81.43% | 98.52% | 4,825 | ✓ Very Good |
| **Tinba** | **0.5064** | **51.38%** | 49.92% | 2,326 | ⚠️ Poor |
| **TrickBot** | **0.3043** | **32.22%** | 28.84% | 1,282 | 🔴 **Critical** |
| **Trickster** | **0.0000** | **0.00%** | 0% | 7 | 🔴 **Not Detected** |
| TrojanDownloader | 0.6813 | 99.56% | 51.78% | 1,358 | ⚠️ High False Positives |
| **Ursnif** | **0.2509** | **15.89%** | 59.65% | 214 | 🔴 **Critical** |
| **WebCompanion** | **0.0000** | **0.00%** | 0% | 656 | 🔴 **Not Detected** |
| benign | 0.9593 | 92.19% | 100% | 64 | ✓ Excellent |

**Legend**: ✓ = Good (>70%) | ⚠️ = Acceptable (50-70%) | 🔴 = Critical (<50%)

---

## Confusion Matrix Issues

### Classes with ZERO Recall (Not Being Detected at All)
1. **CCleaner** (0/1268) → Being classified as **TrojanDownloader** (1259 times, 99.3%)
2. **MagicHound** (0/615) → Being classified as **Cobalt** (228x) or **HTBot** (195x)
3. **WebCompanion** (0/656) → Being classified as **Cobalt** (231x) or **HTBot** (148x)
4. **Trickster** (0/7) → Being classified as **Tinba** (4x)

**Impact**: These classes are completely invisible to your model. Any samples from these families will be misclassified as something else.

### Top Misclassifications (Confusion Pairs)

**Most Confused Pair**:
- **Cobalt ↔ Adload** (Bidirectional confusion)
  - 3,267 Cobalt samples predicted as Adload (55.5% of Cobalt)
  - 1,821 Adload samples predicted as Cobalt (20.3% of Adload)
  - **Root cause**: Likely similar feature distributions

**Other Major Confusions**:
- Cobalt → Adload: 55.5% of true Cobalt cases
- Tinba → Cobalt: 38.7% of true Tinba cases
- TrickBot → Cobalt: 37.4% of true TrickBot cases
- HTBot → Cobalt: 38.0% of true HTBot cases

**Pattern**: Many classes are being misclassified as **Cobalt**, suggesting the model has learned Cobalt's features too broadly.

---

## Problem Analysis

### Problem 1: Severely Imbalanced Classes
Some classes have very few training samples:
- Trickster: 7 samples
- Ursnif: 214 samples
- Downware: 617 samples
- WebCompanion: 656 samples
- MagicHound: 615 samples

**Impact**: Model struggles to learn these rare classes.

### Problem 2: Class Overlap/Similarity
Certain malware families have very similar feature distributions (Cobalt, Adload, Tinba, HTBot):
- They attack similar targets
- They use similar protocols/payloads
- Similar network behavior

**Impact**: Model cannot reliably distinguish between them.

### Problem 3: Two-Way Confusion
When model confuses A with B AND confuses B with A:
- Adload ↔ Cobalt
- Shows these classes are genuinely difficult to separate

---

## Recommendations

### Quick Wins (Easy Fixes)
1. **Fix CCleaner** (0% recall):
   - Currently 99.3% predicted as TrojanDownloader
   - Check if these are truly different or mislabeled in training data
   - Likely just needs threshold adjustment

2. **Fix TrojanDownloader** (High false positives):
   - 51.78% precision means many false positives
   - Model is too aggressive with this prediction
   - Need to increase decision threshold

3. **Balance underrepresented classes**:
   - Oversample: Trickster, Ursnif, Downware, WebCompanion, MagicHound
   - Use techniques like SMOTE or class-weighted loss

### Medium-Term Fixes
1. **Extract better features**:
   - Current features may not capture differences between Cobalt/Adload/Tinba
   - Consider adding: Protocol ratios, temporal patterns, application-layer data

2. **Use ensemble methods**:
   - Train separate models for high-confusion pairs
   - Route through ensemble to disambiguate

3. **Investigate feature engineering**:
   - Which features distinguish Cobalt from Adload?
   - Which features distinguish HTBot/TrickBot from others?

### Long-Term Improvements
1. **Collect more data for rare classes**:
   - Current 7 Trickster samples insufficient
   - Target collection for Downware, WebCompanion, MagicHound

2. **Multi-task learning**:
   - Predict both family AND characteristics (payload type, protocol, target)
   - Use auxiliary tasks to improve feature learning

3. **Focal loss or class weighting**:
   - Already using FocalLoss (good!)
   - But may need to tune gamma and class weights more aggressively

---

## JSON Output File

Full F1 scores saved to: `results/f1_scores_latest.json`

Contains:
- Per-class F1 scores for all 21 classes
- Macro F1, Weighted F1, and accuracy metrics
- Class names for reference

---

## How to Use This Report

**For debugging**: Focus on the "Top Misclassifications" section - understand what the model is predicting instead of what it should

**For improvement**: Start with the "Quick Wins" section - fixing CCleaner and TrojanDownloader threshold should immediately improve metrics

**For long-term**: The "Recommendations" section provides a roadmap for architectural improvements

---

**Model Location**: `/Users/animesh/IDS_NetML/src/encoder_best.pth`  
**Analysis Script**: `get_f1_scores.py`  
**Last Updated**: December 15, 2025
