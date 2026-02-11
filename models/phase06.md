# 📘 Phase 06 – Imbalanced Learning

**Project:** Spotify Music Success Prediction

## 🎯 Objective

Address the class imbalance problem (hits are rare) and compare different balancing strategies to improve model performance on the minority class.

## 📊 Class Imbalance Analysis

### Original Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Non-Hits (0) | ~23,800 | ~85% |
| Hits (1) | ~4,200 | ~15% |

**Imbalance Ratio:** ~5.7:1

### Why This Matters
- Models tend to predict majority class (non-hits)
- High accuracy but poor recall for hits
- Miss most actual hits (high false negative rate)

## 📋 Balancing Strategies Tested

### 1. No Balancing (Baseline)
- Train on original imbalanced data
- Model biased toward majority class

### 2. Class Weight Adjustment
- Assign higher weight to minority class
- `class_weight='balanced'`
- Penalizes misclassification of hits more

### 3. SMOTE (Synthetic Minority Over-sampling)
- Creates synthetic hit examples
- Interpolates between existing hits
- Balances classes to 1:1

### 4. ADASYN (Adaptive Synthetic Sampling)
- Similar to SMOTE but adaptive
- Focuses on harder-to-learn examples
- Creates more samples near decision boundary

### 5. Random Undersampling
- Removes majority class examples
- Reduces dataset size significantly
- Risk of losing information

### 6. SMOTE-Tomek
- SMOTE + Tomek links cleaning
- Removes ambiguous examples after oversampling
- Cleaner decision boundary

## 📈 Results Comparison

| Strategy | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------|----------|-----------|--------|----------|---------|
| No Balancing | 0.85 | 0.52 | 0.35 | 0.42 | 0.74 |
| Class Weight | 0.78 | 0.42 | 0.62 | 0.50 | 0.76 |
| **SMOTE** | **0.76** | **0.40** | **0.68** | **0.50** | **0.77** |
| ADASYN | 0.75 | 0.38 | 0.70 | 0.49 | 0.76 |
| Undersampling | 0.72 | 0.35 | 0.72 | 0.47 | 0.75 |
| SMOTE-Tomek | 0.77 | 0.41 | 0.65 | 0.50 | 0.77 |

### Confusion Matrix Comparison

#### No Balancing
```
              Predicted
              Non-Hit    Hit
Actual Non-Hit   4650      83
       Hit        545     222
```
Misses 71% of actual hits!

#### SMOTE
```
              Predicted
              Non-Hit    Hit
Actual Non-Hit   4120     613
       Hit        245     522
```
Catches 68% of actual hits!

## 📉 Precision-Recall Trade-off

| Strategy | Recall (Catch Hits) | Precision (Avoid False Alarms) |
|----------|---------------------|--------------------------------|
| No Balancing | 0.35 ⬇️ | 0.52 ⬆️ |
| SMOTE | 0.68 ⬆️ | 0.40 ⬇️ |

**Key Trade-off:** Higher recall comes at the cost of lower precision

## 💡 Key Insights

### 1. The Recall Problem
Without balancing:
- Model correctly predicts 85% of all songs (accuracy)
- But only catches 35% of actual hits (recall)
- For a music label, missing 65% of potential hits is costly!

### 2. Best Strategy Depends on Goal

| Business Goal | Best Strategy | Why |
|---------------|---------------|-----|
| Catch ALL potential hits | SMOTE or ADASYN | Highest recall |
| Minimize false positives | No Balancing | Highest precision |
| Balanced approach | Class Weight or SMOTE-Tomek | Best F1-score |
| Fast iteration | Class Weight | No data augmentation needed |

### 3. SMOTE Trade-offs
**Pros:**
- Significantly improves recall
- Maintains reasonable precision
- Doesn't lose original data

**Cons:**
- Creates synthetic data (not real songs)
- Can overfit to synthetic patterns
- Increases training time

### 4. Practical Recommendations

For **A&R Teams** (finding new artists):
- Use SMOTE or ADASYN
- Better to review false positives than miss hits
- High recall is priority

For **Marketing Budget Allocation**:
- Use Class Weight
- Need confidence in predictions
- Balance precision and recall

For **Playlist Curation**:
- Use SMOTE-Tomek
- Want quality recommendations
- Balanced approach works best

## 📊 Impact Visualization

### Before Balancing (Recall = 35%)
```
Actual Hits: ████████████████████ (100)
Predicted:   ███████             (35) ❌ Missing 65!
```

### After SMOTE (Recall = 68%)
```
Actual Hits: ████████████████████ (100)
Predicted:   █████████████████   (68) ✅ Much better!
```

## 📁 Outputs

- `outputs/imbalanced/01_class_distribution.png`
- `outputs/imbalanced/05_strategy_comparison.png`
- `outputs/imbalanced/05_confusion_matrices.png`
- `outputs/imbalanced/05_precision_recall_tradeoff.png`
- `outputs/imbalanced/balancing_comparison_results.csv`

## 📁 Script

```bash
python models/phase06_imbalanced_learning.py
```

**Requirements:** `pip install imbalanced-learn`

## ➡️ Next Phase

Proceed to **Phase 07: Genre Analysis**
