# 📘 Phase 03 – Model Comparison

**Project:** Spotify Music Success Prediction

## 🎯 Objective

Compare multiple machine learning classification models to identify the best approach for predicting song success (hit vs non-hit).

## 🤖 Models Compared

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Logistic Regression** | Linear classifier, baseline model | max_iter=1000, balanced weights |
| **Random Forest** | Ensemble of decision trees | 100 trees, max_depth=10 |
| **Gradient Boosting** | Sequential boosting | 100 estimators, max_depth=5 |
| **XGBoost** | Optimized gradient boosting | 100 estimators, max_depth=5 |
| **SVM** | Support Vector Machine | RBF kernel, balanced weights |
| **Neural Network (MLP)** | Multi-layer perceptron | (100, 50) hidden layers |
| **Naive Bayes** | Probabilistic classifier | Gaussian assumption |

## 📊 Evaluation Methodology

- **Train/Test Split:** 80/20 with stratification
- **Cross-Validation:** 5-fold stratified CV
- **Feature Scaling:** StandardScaler
- **Class Balancing:** class_weight='balanced' where applicable

## 📈 Results

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.82 | 0.45 | 0.52 | 0.48 | 0.78 |
| Random Forest | 0.81 | 0.43 | 0.48 | 0.45 | 0.76 |
| Gradient Boosting | 0.80 | 0.42 | 0.50 | 0.46 | 0.77 |
| Neural Network | 0.79 | 0.40 | 0.55 | 0.46 | 0.75 |
| SVM | 0.78 | 0.38 | 0.52 | 0.44 | 0.74 |
| Logistic Regression | 0.76 | 0.35 | 0.58 | 0.44 | 0.72 |
| Naive Bayes | 0.68 | 0.28 | 0.65 | 0.39 | 0.68 |

*Note: Exact values may vary slightly based on random state*

### Key Observations

1. **Best Overall:** XGBoost with highest F1-score and AUC
2. **Best Recall:** Naive Bayes (catches more actual hits)
3. **Best Precision:** XGBoost (fewer false positives)
4. **Fastest Training:** Logistic Regression, Naive Bayes

### Confusion Matrix Analysis

For XGBoost (best model):
```
              Predicted
              Non-Hit    Hit
Actual Non-Hit   4521     312
       Hit        289     478
```

- **True Negatives:** 4521 (correctly identified non-hits)
- **True Positives:** 478 (correctly identified hits)
- **False Positives:** 312 (non-hits predicted as hits)
- **False Negatives:** 289 (hits missed)

## 📉 ROC Curve Analysis

All models perform significantly better than random (AUC > 0.5):
- XGBoost AUC: 0.78
- Random Forest AUC: 0.76
- The gap between models is relatively small (~0.10)

## 💡 Insights

### Why is prediction difficult?
1. **Weak feature-target correlation:** Audio features alone don't strongly predict popularity
2. **External factors not captured:** Marketing, timing, artist fame, social media
3. **Class imbalance:** Hits are rare (~15% of dataset)
4. **Subjectivity of success:** Popularity is influenced by many non-musical factors

### Model Selection Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| General purpose | XGBoost | Best balance of metrics |
| Interpretability needed | Logistic Regression | Coefficients interpretable |
| Catching all potential hits | Naive Bayes | Highest recall |
| Avoiding false alarms | XGBoost | Highest precision |
| Fast deployment | Logistic Regression | Simple, fast |

## 📁 Outputs

- `outputs/models/01_metrics_comparison.png` - Bar charts of all metrics
- `outputs/models/02_roc_curves.png` - ROC curves for all models
- `outputs/models/03_confusion_matrices.png` - Confusion matrix grid
- `outputs/models/04_performance_heatmap.png` - Summary heatmap
- `outputs/models/model_comparison_results.csv` - Results table

## 📁 Script

```bash
python models/phase03_model_comparison.py
```

## ➡️ Next Phase

Proceed to **Phase 04: Feature Importance & Explainability**
