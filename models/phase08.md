# 📘 Phase 08 – Regression Analysis

**Project:** Spotify Music Success Prediction

## 🎯 Objective

Predict the exact popularity score (0-100) instead of binary classification, and compare regression vs classification approaches.

## 📋 Regression vs Classification

| Approach | Output | Use Case |
|----------|--------|----------|
| **Classification** | Hit (1) or Non-Hit (0) | Simple decision making |
| **Regression** | Popularity score (0-100) | Granular ranking, prioritization |

## 🤖 Regression Models Tested

| Model | Description |
|-------|-------------|
| Linear Regression | Basic linear model |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization (feature selection) |
| ElasticNet | L1 + L2 combined |
| Random Forest Regressor | Ensemble of trees |
| Gradient Boosting Regressor | Sequential boosting |
| XGBoost Regressor | Optimized gradient boosting |
| Neural Network (MLP) | Multi-layer perceptron |

## 📊 Results

### Regression Metrics

| Model | MAE | RMSE | R² | CV MAE |
|-------|-----|------|-----|--------|
| XGBoost | 13.2 | 17.8 | 0.28 | 13.5 |
| Random Forest | 13.5 | 18.2 | 0.26 | 13.8 |
| Gradient Boosting | 13.8 | 18.5 | 0.25 | 14.1 |
| Neural Network | 14.2 | 19.0 | 0.23 | 14.5 |
| Ridge | 15.1 | 19.8 | 0.19 | 15.3 |
| ElasticNet | 15.2 | 19.9 | 0.19 | 15.4 |
| Linear Regression | 15.3 | 20.0 | 0.18 | 15.5 |
| Lasso | 15.5 | 20.2 | 0.17 | 15.7 |

### Interpretation

**Best Model: XGBoost**
- **MAE = 13.2:** On average, predictions are off by 13 popularity points
- **R² = 0.28:** Model explains 28% of variance in popularity
- **RMSE = 17.8:** Penalizes larger errors more

### Why R² is "Low"

R² of 0.28 might seem poor, but consider:
1. **Popularity is inherently unpredictable** - marketing, timing, luck
2. **Audio features capture ~30% of what makes a hit**
3. **External factors** (artist fame, promotion, timing) not in data
4. **Subjectivity** - popularity is partly random

## 📉 Residual Analysis

### Error Distribution (XGBoost)
```
Mean Error: 0.2 (nearly unbiased)
Std of Errors: 17.8
95% of predictions within: ±35 popularity points
```

### Error Patterns
- Model tends to **overpredict low-popularity songs**
- Model tends to **underpredict high-popularity songs**
- **Regression to the mean** effect visible

### Actual vs Predicted Plot
- Points should lie on diagonal line
- Spread increases at extremes (0 and 100)
- Tighter clustering in middle range (40-60)

## 🔄 Regression → Classification

Converting regression predictions to binary classification:
- **Threshold:** If predicted ≥ 70 → Hit, else Non-Hit

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.80 | 0.42 | 0.48 | 0.45 |
| Random Forest | 0.79 | 0.40 | 0.45 | 0.42 |
| Gradient Boosting | 0.78 | 0.39 | 0.46 | 0.42 |
| Neural Network | 0.77 | 0.37 | 0.50 | 0.42 |

**Key Finding:** Classification from regression performs similarly to direct classification models.

## 💡 Regression vs Classification: Which to Use?

### Use Classification When:
- ✅ Simple yes/no decision needed
- ✅ Clear threshold exists (e.g., "hit" = top 20%)
- ✅ Interpretability important
- ✅ Small differences don't matter

### Use Regression When:
- ✅ Need to rank/prioritize songs
- ✅ Want granular predictions
- ✅ Building recommendation systems
- ✅ Optimizing playlists by expected popularity

### Hybrid Approach
1. Use **regression** for ranking candidates
2. Apply **threshold** for final decision
3. Get benefits of both approaches

## 📊 Feature Importance (Regression)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | instrumentalness | 0.148 |
| 2 | loudness | 0.132 |
| 3 | speechiness | 0.118 |
| 4 | duration_ms | 0.105 |
| 5 | danceability | 0.098 |
| 6 | energy | 0.092 |
| 7 | acousticness | 0.085 |
| 8 | valence | 0.078 |
| 9 | tempo | 0.072 |
| 10 | liveness | 0.062 |

Similar to classification importance - **instrumentalness, loudness, speechiness** remain top predictors.

## 💡 Key Insights

### 1. Popularity is Hard to Predict
- Maximum R² of 0.28 with best model
- ~70% of variance unexplained by audio features
- Non-audio factors dominate success

### 2. Prediction Error Context
MAE of 13 points means:
- Predicted 50 → Actual likely 37-63
- Useful for ranking, not exact prediction
- Good enough for A&R screening

### 3. Linear vs Non-Linear
- Tree-based models outperform linear by ~15%
- Non-linear relationships exist in data
- Feature interactions captured by ensembles

### 4. Practical Use Cases

| MAE | Good For | Not Good For |
|-----|----------|--------------|
| 13 points | Ranking 100 songs | Predicting exact chart position |
| 13 points | Identifying top quartile | Distinguishing 65 vs 75 |
| 13 points | A&R screening | Marketing budget allocation |

## 📁 Outputs

- `outputs/regression/01_regression_metrics.png`
- `outputs/regression/02_actual_vs_predicted.png`
- `outputs/regression/03_residual_plots.png`
- `outputs/regression/04_error_distribution.png`
- `outputs/regression/05_regression_vs_classification.png`
- `outputs/regression/06_regression_feature_importance.png`
- `outputs/regression/regression_results.csv`

## 📁 Script

```bash
python models/phase08_regression.py
```

## ✅ Phase 08 Complete

This concludes the core analysis phases. See the **README** for project summary and conclusions.
