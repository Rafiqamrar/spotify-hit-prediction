# 📘 Phase 04 – Feature Importance & Explainability

**Project:** Spotify Music Success Prediction

## 🎯 Objective

Understand which audio features contribute most to predicting song success and explain individual predictions using interpretability techniques.

## 📋 Methods Used

### 1. Random Forest Feature Importance
- Based on mean decrease in Gini impurity
- Measures how much each feature contributes to node splits

### 2. Permutation Importance
- Measures accuracy decrease when feature values are shuffled
- Model-agnostic approach

### 3. XGBoost Feature Importance
- Based on information gain
- Captures feature contribution in gradient boosting

### 4. SHAP (SHapley Additive exPlanations)
- Game-theoretic approach to feature attribution
- Provides both global and local explanations

### 5. LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions
- Creates local linear approximations

## 📊 Results

### Global Feature Importance Ranking

| Rank | Feature | RF Importance | Permutation | XGBoost | Consensus |
|------|---------|---------------|-------------|---------|-----------|
| 1 | **instrumentalness** | 0.142 | 0.031 | 0.158 | ⭐ Top |
| 2 | **loudness** | 0.128 | 0.028 | 0.134 | ⭐ Top |
| 3 | **speechiness** | 0.115 | 0.024 | 0.112 | ⭐ Top |
| 4 | **danceability** | 0.108 | 0.019 | 0.095 | High |
| 5 | **energy** | 0.102 | 0.018 | 0.088 | High |
| 6 | **duration_ms** | 0.098 | 0.015 | 0.092 | Medium |
| 7 | **acousticness** | 0.089 | 0.012 | 0.078 | Medium |
| 8 | **valence** | 0.082 | 0.011 | 0.072 | Medium |
| 9 | **tempo** | 0.075 | 0.009 | 0.065 | Lower |
| 10 | **liveness** | 0.061 | 0.007 | 0.056 | Lower |

### SHAP Analysis Insights

#### Direction of Feature Effects:
- **Instrumentalness LOW** → More likely to be a hit
- **Loudness HIGH** → More likely to be a hit
- **Speechiness MODERATE** → Optimal for hits
- **Danceability HIGH** → Slightly increases hit probability
- **Energy HIGH** → Slightly increases hit probability
- **Acousticness LOW** → More likely to be a hit

#### SHAP Dependence Plots Key Findings:
1. Songs with instrumentalness > 0.5 rarely become hits
2. Optimal loudness range: -8 to -4 dB
3. Speechiness sweet spot: 0.05 to 0.15

### LIME Example Explanations

#### Example: Predicted HIT (probability = 0.82)
```
Contributing factors:
+ instrumentalness = 0.001  (strong positive)
+ loudness = -5.2 dB        (positive)
+ danceability = 0.78       (positive)
- speechiness = 0.35        (negative - too high)
+ energy = 0.85             (positive)
```

#### Example: Predicted NON-HIT (probability = 0.23)
```
Contributing factors:
- instrumentalness = 0.85   (strong negative)
- acousticness = 0.92       (negative)
- loudness = -12.5 dB       (negative - too quiet)
+ valence = 0.65            (slight positive)
- speechiness = 0.02        (negative - too low)
```

## 💡 Key Insights

### The "Hit Formula" (Based on Feature Importance)

A song is more likely to be a hit if it has:

| Feature | Optimal Range | Importance |
|---------|---------------|------------|
| Instrumentalness | < 0.1 | Critical |
| Loudness | -8 to -4 dB | High |
| Speechiness | 0.05 - 0.20 | High |
| Danceability | > 0.6 | Medium |
| Energy | > 0.6 | Medium |
| Acousticness | < 0.3 | Medium |
| Valence | 0.4 - 0.7 | Low |
| Tempo | 100 - 130 BPM | Low |

### Why These Features Matter

1. **Instrumentalness is #1:** Vocals are essential for hits - purely instrumental tracks rarely succeed in mainstream charts

2. **Loudness matters:** The "loudness war" is real - louder (compressed) tracks tend to be more attention-grabbing

3. **Speechiness sweet spot:** Too much speech (>0.3) indicates spoken word; too little may lack engagement

4. **Danceability & Energy:** Upbeat, danceable songs align with pop music consumption patterns

## 📁 Outputs

- `outputs/feature_importance/01_rf_feature_importance.png`
- `outputs/feature_importance/02_permutation_importance.png`
- `outputs/feature_importance/03_xgb_feature_importance.png`
- `outputs/feature_importance/04_shap_importance_bar.png`
- `outputs/feature_importance/04_shap_summary.png`
- `outputs/feature_importance/04_shap_dependence.png`
- `outputs/feature_importance/05_lime_explanations.png`
- `outputs/feature_importance/06_importance_comparison.png`

## 📁 Script

```bash
python models/phase04_feature_importance.py
```

**Requirements:** `pip install shap lime`

## ➡️ Next Phase

Proceed to **Phase 05: Clustering Analysis**
