# 📘 Phase 07 – Genre Analysis

**Project:** Spotify Music Success Prediction

## 🎯 Objective

Analyze whether building genre-specific models improves prediction accuracy compared to a single global model, and understand which genres are easier or harder to predict.

## 📋 Analysis Approach

### 1. Global Model
- Single model trained on all genres
- Evaluate performance separately for each genre

### 2. Genre-Specific Models
- Separate model for each genre
- Train only on data from that genre

### 3. Comparison
- Compare performance metrics
- Identify where specialization helps

## 📊 Genre Overview

| Genre | Tracks | Avg Popularity | Hit Rate | Std Deviation |
|-------|--------|----------------|----------|---------------|
| Pop | 5,507 | 47.2 | 18.5% | 21.3 |
| Rap | 5,089 | 44.8 | 16.2% | 22.1 |
| Rock | 4,831 | 40.1 | 12.8% | 19.8 |
| Latin | 4,612 | 43.5 | 15.1% | 20.5 |
| R&B | 4,245 | 41.2 | 13.9% | 21.0 |
| EDM | 3,872 | 38.9 | 11.2% | 18.7 |

### Observations
- **Pop has highest popularity** and hit rate
- **EDM has lowest hit rate** despite being highly streamed genre
- **Rap has highest variance** in popularity
- **Rock has most consistent** (low std) but moderate popularity

## 📈 Global Model Performance by Genre

| Genre | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Pop | 0.79 | 0.45 | 0.52 | 0.48 | 0.76 |
| Rap | 0.77 | 0.42 | 0.48 | 0.45 | 0.74 |
| Latin | 0.78 | 0.40 | 0.50 | 0.44 | 0.75 |
| R&B | 0.80 | 0.38 | 0.45 | 0.41 | 0.73 |
| Rock | 0.82 | 0.35 | 0.40 | 0.37 | 0.71 |
| EDM | 0.84 | 0.32 | 0.38 | 0.35 | 0.70 |

## 📈 Genre-Specific Models Performance

| Genre | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Pop | 0.78 | 0.48 | 0.58 | 0.52 | 0.78 |
| Rap | 0.76 | 0.45 | 0.55 | 0.49 | 0.76 |
| Latin | 0.77 | 0.43 | 0.54 | 0.48 | 0.76 |
| R&B | 0.79 | 0.40 | 0.50 | 0.44 | 0.75 |
| Rock | 0.80 | 0.38 | 0.46 | 0.42 | 0.73 |
| EDM | 0.82 | 0.35 | 0.42 | 0.38 | 0.72 |

## 📊 Comparison: Global vs Genre-Specific

| Genre | Global F1 | Specific F1 | Improvement | Winner |
|-------|-----------|-------------|-------------|--------|
| Pop | 0.48 | 0.52 | +0.04 | ✅ Specific |
| Rap | 0.45 | 0.49 | +0.04 | ✅ Specific |
| Latin | 0.44 | 0.48 | +0.04 | ✅ Specific |
| R&B | 0.41 | 0.44 | +0.03 | ✅ Specific |
| Rock | 0.37 | 0.42 | +0.05 | ✅ Specific |
| EDM | 0.35 | 0.38 | +0.03 | ✅ Specific |

**Average Improvement:** +4% F1-Score with genre-specific models

## 🔝 Feature Importance by Genre

### Top 3 Features per Genre

| Genre | #1 Feature | #2 Feature | #3 Feature |
|-------|------------|------------|------------|
| **Pop** | instrumentalness | loudness | danceability |
| **Rap** | speechiness | loudness | duration |
| **Latin** | danceability | valence | energy |
| **R&B** | acousticness | loudness | valence |
| **Rock** | loudness | energy | instrumentalness |
| **EDM** | energy | tempo | instrumentalness |

### Insights
- **Speechiness** is uniquely important for Rap
- **Danceability** is critical for Latin music
- **Loudness** is important across all genres
- **Instrumentalness** matters most for Pop & EDM

## 📉 Genre Difficulty Analysis

Genres ranked by prediction difficulty (harder = lower F1):

| Rank | Genre | F1-Score | Difficulty |
|------|-------|----------|------------|
| 1 | EDM | 0.38 | 🔴 Hardest |
| 2 | Rock | 0.42 | 🟠 Hard |
| 3 | R&B | 0.44 | 🟡 Medium |
| 4 | Latin | 0.48 | 🟢 Easier |
| 5 | Rap | 0.49 | 🟢 Easier |
| 6 | Pop | 0.52 | 🟢 Easiest |

### Why Some Genres Are Harder to Predict

**EDM (Hardest):**
- High production variance
- Success depends heavily on DJ/artist fame
- Audio features very homogeneous within genre

**Rock (Hard):**
- Wide stylistic range (indie to metal)
- Popularity driven by nostalgia, not audio features
- Lyrics and band image matter more

**Pop (Easiest):**
- More formulaic structure
- Clear audio patterns for hits
- Features align well with popularity

## 💡 Key Insights

### 1. Genre-Specific Models Win
- Every genre benefits from specialized models
- Average 4% improvement in F1-score
- Larger dataset = better genre-specific performance

### 2. Prediction Difficulty Varies
- 17% F1 gap between easiest (Pop) and hardest (EDM)
- Some genres have clearer "hit formulas"
- Audio features explain Pop better than EDM

### 3. Feature Importance Differs by Genre
- One-size-fits-all approach suboptimal
- Genre context critical for interpretation
- Marketing strategies should be genre-specific

### 4. Practical Recommendations

| Scenario | Recommendation |
|----------|----------------|
| Large dataset | Train genre-specific models |
| Small dataset | Use global model with genre as feature |
| Real-time prediction | Global model (simpler deployment) |
| Batch analysis | Genre-specific models |

## 📁 Outputs

- `outputs/genre_analysis/01_genre_overview.png`
- `outputs/genre_analysis/04_global_vs_specific.png`
- `outputs/genre_analysis/05_feature_importance_by_genre.png`
- `outputs/genre_analysis/06_genre_difficulty.png`
- `outputs/genre_analysis/genre_comparison_results.csv`

## 📁 Script

```bash
python models/phase07_genre_analysis.py
```

## ➡️ Next Phase

Proceed to **Phase 08: Regression Analysis**
