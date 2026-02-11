# 📘 Phase 02 – Exploratory Data Analysis (EDA)

**Project:** Spotify Music Success Prediction

## 🎯 Objective

Perform comprehensive exploratory analysis to understand data distributions, relationships between features, and patterns that distinguish successful songs from unsuccessful ones.

## 📋 Analyses Performed

### 1. Popularity Distribution Analysis
- Distribution of `track_popularity` (0-100)
- Mean popularity: ~42
- Median popularity: ~45
- Distribution is roughly normal with slight left skew

### 2. Audio Features Distribution
Analysis of all 9 audio features:
- **Danceability:** Mean ~0.65 (right-skewed)
- **Energy:** Mean ~0.70 (left-skewed)
- **Loudness:** Mean ~-6 dB (approximately normal)
- **Speechiness:** Mean ~0.10 (heavily right-skewed)
- **Acousticness:** Mean ~0.18 (heavily right-skewed)
- **Instrumentalness:** Mean ~0.08 (extremely right-skewed)
- **Liveness:** Mean ~0.19 (right-skewed)
- **Valence:** Mean ~0.51 (approximately uniform)
- **Tempo:** Mean ~121 BPM (approximately normal)

### 3. Correlation Analysis
Key findings from correlation heatmap:
- **Energy ↔ Loudness:** Strong positive correlation (+0.76)
- **Energy ↔ Acousticness:** Strong negative correlation (-0.73)
- **Danceability ↔ Valence:** Moderate positive correlation (+0.45)
- **Popularity correlations:** Generally weak (|r| < 0.15)

### 4. Genre Analysis
| Genre | Track Count | Avg Popularity | Hit Rate |
|-------|-------------|----------------|----------|
| Pop | ~5,500 | 47.2 | 18.5% |
| Rap | ~5,000 | 44.8 | 16.2% |
| Rock | ~4,800 | 40.1 | 12.8% |
| Latin | ~4,600 | 43.5 | 15.1% |
| R&B | ~4,200 | 41.2 | 13.9% |
| EDM | ~4,000 | 38.9 | 11.2% |

### 5. Artist Analysis
- **10,692 unique artists** in dataset
- Top prolific artists: Martin Garrix, David Guetta, Calvin Harris
- High track count ≠ High average popularity
- Artist consistency varies significantly

### 6. Dimensionality Reduction (PCA)
- PC1 explains ~25% of variance
- PC2 explains ~15% of variance
- 5 components capture ~70% of variance
- Genres show moderate clustering in PCA space

### 7. t-SNE Visualization
- Clear genre clusters visible
- Pop and EDM overlap significantly
- Acoustic genres (R&B, Latin) form distinct clusters

### 8. Hit vs Non-Hit Comparison
Key differences between hits and non-hits:
| Feature | Hits (Mean) | Non-Hits (Mean) | Difference |
|---------|-------------|-----------------|------------|
| Danceability | 0.68 | 0.63 | +0.05 |
| Energy | 0.72 | 0.68 | +0.04 |
| Loudness | -5.2 dB | -6.8 dB | +1.6 dB |
| Speechiness | 0.12 | 0.09 | +0.03 |
| Valence | 0.54 | 0.50 | +0.04 |

## 📊 Visualizations Generated

1. `01_popularity_distribution.png` - Histogram, boxplot, class distribution
2. `02_audio_features_distribution.png` - 9 feature histograms
3. `03_correlation_heatmap.png` - Full correlation matrix
4. `03_popularity_correlation.png` - Features vs popularity
5. `04_genre_analysis.png` - Genre comparisons
6. `05_artist_analysis.png` - Artist statistics
7. `06_pca_visualization.png` - 2D PCA projection
8. `06_pca_explained_variance.png` - Variance analysis
9. `07_tsne_visualization.png` - t-SNE clusters
10. `08_hit_vs_nonhit_comparison.png` - Feature distributions

## 💡 Key Insights

1. **No single feature strongly predicts popularity** - multivariate approach needed
2. **Hits tend to be slightly more danceable and energetic**
3. **Genre significantly affects baseline popularity**
4. **High artist productivity doesn't guarantee popularity**
5. **Audio features cluster by genre** - genre is important context

## 📁 Script

Run the EDA script:
```bash
python models/phase02_eda.py
```

**Output directory:** `outputs/eda/`

## ➡️ Next Phase

Proceed to **Phase 03: Model Comparison**
