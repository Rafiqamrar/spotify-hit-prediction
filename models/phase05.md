# 📘 Phase 05 – Clustering Analysis

**Project:** Spotify Music Success Prediction

## 🎯 Objective

Discover natural groupings (archetypes) of songs using unsupervised learning to understand the different types of music in the dataset and their relationship to popularity.

## 📋 Methods Used

### 1. K-Means Clustering
- Partitions data into K distinct clusters
- Minimizes within-cluster variance

### 2. DBSCAN
- Density-based clustering
- Automatically detects number of clusters
- Can identify outliers

### 3. Cluster Evaluation Metrics
- **Silhouette Score:** Measures cluster cohesion and separation
- **Elbow Method:** Identifies optimal K via inertia
- **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster variance

## 📊 Results

### Optimal Number of Clusters

| K | Silhouette Score | Inertia | Calinski-Harabasz |
|---|------------------|---------|-------------------|
| 2 | 0.21 | 45,230 | 8,450 |
| 3 | 0.23 | 38,120 | 7,890 |
| 4 | 0.25 | 32,450 | 7,230 |
| **5** | **0.26** | 28,100 | 6,980 |
| 6 | 0.24 | 24,890 | 6,540 |
| 7 | 0.22 | 22,340 | 6,120 |

**Optimal K = 5** (based on silhouette score)

### Cluster Profiles

#### Cluster 0: "Dance Floor Bangers" 🕺
| Feature | Value | Description |
|---------|-------|-------------|
| Danceability | 0.78 | High |
| Energy | 0.82 | High |
| Valence | 0.68 | Happy |
| Tempo | 125 BPM | Fast |
| Acousticness | 0.08 | Electronic |

**Songs:** 4,850 (17.2%)
**Avg Popularity:** 48.5
**Hit Rate:** 22.3%

---

#### Cluster 1: "Acoustic Ballads" 🎸
| Feature | Value | Description |
|---------|-------|-------------|
| Danceability | 0.45 | Low |
| Energy | 0.35 | Low |
| Acousticness | 0.72 | High |
| Instrumentalness | 0.12 | Some |
| Valence | 0.38 | Melancholic |

**Songs:** 5,120 (18.2%)
**Avg Popularity:** 38.2
**Hit Rate:** 8.5%

---

#### Cluster 2: "Hip-Hop/Rap Tracks" 🎤
| Feature | Value | Description |
|---------|-------|-------------|
| Speechiness | 0.28 | High |
| Danceability | 0.72 | High |
| Energy | 0.68 | Medium-High |
| Valence | 0.52 | Neutral |
| Loudness | -5.8 dB | Loud |

**Songs:** 5,890 (20.9%)
**Avg Popularity:** 45.8
**Hit Rate:** 18.7%

---

#### Cluster 3: "Mainstream Pop" ⭐
| Feature | Value | Description |
|---------|-------|-------------|
| Danceability | 0.68 | Good |
| Energy | 0.72 | Good |
| Loudness | -5.2 dB | Loud |
| Valence | 0.58 | Positive |
| Instrumentalness | 0.002 | Vocal |

**Songs:** 6,450 (22.9%)
**Avg Popularity:** 52.3
**Hit Rate:** 24.8%

---

#### Cluster 4: "Chill/Ambient" 🌙
| Feature | Value | Description |
|---------|-------|-------------|
| Energy | 0.42 | Low |
| Instrumentalness | 0.45 | High |
| Acousticness | 0.48 | Medium |
| Tempo | 98 BPM | Slow |
| Loudness | -10.2 dB | Quiet |

**Songs:** 5,850 (20.8%)
**Avg Popularity:** 35.6
**Hit Rate:** 6.2%

### Cluster Popularity Analysis

| Cluster | Name | Avg Popularity | Hit Rate | Rank |
|---------|------|----------------|----------|------|
| 3 | Mainstream Pop | 52.3 | 24.8% | 🥇 |
| 0 | Dance Floor Bangers | 48.5 | 22.3% | 🥈 |
| 2 | Hip-Hop/Rap | 45.8 | 18.7% | 🥉 |
| 1 | Acoustic Ballads | 38.2 | 8.5% | 4th |
| 4 | Chill/Ambient | 35.6 | 6.2% | 5th |

### DBSCAN Results

With eps=1.5, min_samples=10:
- **Clusters found:** 8
- **Noise points:** 2,340 (8.3%)
- More granular clusters than K-Means
- Identified outlier tracks (experimental music)

## 💡 Key Insights

### 1. Cluster-Popularity Relationship
- **Mainstream Pop cluster has highest hit rate** (24.8%)
- **Chill/Ambient has lowest hit rate** (6.2%)
- 4x difference between best and worst performing clusters

### 2. The "Perfect Song" Profile
Based on highest-performing cluster (Mainstream Pop):
```
danceability: 0.65-0.72
energy: 0.68-0.75
loudness: -6 to -4 dB
valence: 0.55-0.65
instrumentalness: < 0.01
speechiness: 0.03-0.08
tempo: 115-128 BPM
```

### 3. Genre-Cluster Alignment
- Pop songs predominantly in Clusters 0 & 3
- R&B/Hip-Hop in Cluster 2
- Rock/Alternative in Clusters 1 & 4
- EDM split between Clusters 0 & 4

### 4. Strategic Implications

| If you want... | Target Cluster | Audio Profile |
|----------------|----------------|---------------|
| Maximum reach | Mainstream Pop | Balanced, vocal, loud |
| Dance audience | Dance Floor | High energy, fast |
| Niche appeal | Chill/Ambient | Calm, instrumental |

## 📁 Outputs

- `outputs/clustering/01_optimal_k_selection.png`
- `outputs/clustering/02_kmeans_clusters.png`
- `outputs/clustering/03_cluster_profiles_heatmap.png`
- `outputs/clustering/03_cluster_radar_charts.png`
- `outputs/clustering/04_cluster_popularity.png`
- `outputs/clustering/04_cluster_hit_rate.png`
- `outputs/clustering/05_dbscan_clusters.png`
- `outputs/clustering/spotify_clustered.csv`

## 📁 Script

```bash
python models/phase05_clustering.py
```

## ➡️ Next Phase

Proceed to **Phase 06: Imbalanced Learning**
