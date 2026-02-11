# 📘 Phase 01 – Data Cleaning

**Project:** Spotify Music Success Prediction

## 🎯 Objective

The objective of this phase is to clean and preprocess the raw Spotify dataset to prepare it for exploratory analysis and machine learning modeling.

## 📋 Tasks Performed

### 1. Handling Missing Values
- **Identified:** 5 rows with missing values in `track_name`, `track_artist`, and `track_album_name`
- **Action:** Removed affected rows (< 0.02% of data)
- **Justification:** Minimal impact on dataset size

### 2. Handling Duplicates
- **Identified:** 4,477 duplicate `track_id` entries
- **Cause:** Same track appearing in multiple playlists
- **Action:** Kept first occurrence of each track
- **Result:** Unique track dataset for unbiased model training

### 3. Categorical Variable Encoding
- **Variables encoded:**
  - `playlist_genre` → `playlist_genre_encoded`
  - `playlist_subgenre` → `playlist_subgenre_encoded`
- **Method:** Label Encoding
- **Purpose:** Enable use in machine learning models

### 4. Target Variable Creation
- **Binary Classification:**
  - `is_hit = 1` if `track_popularity >= 70`
  - `is_hit = 0` otherwise
- **Multi-class Classification:**
  - `popularity_class`: low (0-30), medium (31-60), high (61-100)

### 5. Feature Engineering
New features created:
| Feature | Description | Formula |
|---------|-------------|---------|
| `duration_min` | Duration in minutes | `duration_ms / 60000` |
| `energy_dance_ratio` | Energy to danceability ratio | `energy / danceability` |
| `acoustic_electronic` | Acoustic vs electronic spectrum | `acousticness - energy` |
| `mood_score` | Combined mood indicator | `(valence + energy) / 2` |
| `speech_instrumental_ratio` | Speech vs instrumental | `speechiness / instrumentalness` |

## 📊 Output

- **Cleaned dataset:** `data/spotify_cleaned.csv`
- **Records:** ~28,000 unique tracks (after deduplication)
- **Features:** 28 columns (original + engineered)

## ✅ Validation

- No missing values in cleaned dataset
- No duplicate track IDs
- All audio features within expected ranges
- Target variable properly distributed

## 📁 Script

Run the cleaning script:
```bash
python models/phase01_data_cleaning.py
```

## ➡️ Next Phase

Proceed to **Phase 02: Exploratory Data Analysis (EDA)**
