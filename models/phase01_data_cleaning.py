"""
Phase 01 - Data Cleaning
========================
This script handles data cleaning for the Spotify dataset:
- Handling missing values
- Removing duplicates
- Encoding categorical variables
- Creating target variable for classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = '../data/spotify_songs.csv'
OUTPUT_PATH = '../data/spotify_cleaned.csv'
POPULARITY_THRESHOLD = 70  # Threshold for "hit" classification

# ============================================
# LOAD DATA
# ============================================
def load_data(path):
    """Load the Spotify dataset"""
    print("=" * 60)
    print("PHASE 01 - DATA CLEANING")
    print("=" * 60)
    
    df = pd.read_csv(path)
    print(f"\n📂 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ============================================
# HANDLE MISSING VALUES
# ============================================
def handle_missing_values(df):
    """Remove rows with missing values"""
    print("\n" + "-" * 40)
    print("1. HANDLING MISSING VALUES")
    print("-" * 40)
    
    missing_before = df.isnull().sum().sum()
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    print(f"Missing values found: {missing_before}")
    if missing_cols:
        print(f"Columns with missing values: {missing_cols}")
        for col in missing_cols:
            print(f"  - {col}: {df[col].isnull().sum()} missing")
    
    # Remove rows with missing values
    df_clean = df.dropna()
    rows_removed = len(df) - len(df_clean)
    
    print(f"\n✅ Removed {rows_removed} rows with missing values")
    print(f"Dataset size: {len(df_clean)} rows")
    
    return df_clean

# ============================================
# HANDLE DUPLICATES
# ============================================
def handle_duplicates(df, strategy='first'):
    """
    Handle duplicate track_ids
    Strategy options:
    - 'first': Keep first occurrence
    - 'last': Keep last occurrence
    - 'aggregate': Aggregate numerical values (mean)
    """
    print("\n" + "-" * 40)
    print("2. HANDLING DUPLICATES")
    print("-" * 40)
    
    duplicates = df['track_id'].duplicated().sum()
    print(f"Duplicate track_ids found: {duplicates}")
    
    if duplicates > 0:
        if strategy in ['first', 'last']:
            df_clean = df.drop_duplicates(subset='track_id', keep=strategy)
            print(f"Strategy: Keep '{strategy}' occurrence")
        elif strategy == 'aggregate':
            # Aggregate: keep categorical from first, mean for numerical
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            agg_dict = {col: 'mean' for col in numeric_cols if col != 'track_id'}
            agg_dict.update({col: 'first' for col in categorical_cols if col != 'track_id'})
            
            df_clean = df.groupby('track_id').agg(agg_dict).reset_index()
            print("Strategy: Aggregate (mean for numerical, first for categorical)")
        
        print(f"\n✅ Removed {len(df) - len(df_clean)} duplicate entries")
        print(f"Dataset size: {len(df_clean)} rows")
        return df_clean
    
    return df

# ============================================
# ENCODE CATEGORICAL VARIABLES
# ============================================
def encode_categorical(df):
    """Encode categorical variables using Label Encoding"""
    print("\n" + "-" * 40)
    print("3. ENCODING CATEGORICAL VARIABLES")
    print("-" * 40)
    
    categorical_cols = ['playlist_genre', 'playlist_subgenre']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"✅ Encoded '{col}': {len(le.classes_)} unique values")
            print(f"   Classes: {list(le.classes_)[:5]}{'...' if len(le.classes_) > 5 else ''}")
    
    return df, label_encoders

# ============================================
# CREATE TARGET VARIABLE
# ============================================
def create_target_variable(df, threshold=POPULARITY_THRESHOLD):
    """Create binary target variable for classification"""
    print("\n" + "-" * 40)
    print("4. CREATING TARGET VARIABLE")
    print("-" * 40)
    
    # Binary classification: hit (1) vs non-hit (0)
    df['is_hit'] = (df['track_popularity'] >= threshold).astype(int)
    
    hit_count = df['is_hit'].sum()
    non_hit_count = len(df) - hit_count
    hit_ratio = hit_count / len(df) * 100
    
    print(f"Popularity threshold for 'hit': >= {threshold}")
    print(f"\n📊 Class Distribution:")
    print(f"   - Hits (1): {hit_count} ({hit_ratio:.2f}%)")
    print(f"   - Non-hits (0): {non_hit_count} ({100-hit_ratio:.2f}%)")
    
    # Multi-class classification (optional)
    df['popularity_class'] = pd.cut(
        df['track_popularity'],
        bins=[0, 30, 60, 100],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    
    print(f"\n📊 Multi-class Distribution:")
    for cls in ['low', 'medium', 'high']:
        count = (df['popularity_class'] == cls).sum()
        print(f"   - {cls}: {count} ({count/len(df)*100:.2f}%)")
    
    return df

# ============================================
# FEATURE ENGINEERING
# ============================================
def feature_engineering(df):
    """Create additional features"""
    print("\n" + "-" * 40)
    print("5. FEATURE ENGINEERING")
    print("-" * 40)
    
    # Duration in minutes
    df['duration_min'] = df['duration_ms'] / 60000
    print("✅ Created 'duration_min' (duration in minutes)")
    
    # Energy-Danceability ratio
    df['energy_dance_ratio'] = df['energy'] / (df['danceability'] + 0.001)
    print("✅ Created 'energy_dance_ratio'")
    
    # Acoustic-Electronic spectrum
    df['acoustic_electronic'] = df['acousticness'] - df['energy']
    print("✅ Created 'acoustic_electronic' spectrum")
    
    # Mood score (valence + energy combination)
    df['mood_score'] = (df['valence'] + df['energy']) / 2
    print("✅ Created 'mood_score'")
    
    # Speech vs Instrumental
    df['speech_instrumental_ratio'] = df['speechiness'] / (df['instrumentalness'] + 0.001)
    print("✅ Created 'speech_instrumental_ratio'")
    
    return df

# ============================================
# DATA SUMMARY
# ============================================
def print_summary(df):
    """Print final dataset summary"""
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    print(f"\n📋 Columns:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"   {i:2d}. {col} ({dtype})")
    
    print(f"\n📈 Audio Features Statistics:")
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    for feature in audio_features:
        if feature in df.columns:
            print(f"   - {feature}: mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load data
    df = load_data(DATA_PATH)
    
    # Step 1: Handle missing values
    df = handle_missing_values(df)
    
    # Step 2: Handle duplicates
    df = handle_duplicates(df, strategy='first')
    
    # Step 3: Encode categorical variables
    df, encoders = encode_categorical(df)
    
    # Step 4: Create target variable
    df = create_target_variable(df)
    
    # Step 5: Feature engineering
    df = feature_engineering(df)
    
    # Print summary
    print_summary(df)
    
    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Cleaned dataset saved to: {OUTPUT_PATH}")
    print("\n✅ PHASE 01 - DATA CLEANING COMPLETED!")
    
    return df

if __name__ == "__main__":
    df_cleaned = main()
