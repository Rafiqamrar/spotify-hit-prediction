"""
Phase 02 - Exploratory Data Analysis (EDA)
==========================================
This script performs comprehensive EDA on the Spotify dataset:
- Distribution analysis
- Correlation analysis
- Genre-based analysis
- Visualization with PCA/t-SNE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = '../data/spotify_cleaned.csv'
OUTPUT_DIR = '../outputs/eda/'
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================
# LOAD DATA
# ============================================
def load_data():
    """Load the cleaned dataset"""
    print("=" * 60)
    print("PHASE 02 - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ============================================
# 1. POPULARITY DISTRIBUTION
# ============================================
def analyze_popularity_distribution(df):
    """Analyze the distribution of track popularity"""
    print("\n" + "-" * 40)
    print("1. POPULARITY DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    axes[0].hist(df['track_popularity'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['track_popularity'].mean(), color='red', linestyle='--', label=f"Mean: {df['track_popularity'].mean():.1f}")
    axes[0].axvline(df['track_popularity'].median(), color='green', linestyle='--', label=f"Median: {df['track_popularity'].median():.1f}")
    axes[0].set_xlabel('Track Popularity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Track Popularity')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(df['track_popularity'])
    axes[1].set_ylabel('Track Popularity')
    axes[1].set_title('Box Plot of Track Popularity')
    
    # Class distribution (if exists)
    if 'is_hit' in df.columns:
        hit_counts = df['is_hit'].value_counts()
        axes[2].bar(['Non-Hit', 'Hit'], [hit_counts[0], hit_counts[1]], color=['#ff6b6b', '#4ecdc4'])
        axes[2].set_ylabel('Count')
        axes[2].set_title('Hit vs Non-Hit Distribution')
        for i, v in enumerate([hit_counts[0], hit_counts[1]]):
            axes[2].text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_popularity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: 01_popularity_distribution.png")
    print(f"   Mean popularity: {df['track_popularity'].mean():.2f}")
    print(f"   Median popularity: {df['track_popularity'].median():.2f}")
    print(f"   Std deviation: {df['track_popularity'].std():.2f}")

# ============================================
# 2. AUDIO FEATURES DISTRIBUTION
# ============================================
def analyze_audio_features_distribution(df):
    """Analyze distribution of all audio features"""
    print("\n" + "-" * 40)
    print("2. AUDIO FEATURES DISTRIBUTION")
    print("-" * 40)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(AUDIO_FEATURES):
        if feature in df.columns:
            axes[i].hist(df[feature], bins=50, edgecolor='black', alpha=0.7, color=plt.cm.Set3(i))
            axes[i].axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2)
            axes[i].set_xlabel(feature.capitalize())
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {feature.capitalize()}')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}02_audio_features_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: 02_audio_features_distribution.png")

# ============================================
# 3. CORRELATION HEATMAP
# ============================================
def create_correlation_heatmap(df):
    """Create correlation heatmap for audio features"""
    print("\n" + "-" * 40)
    print("3. CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Select numeric columns
    numeric_cols = AUDIO_FEATURES + ['track_popularity', 'duration_ms']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    corr_matrix = df[numeric_cols].corr()
    
    # Full heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Audio Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation with popularity
    pop_corr = corr_matrix['track_popularity'].drop('track_popularity').sort_values()
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in pop_corr.values]
    pop_corr.plot(kind='barh', color=colors)
    plt.xlabel('Correlation with Track Popularity')
    plt.title('Feature Correlation with Track Popularity')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_popularity_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: 03_correlation_heatmap.png")
    print(f"✅ Saved: 03_popularity_correlation.png")
    print(f"\n📊 Top correlations with popularity:")
    for feature, corr in pop_corr.tail(5).items():
        print(f"   {feature}: {corr:.3f}")

# ============================================
# 4. GENRE ANALYSIS
# ============================================
def analyze_genres(df):
    """Analyze popularity and features by genre"""
    print("\n" + "-" * 40)
    print("4. GENRE ANALYSIS")
    print("-" * 40)
    
    if 'playlist_genre' not in df.columns:
        print("⚠️ Genre column not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Popularity by genre (boxplot)
    genre_order = df.groupby('playlist_genre')['track_popularity'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='playlist_genre', y='track_popularity', order=genre_order, ax=axes[0, 0])
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    axes[0, 0].set_title('Track Popularity by Genre')
    axes[0, 0].set_xlabel('Genre')
    axes[0, 0].set_ylabel('Popularity')
    
    # Genre count
    genre_counts = df['playlist_genre'].value_counts()
    axes[0, 1].bar(genre_counts.index, genre_counts.values, color=plt.cm.Set3(range(len(genre_counts))))
    axes[0, 1].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    axes[0, 1].set_title('Number of Tracks by Genre')
    axes[0, 1].set_xlabel('Genre')
    axes[0, 1].set_ylabel('Count')
    
    # Average audio features by genre (heatmap)
    genre_features = df.groupby('playlist_genre')[AUDIO_FEATURES].mean()
    # Normalize for visualization
    genre_features_norm = (genre_features - genre_features.min()) / (genre_features.max() - genre_features.min())
    
    sns.heatmap(genre_features_norm.T, annot=False, cmap='YlOrRd', ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Audio Features by Genre')
    axes[1, 0].set_xlabel('Genre')
    axes[1, 0].set_ylabel('Feature')
    
    # Hit rate by genre
    if 'is_hit' in df.columns:
        hit_rate = df.groupby('playlist_genre')['is_hit'].mean() * 100
        hit_rate = hit_rate.sort_values(ascending=False)
        axes[1, 1].bar(hit_rate.index, hit_rate.values, color='#4ecdc4')
        axes[1, 1].set_xticklabels(hit_rate.index, rotation=45, ha='right')
        axes[1, 1].set_title('Hit Rate by Genre (%)')
        axes[1, 1].set_xlabel('Genre')
        axes[1, 1].set_ylabel('Hit Rate (%)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_genre_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: 04_genre_analysis.png")
    print(f"\n📊 Genres in dataset: {df['playlist_genre'].nunique()}")
    for genre in df['playlist_genre'].unique():
        count = len(df[df['playlist_genre'] == genre])
        avg_pop = df[df['playlist_genre'] == genre]['track_popularity'].mean()
        print(f"   - {genre}: {count} tracks, avg popularity: {avg_pop:.1f}")

# ============================================
# 5. ARTIST ANALYSIS
# ============================================
def analyze_artists(df):
    """Analyze artists: prolific vs popular"""
    print("\n" + "-" * 40)
    print("5. ARTIST ANALYSIS")
    print("-" * 40)
    
    if 'track_artist' not in df.columns:
        print("⚠️ Artist column not found")
        return
    
    # Artist statistics
    artist_stats = df.groupby('track_artist').agg({
        'track_popularity': ['mean', 'std', 'count'],
        'track_id': 'count'
    }).round(2)
    artist_stats.columns = ['avg_popularity', 'std_popularity', 'track_count', 'appearances']
    artist_stats = artist_stats.sort_values('track_count', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top 15 most prolific artists
    top_prolific = artist_stats.head(15)
    axes[0].barh(range(len(top_prolific)), top_prolific['track_count'].values, color='#667eea')
    axes[0].set_yticks(range(len(top_prolific)))
    axes[0].set_yticklabels(top_prolific.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Number of Tracks')
    axes[0].set_title('Top 15 Most Prolific Artists')
    
    # Top 15 highest average popularity (min 5 tracks)
    popular_artists = artist_stats[artist_stats['track_count'] >= 5].sort_values('avg_popularity', ascending=False).head(15)
    axes[1].barh(range(len(popular_artists)), popular_artists['avg_popularity'].values, color='#f093fb')
    axes[1].set_yticks(range(len(popular_artists)))
    axes[1].set_yticklabels(popular_artists.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Average Popularity')
    axes[1].set_title('Top 15 Most Popular Artists (min 5 tracks)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_artist_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: 05_artist_analysis.png")
    print(f"   Total unique artists: {df['track_artist'].nunique()}")

# ============================================
# 6. PCA VISUALIZATION
# ============================================
def perform_pca_visualization(df):
    """Perform PCA and visualize in 2D"""
    print("\n" + "-" * 40)
    print("6. PCA VISUALIZATION")
    print("-" * 40)
    
    # Prepare features
    features = [f for f in AUDIO_FEATURES if f in df.columns]
    X = df[features].dropna()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color by popularity
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=df.loc[X.index, 'track_popularity'], 
                               cmap='viridis', alpha=0.5, s=10)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('PCA - Colored by Popularity')
    plt.colorbar(scatter1, ax=axes[0], label='Popularity')
    
    # Color by genre
    if 'playlist_genre' in df.columns:
        genres = df.loc[X.index, 'playlist_genre']
        genre_codes = pd.Categorical(genres).codes
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=genre_codes, cmap='Set2', alpha=0.5, s=10)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1].set_title('PCA - Colored by Genre')
        
        # Legend
        unique_genres = genres.unique()
        handles = [plt.scatter([], [], c=plt.cm.Set2(i/len(unique_genres)), label=g, s=50) 
                   for i, g in enumerate(unique_genres)]
        axes[1].legend(handles=handles, loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06_pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Explained variance
    pca_full = PCA().fit(X_scaled)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
            pca_full.explained_variance_ratio_, alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
             np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06_pca_explained_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: 06_pca_visualization.png")
    print(f"✅ Saved: 06_pca_explained_variance.png")
    print(f"   PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}% variance")
    print(f"   PC2 explains: {pca.explained_variance_ratio_[1]*100:.1f}% variance")

# ============================================
# 7. t-SNE VISUALIZATION
# ============================================
def perform_tsne_visualization(df, sample_size=5000):
    """Perform t-SNE visualization (on sample for speed)"""
    print("\n" + "-" * 40)
    print("7. t-SNE VISUALIZATION")
    print("-" * 40)
    
    # Sample for speed
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"   Using sample of {sample_size} tracks for t-SNE")
    else:
        df_sample = df
    
    # Prepare features
    features = [f for f in AUDIO_FEATURES if f in df_sample.columns]
    X = df_sample[features].dropna()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE
    print("   Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color by popularity
    scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                               c=df_sample.loc[X.index, 'track_popularity'], 
                               cmap='viridis', alpha=0.5, s=10)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].set_title('t-SNE - Colored by Popularity')
    plt.colorbar(scatter1, ax=axes[0], label='Popularity')
    
    # Color by genre
    if 'playlist_genre' in df_sample.columns:
        genres = df_sample.loc[X.index, 'playlist_genre']
        genre_codes = pd.Categorical(genres).codes
        scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                   c=genre_codes, cmap='Set2', alpha=0.5, s=10)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('t-SNE - Colored by Genre')
        
        unique_genres = genres.unique()
        handles = [plt.scatter([], [], c=plt.cm.Set2(i/len(unique_genres)), label=g, s=50) 
                   for i, g in enumerate(unique_genres)]
        axes[1].legend(handles=handles, loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}07_tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: 07_tsne_visualization.png")

# ============================================
# 8. HIT VS NON-HIT COMPARISON
# ============================================
def compare_hit_vs_nonhit(df):
    """Compare audio features between hits and non-hits"""
    print("\n" + "-" * 40)
    print("8. HIT VS NON-HIT COMPARISON")
    print("-" * 40)
    
    if 'is_hit' not in df.columns:
        print("⚠️ is_hit column not found")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(AUDIO_FEATURES):
        if feature in df.columns:
            hits = df[df['is_hit'] == 1][feature]
            non_hits = df[df['is_hit'] == 0][feature]
            
            axes[i].hist(non_hits, bins=30, alpha=0.5, label='Non-Hit', color='#ff6b6b', density=True)
            axes[i].hist(hits, bins=30, alpha=0.5, label='Hit', color='#4ecdc4', density=True)
            axes[i].axvline(hits.mean(), color='#4ecdc4', linestyle='--', linewidth=2)
            axes[i].axvline(non_hits.mean(), color='#ff6b6b', linestyle='--', linewidth=2)
            axes[i].set_xlabel(feature.capitalize())
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{feature.capitalize()} Distribution')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}08_hit_vs_nonhit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical comparison
    print(f"\n📊 Mean Feature Values:")
    print(f"{'Feature':<20} {'Hits':>10} {'Non-Hits':>10} {'Difference':>12}")
    print("-" * 55)
    for feature in AUDIO_FEATURES:
        if feature in df.columns:
            hit_mean = df[df['is_hit'] == 1][feature].mean()
            nonhit_mean = df[df['is_hit'] == 0][feature].mean()
            diff = hit_mean - nonhit_mean
            print(f"{feature:<20} {hit_mean:>10.3f} {nonhit_mean:>10.3f} {diff:>+12.3f}")
    
    print(f"\n✅ Saved: 08_hit_vs_nonhit_comparison.png")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load data
    df = load_data()
    
    # Run all analyses
    analyze_popularity_distribution(df)
    analyze_audio_features_distribution(df)
    create_correlation_heatmap(df)
    analyze_genres(df)
    analyze_artists(df)
    perform_pca_visualization(df)
    perform_tsne_visualization(df)
    compare_hit_vs_nonhit(df)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 02 - EDA COMPLETED!")
    print(f"   All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
