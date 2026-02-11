"""
Phase 07 - Genre Analysis
=========================
This script analyzes model performance by genre:
- Global model vs genre-specific models
- Feature importance by genre
- Genre difficulty analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'spotify_cleaned.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'genre_analysis/')
RANDOM_STATE = 42

AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 
                  'tempo', 'duration_ms']

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# LOAD AND PREPARE DATA
# ============================================
def load_and_prepare_data():
    """Load data and prepare features"""
    print("=" * 60)
    print("PHASE 07 - GENRE ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dataset loaded: {df.shape[0]} rows")
    
    # Select features
    features = [f for f in AUDIO_FEATURES if f in df.columns]
    engineered = ['energy_dance_ratio', 'mood_score', 'acoustic_electronic']
    features.extend([f for f in engineered if f in df.columns])
    
    print(f"📊 Genres available: {df['playlist_genre'].nunique()}")
    print(f"   {df['playlist_genre'].unique()}")
    
    return df, features

# ============================================
# 1. GENRE OVERVIEW
# ============================================
def genre_overview(df):
    """Overview of dataset by genre"""
    print("\n" + "-" * 40)
    print("1. GENRE OVERVIEW")
    print("-" * 40)
    
    genre_stats = df.groupby('playlist_genre').agg({
        'track_id': 'count',
        'track_popularity': ['mean', 'std'],
        'is_hit': 'mean'
    }).round(2)
    genre_stats.columns = ['Count', 'Avg Popularity', 'Std Popularity', 'Hit Rate']
    genre_stats['Hit Rate'] = (genre_stats['Hit Rate'] * 100).round(1)
    genre_stats = genre_stats.sort_values('Count', ascending=False)
    
    print("\n📊 Genre Statistics:")
    print(genre_stats)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    genres = genre_stats.index.tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))
    
    # Count by genre
    axes[0, 0].bar(genres, genre_stats['Count'], color=colors)
    axes[0, 0].set_xlabel('Genre')
    axes[0, 0].set_ylabel('Number of Tracks')
    axes[0, 0].set_title('Tracks per Genre')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Average popularity
    axes[0, 1].bar(genres, genre_stats['Avg Popularity'], color=colors)
    axes[0, 1].set_xlabel('Genre')
    axes[0, 1].set_ylabel('Average Popularity')
    axes[0, 1].set_title('Average Popularity by Genre')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Hit rate
    axes[1, 0].bar(genres, genre_stats['Hit Rate'], color=colors)
    axes[1, 0].set_xlabel('Genre')
    axes[1, 0].set_ylabel('Hit Rate (%)')
    axes[1, 0].set_title('Hit Rate by Genre')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Popularity distribution by genre
    df.boxplot(column='track_popularity', by='playlist_genre', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Genre')
    axes[1, 1].set_ylabel('Track Popularity')
    axes[1, 1].set_title('Popularity Distribution by Genre')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_genre_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Saved: 01_genre_overview.png")
    
    return genre_stats

# ============================================
# 2. GLOBAL MODEL PERFORMANCE BY GENRE
# ============================================
def global_model_by_genre(df, features):
    """Train global model and evaluate performance by genre"""
    print("\n" + "-" * 40)
    print("2. GLOBAL MODEL PERFORMANCE BY GENRE")
    print("-" * 40)
    
    # Prepare data
    X = df[features].dropna()
    y = df.loc[X.index, 'is_hit']
    genres = df.loc[X.index, 'playlist_genre']
    
    # Split data
    X_train, X_test, y_train, y_test, genres_train, genres_test = train_test_split(
        X, y, genres, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train global model
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, 
                                   class_weight='balanced', n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Overall performance
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n📊 Global Model Overall Performance:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Performance by genre
    results = []
    for genre in genres_test.unique():
        mask = genres_test == genre
        if mask.sum() > 10:  # Minimum samples
            genre_pred = y_pred[mask]
            genre_true = y_test[mask]
            genre_proba = y_proba[mask]
            
            results.append({
                'Genre': genre,
                'Samples': mask.sum(),
                'Accuracy': accuracy_score(genre_true, genre_pred),
                'Precision': precision_score(genre_true, genre_pred, zero_division=0),
                'Recall': recall_score(genre_true, genre_pred, zero_division=0),
                'F1-Score': f1_score(genre_true, genre_pred, zero_division=0),
                'AUC-ROC': roc_auc_score(genre_true, genre_proba) if len(np.unique(genre_true)) > 1 else 0
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n📊 Performance by Genre:")
    print(results_df.to_string(index=False))
    
    return results_df, model, scaler

# ============================================
# 3. GENRE-SPECIFIC MODELS
# ============================================
def genre_specific_models(df, features):
    """Train separate models for each genre"""
    print("\n" + "-" * 40)
    print("3. GENRE-SPECIFIC MODELS")
    print("-" * 40)
    
    results = []
    genre_models = {}
    
    for genre in df['playlist_genre'].unique():
        df_genre = df[df['playlist_genre'] == genre]
        
        if len(df_genre) < 100:
            print(f"   Skipping {genre}: insufficient samples ({len(df_genre)})")
            continue
        
        print(f"\n🔄 Training model for {genre}...")
        
        # Prepare data
        X = df_genre[features].dropna()
        y = df_genre.loc[X.index, 'is_hit']
        
        if len(y.unique()) < 2:
            print(f"   Skipping {genre}: only one class present")
            continue
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, 
            stratify=y if y.value_counts().min() >= 2 else None
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                       class_weight='balanced', n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results.append({
            'Genre': genre,
            'Train Size': len(y_train),
            'Test Size': len(y_test),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
        })
        
        genre_models[genre] = {'model': model, 'scaler': scaler}
        
        print(f"   ✅ F1: {results[-1]['F1-Score']:.4f}, AUC: {results[-1]['AUC-ROC']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    return results_df, genre_models

# ============================================
# 4. COMPARE GLOBAL VS GENRE-SPECIFIC
# ============================================
def compare_approaches(global_results, specific_results):
    """Compare global model vs genre-specific models"""
    print("\n" + "-" * 40)
    print("4. GLOBAL VS GENRE-SPECIFIC COMPARISON")
    print("-" * 40)
    
    # Merge results
    comparison = global_results[['Genre', 'F1-Score', 'AUC-ROC']].copy()
    comparison.columns = ['Genre', 'Global F1', 'Global AUC']
    
    specific_subset = specific_results[['Genre', 'F1-Score', 'AUC-ROC']].copy()
    specific_subset.columns = ['Genre', 'Specific F1', 'Specific AUC']
    
    comparison = comparison.merge(specific_subset, on='Genre')
    comparison['F1 Improvement'] = comparison['Specific F1'] - comparison['Global F1']
    comparison['AUC Improvement'] = comparison['Specific AUC'] - comparison['Global AUC']
    
    print("\n📊 Comparison Table:")
    print(comparison.to_string(index=False))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(comparison))
    width = 0.35
    
    # F1 comparison
    axes[0].bar(x - width/2, comparison['Global F1'], width, label='Global Model', color='#667eea')
    axes[0].bar(x + width/2, comparison['Specific F1'], width, label='Genre-Specific', color='#f093fb')
    axes[0].set_xlabel('Genre')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_title('F1-Score: Global vs Genre-Specific Models')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison['Genre'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # AUC comparison
    axes[1].bar(x - width/2, comparison['Global AUC'], width, label='Global Model', color='#667eea')
    axes[1].bar(x + width/2, comparison['Specific AUC'], width, label='Genre-Specific', color='#f093fb')
    axes[1].set_xlabel('Genre')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('AUC-ROC: Global vs Genre-Specific Models')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison['Genre'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_global_vs_specific.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Saved: 04_global_vs_specific.png")
    
    # Summary
    avg_improvement_f1 = comparison['F1 Improvement'].mean()
    avg_improvement_auc = comparison['AUC Improvement'].mean()
    
    print(f"\n📈 Average Improvement with Genre-Specific Models:")
    print(f"   F1-Score: {avg_improvement_f1:+.4f}")
    print(f"   AUC-ROC: {avg_improvement_auc:+.4f}")
    
    return comparison

# ============================================
# 5. FEATURE IMPORTANCE BY GENRE
# ============================================
def feature_importance_by_genre(df, features, genre_models):
    """Compare feature importance across genres"""
    print("\n" + "-" * 40)
    print("5. FEATURE IMPORTANCE BY GENRE")
    print("-" * 40)
    
    importance_data = []
    
    for genre, model_info in genre_models.items():
        model = model_info['model']
        for feature, importance in zip(features, model.feature_importances_):
            importance_data.append({
                'Genre': genre,
                'Feature': feature,
                'Importance': importance
            })
    
    importance_df = pd.DataFrame(importance_data)
    
    # Pivot for heatmap
    pivot_df = importance_df.pivot(index='Feature', columns='Genre', values='Importance')
    
    # Visualize
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f', linewidths=0.5)
    plt.title('Feature Importance by Genre')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_feature_importance_by_genre.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 05_feature_importance_by_genre.png")
    
    # Top feature per genre
    print("\n🔝 Most Important Feature per Genre:")
    for genre in genre_models.keys():
        genre_imp = importance_df[importance_df['Genre'] == genre]
        top_feature = genre_imp.loc[genre_imp['Importance'].idxmax()]
        print(f"   {genre}: {top_feature['Feature']} ({top_feature['Importance']:.3f})")
    
    return importance_df

# ============================================
# 6. GENRE DIFFICULTY ANALYSIS
# ============================================
def genre_difficulty_analysis(specific_results):
    """Analyze which genres are harder to predict"""
    print("\n" + "-" * 40)
    print("6. GENRE DIFFICULTY ANALYSIS")
    print("-" * 40)
    
    # Sort by F1-Score (lower = harder)
    difficulty = specific_results.sort_values('F1-Score')
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(difficulty)))[::-1]
    
    plt.barh(difficulty['Genre'], difficulty['F1-Score'], color=colors)
    plt.xlabel('F1-Score')
    plt.ylabel('Genre')
    plt.title('Prediction Difficulty by Genre (Lower F1 = Harder)')
    plt.axvline(difficulty['F1-Score'].mean(), color='black', linestyle='--', 
                label=f"Mean: {difficulty['F1-Score'].mean():.3f}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06_genre_difficulty.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 06_genre_difficulty.png")
    
    # Analysis
    hardest = difficulty.iloc[0]
    easiest = difficulty.iloc[-1]
    
    print(f"\n🎯 Easiest to predict: {easiest['Genre']} (F1: {easiest['F1-Score']:.4f})")
    print(f"🎯 Hardest to predict: {hardest['Genre']} (F1: {hardest['F1-Score']:.4f})")
    
    print("\n💡 Possible reasons for difficulty:")
    print("   - High variance in popularity within genre")
    print("   - Genre has less distinctive audio features")
    print("   - Smaller sample size leading to overfitting")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load data
    df, features = load_and_prepare_data()
    
    # 1. Genre overview
    genre_stats = genre_overview(df)
    
    # 2. Global model by genre
    global_results, global_model, scaler = global_model_by_genre(df, features)
    
    # 3. Genre-specific models
    specific_results, genre_models = genre_specific_models(df, features)
    
    # 4. Compare approaches
    comparison = compare_approaches(global_results, specific_results)
    
    # 5. Feature importance by genre
    if genre_models:
        importance_df = feature_importance_by_genre(df, features, genre_models)
    
    # 6. Genre difficulty
    genre_difficulty_analysis(specific_results)
    
    # Save results
    comparison.to_csv(f'{OUTPUT_DIR}genre_comparison_results.csv', index=False)
    print(f"\n💾 Results saved to: {OUTPUT_DIR}genre_comparison_results.csv")
    
    print("\n" + "=" * 60)
    print("✅ PHASE 07 - GENRE ANALYSIS COMPLETED!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return comparison, genre_models

if __name__ == "__main__":
    comparison, models = main()
