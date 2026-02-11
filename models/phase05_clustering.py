"""
Phase 05 - Clustering Analysis
==============================
This script performs unsupervised clustering to discover song archetypes:
- K-Means clustering
- DBSCAN clustering
- Cluster profiling and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'spotify_cleaned.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'clustering/')
RANDOM_STATE = 42

AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# LOAD AND PREPARE DATA
# ============================================
def load_and_prepare_data():
    """Load data and prepare features for clustering"""
    print("=" * 60)
    print("PHASE 05 - CLUSTERING ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dataset loaded: {df.shape[0]} rows")
    
    # Select features for clustering
    features = [f for f in AUDIO_FEATURES if f in df.columns]
    X = df[features].dropna()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"📊 Features used for clustering: {len(features)}")
    
    return df, X, X_scaled, features, scaler

# ============================================
# 1. OPTIMAL K SELECTION
# ============================================
def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Find optimal number of clusters using elbow method and silhouette score"""
    print("\n" + "-" * 40)
    print("1. FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("-" * 40)
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
        
        print(f"   K={k}: Silhouette={silhouette_scores[-1]:.3f}, Inertia={inertias[-1]:.0f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Elbow plot
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette plot
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score')
    axes[1].grid(True, alpha=0.3)
    
    # Mark optimal
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    axes[1].axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[1].legend()
    
    # Calinski-Harabasz plot
    axes[2].plot(k_range, calinski_scores, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Index')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_optimal_k_selection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: 01_optimal_k_selection.png")
    print(f"🎯 Optimal K based on silhouette: {optimal_k}")
    
    return optimal_k, silhouette_scores

# ============================================
# 2. K-MEANS CLUSTERING
# ============================================
def perform_kmeans(X_scaled, df, features, optimal_k):
    """Perform K-Means clustering"""
    print("\n" + "-" * 40)
    print(f"2. K-MEANS CLUSTERING (K={optimal_k})")
    print("-" * 40)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to dataframe
    df_clustered = df.iloc[:len(clusters)].copy()
    df_clustered['cluster'] = clusters
    
    # Cluster sizes
    cluster_sizes = df_clustered['cluster'].value_counts().sort_index()
    print("\n📊 Cluster Sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"   Cluster {cluster}: {size} songs ({size/len(df_clustered)*100:.1f}%)")
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualize clusters
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                              cmap='Set2', alpha=0.6, s=10)
    
    # Plot centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('K-Means Clusters (PCA Projection)')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Cluster size bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, optimal_k))
    axes[1].bar(range(optimal_k), cluster_sizes.values, color=colors)
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Songs')
    axes[1].set_title('Cluster Size Distribution')
    axes[1].set_xticks(range(optimal_k))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}02_kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Saved: 02_kmeans_clusters.png")
    
    return df_clustered, kmeans, clusters

# ============================================
# 3. CLUSTER PROFILING
# ============================================
def profile_clusters(df_clustered, features, optimal_k):
    """Create profiles for each cluster"""
    print("\n" + "-" * 40)
    print("3. CLUSTER PROFILING")
    print("-" * 40)
    
    # Calculate mean features per cluster
    cluster_profiles = df_clustered.groupby('cluster')[features].mean()
    
    # Normalize for visualization
    cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / \
                           (cluster_profiles.max() - cluster_profiles.min())
    
    # Heatmap of cluster profiles
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_profiles_norm.T, annot=True, cmap='YlOrRd', fmt='.2f',
                linewidths=0.5, cbar_kws={'label': 'Normalized Value'})
    plt.xlabel('Cluster')
    plt.ylabel('Audio Feature')
    plt.title('Cluster Profiles (Normalized Audio Features)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Radar chart for each cluster
    fig, axes = plt.subplots(2, (optimal_k + 1) // 2, figsize=(5 * ((optimal_k + 1) // 2), 10),
                             subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i in range(optimal_k):
        values = cluster_profiles_norm.loc[i].values.tolist()
        values += values[:1]  # Complete the circle
        
        axes[i].plot(angles, values, 'o-', linewidth=2, color=plt.cm.Set2(i / optimal_k))
        axes[i].fill(angles, values, alpha=0.25, color=plt.cm.Set2(i / optimal_k))
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels(features, size=8)
        axes[i].set_title(f'Cluster {i}', size=12, fontweight='bold')
    
    # Hide unused subplots
    for j in range(optimal_k, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_cluster_radar_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 03_cluster_profiles_heatmap.png")
    print("✅ Saved: 03_cluster_radar_charts.png")
    
    # Print cluster characteristics
    print("\n📋 Cluster Characteristics:")
    for cluster in range(optimal_k):
        profile = cluster_profiles.loc[cluster]
        print(f"\n   Cluster {cluster}:")
        
        # Find distinctive features (highest values)
        top_features = profile.nlargest(3)
        for feat, val in top_features.items():
            print(f"      - High {feat}: {val:.3f}")
    
    return cluster_profiles

# ============================================
# 4. CLUSTER POPULARITY ANALYSIS
# ============================================
def analyze_cluster_popularity(df_clustered, optimal_k):
    """Analyze popularity distribution per cluster"""
    print("\n" + "-" * 40)
    print("4. CLUSTER POPULARITY ANALYSIS")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot of popularity by cluster
    colors = plt.cm.Set2(np.linspace(0, 1, optimal_k))
    bp = df_clustered.boxplot(column='track_popularity', by='cluster', ax=axes[0])
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Track Popularity')
    axes[0].set_title('Popularity Distribution by Cluster')
    plt.suptitle('')  # Remove automatic title
    
    # Mean popularity bar chart
    mean_popularity = df_clustered.groupby('cluster')['track_popularity'].mean()
    axes[1].bar(range(optimal_k), mean_popularity.values, color=colors)
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Mean Popularity')
    axes[1].set_title('Average Popularity by Cluster')
    axes[1].set_xticks(range(optimal_k))
    
    # Add value labels
    for i, v in enumerate(mean_popularity.values):
        axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_cluster_popularity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Hit rate per cluster
    if 'is_hit' in df_clustered.columns:
        hit_rate = df_clustered.groupby('cluster')['is_hit'].mean() * 100
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(optimal_k), hit_rate.values, color=colors)
        plt.xlabel('Cluster')
        plt.ylabel('Hit Rate (%)')
        plt.title('Hit Rate by Cluster')
        plt.xticks(range(optimal_k))
        
        for i, v in enumerate(hit_rate.values):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}04_cluster_hit_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved: 04_cluster_hit_rate.png")
    
    print("✅ Saved: 04_cluster_popularity.png")
    
    # Print summary
    print("\n📊 Cluster Popularity Summary:")
    summary = df_clustered.groupby('cluster').agg({
        'track_popularity': ['mean', 'std', 'median']
    }).round(2)
    summary.columns = ['Mean', 'Std', 'Median']
    print(summary)
    
    return mean_popularity

# ============================================
# 5. DBSCAN CLUSTERING (ALTERNATIVE)
# ============================================
def perform_dbscan(X_scaled, df, features):
    """Perform DBSCAN clustering as an alternative"""
    print("\n" + "-" * 40)
    print("5. DBSCAN CLUSTERING")
    print("-" * 40)
    
    # Test different eps values
    eps_values = [0.5, 1.0, 1.5, 2.0]
    results = []
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=10)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        results.append({
            'eps': eps,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_pct': n_noise / len(labels) * 100
        })
        
        print(f"   eps={eps}: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")
    
    # Use best configuration
    best_eps = 1.5  # Typically works well
    dbscan = DBSCAN(eps=best_eps, min_samples=10)
    labels = dbscan.fit_predict(X_scaled)
    
    # Visualize
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2', alpha=0.6, s=10)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(f'DBSCAN Clustering (eps={best_eps})')
    plt.colorbar(scatter, label='Cluster (-1 = noise)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_dbscan_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: 05_dbscan_clusters.png")
    
    return labels

# ============================================
# 6. NAME CLUSTERS
# ============================================
def name_clusters(cluster_profiles, optimal_k):
    """Suggest names for clusters based on their profiles"""
    print("\n" + "-" * 40)
    print("6. CLUSTER NAMING")
    print("-" * 40)
    
    cluster_names = {}
    
    for cluster in range(optimal_k):
        profile = cluster_profiles.loc[cluster]
        
        # Determine characteristics
        characteristics = []
        
        if profile['danceability'] > 0.7:
            characteristics.append('Dance')
        if profile['energy'] > 0.7:
            characteristics.append('Energetic')
        elif profile['energy'] < 0.4:
            characteristics.append('Calm')
        if profile['acousticness'] > 0.5:
            characteristics.append('Acoustic')
        if profile['instrumentalness'] > 0.3:
            characteristics.append('Instrumental')
        if profile['speechiness'] > 0.2:
            characteristics.append('Vocal')
        if profile['valence'] > 0.6:
            characteristics.append('Happy')
        elif profile['valence'] < 0.35:
            characteristics.append('Melancholic')
        if profile['tempo'] > 130:
            characteristics.append('Fast')
        elif profile['tempo'] < 100:
            characteristics.append('Slow')
        
        name = ' '.join(characteristics[:3]) if characteristics else f'Cluster {cluster}'
        cluster_names[cluster] = name
        
        print(f"   Cluster {cluster}: \"{name}\"")
    
    return cluster_names

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load and prepare data
    df, X, X_scaled, features, scaler = load_and_prepare_data()
    
    # 1. Find optimal K
    optimal_k, _ = find_optimal_k(X_scaled)
    
    # 2. K-Means clustering
    df_clustered, kmeans, clusters = perform_kmeans(X_scaled, df, features, optimal_k)
    
    # 3. Profile clusters
    cluster_profiles = profile_clusters(df_clustered, features, optimal_k)
    
    # 4. Analyze cluster popularity
    analyze_cluster_popularity(df_clustered, optimal_k)
    
    # 5. DBSCAN clustering
    perform_dbscan(X_scaled, df, features)
    
    # 6. Name clusters
    cluster_names = name_clusters(cluster_profiles, optimal_k)
    
    # Save clustered data
    df_clustered.to_csv(f'{OUTPUT_DIR}spotify_clustered.csv', index=False)
    print(f"\n💾 Clustered dataset saved to: {OUTPUT_DIR}spotify_clustered.csv")
    
    print("\n" + "=" * 60)
    print("✅ PHASE 05 - CLUSTERING COMPLETED!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return df_clustered, cluster_profiles, cluster_names

if __name__ == "__main__":
    df_clustered, profiles, names = main()
