"""
Phase 06 - Imbalanced Learning
==============================
This script handles class imbalance in the Spotify dataset:
- SMOTE oversampling
- Class weight adjustment
- Comparison of balanced vs unbalanced models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Try to import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ imbalanced-learn not installed. Install with: pip install imbalanced-learn")

# ============================================
# CONFIGURATION
# ============================================
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'spotify_cleaned.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'imbalanced/')
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
    """Load data and analyze class distribution"""
    print("=" * 60)
    print("PHASE 06 - IMBALANCED LEARNING")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dataset loaded: {df.shape[0]} rows")
    
    # Select features
    features = [f for f in AUDIO_FEATURES if f in df.columns]
    engineered = ['energy_dance_ratio', 'mood_score', 'acoustic_electronic']
    features.extend([f for f in engineered if f in df.columns])
    
    X = df[features].dropna()
    y = df.loc[X.index, 'is_hit']
    
    return X, y, features

# ============================================
# 1. ANALYZE CLASS IMBALANCE
# ============================================
def analyze_imbalance(y):
    """Analyze and visualize class imbalance"""
    print("\n" + "-" * 40)
    print("1. CLASS IMBALANCE ANALYSIS")
    print("-" * 40)
    
    class_counts = y.value_counts()
    class_ratio = class_counts[0] / class_counts[1]
    
    print(f"\n📊 Class Distribution:")
    print(f"   - Non-Hits (0): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"   - Hits (1): {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")
    print(f"   - Imbalance Ratio: {class_ratio:.2f}:1")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart
    colors = ['#ff6b6b', '#4ecdc4']
    bars = axes[0].bar(['Non-Hit', 'Hit'], class_counts.values, color=colors)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution')
    for bar, val in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                    str(val), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Non-Hit', 'Hit'], autopct='%1.1f%%',
                colors=colors, explode=[0, 0.05])
    axes[1].set_title('Class Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Saved: 01_class_distribution.png")
    
    return class_counts, class_ratio

# ============================================
# 2. BASELINE MODEL (NO BALANCING)
# ============================================
def train_baseline(X, y):
    """Train baseline model without any balancing"""
    print("\n" + "-" * 40)
    print("2. BASELINE MODEL (NO BALANCING)")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n📊 Baseline Results:")
    for metric, value in metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")
    
    return metrics, (X_train_scaled, X_test_scaled, y_train, y_test), scaler

# ============================================
# 3. SMOTE OVERSAMPLING
# ============================================
def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling"""
    if not IMBLEARN_AVAILABLE:
        print("\n⚠️ Skipping SMOTE (imbalanced-learn not installed)")
        return X_train, y_train
    
    print("\n" + "-" * 40)
    print("3. SMOTE OVERSAMPLING")
    print("-" * 40)
    
    print(f"   Before SMOTE: {len(y_train)} samples")
    print(f"   Class distribution: {dict(pd.Series(y_train).value_counts())}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n   After SMOTE: {len(y_resampled)} samples")
    print(f"   Class distribution: {dict(pd.Series(y_resampled).value_counts())}")
    
    return X_resampled, y_resampled

# ============================================
# 4. COMPARE BALANCING STRATEGIES
# ============================================
def compare_strategies(X, y):
    """Compare different balancing strategies"""
    print("\n" + "-" * 40)
    print("4. COMPARING BALANCING STRATEGIES")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    strategies = {
        'No Balancing': {'X_train': X_train_scaled, 'y_train': y_train, 'class_weight': None},
        'Class Weight': {'X_train': X_train_scaled, 'y_train': y_train, 'class_weight': 'balanced'},
    }
    
    if IMBLEARN_AVAILABLE:
        # SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
        strategies['SMOTE'] = {'X_train': X_smote, 'y_train': y_smote, 'class_weight': None}
        
        # ADASYN
        try:
            adasyn = ADASYN(random_state=RANDOM_STATE)
            X_adasyn, y_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
            strategies['ADASYN'] = {'X_train': X_adasyn, 'y_train': y_adasyn, 'class_weight': None}
        except:
            pass
        
        # Random Undersampling
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_rus, y_rus = rus.fit_resample(X_train_scaled, y_train)
        strategies['Undersampling'] = {'X_train': X_rus, 'y_train': y_rus, 'class_weight': None}
        
        # SMOTE + Tomek
        try:
            smotetomek = SMOTETomek(random_state=RANDOM_STATE)
            X_st, y_st = smotetomek.fit_resample(X_train_scaled, y_train)
            strategies['SMOTE-Tomek'] = {'X_train': X_st, 'y_train': y_st, 'class_weight': None}
        except:
            pass
    
    results = []
    confusion_matrices = {}
    
    for name, config in strategies.items():
        print(f"\n🔄 Training with {name}...")
        
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE,
            class_weight=config['class_weight'],
            n_jobs=-1
        )
        model.fit(config['X_train'], config['y_train'])
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results.append({
            'Strategy': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'Train Size': len(config['y_train'])
        })
        
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        print(f"   ✅ F1: {results[-1]['F1-Score']:.4f}, Recall: {results[-1]['Recall']:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df, confusion_matrices, (X_test_scaled, y_test)

# ============================================
# 5. VISUALIZE RESULTS
# ============================================
def visualize_comparison(results_df, confusion_matrices):
    """Visualize comparison of balancing strategies"""
    print("\n" + "-" * 40)
    print("5. VISUALIZING RESULTS")
    print("-" * 40)
    
    # Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(results_df['Strategy'], results_df[metric], color=colors)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Balancing Strategy')
        ax.set_xticklabels(results_df['Strategy'], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 05_strategy_comparison.png")
    
    # Confusion matrices
    n_strategies = len(confusion_matrices)
    n_cols = 3
    n_rows = (n_strategies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_strategies > 1 else [axes]
    
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
        axes[i].set_title(f'{name}')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 05_confusion_matrices.png")
    
    # Precision-Recall trade-off
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Recall'], results_df['Precision'], 
                s=200, c=range(len(results_df)), cmap='Set2')
    
    for i, row in results_df.iterrows():
        plt.annotate(row['Strategy'], (row['Recall'], row['Precision']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Trade-off by Strategy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 05_precision_recall_tradeoff.png")

# ============================================
# 6. PRINT SUMMARY
# ============================================
def print_summary(results_df):
    """Print formatted summary"""
    print("\n" + "=" * 80)
    print("BALANCING STRATEGIES COMPARISON")
    print("=" * 80)
    
    print("\n📊 Results Table:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC':>10}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['Strategy']:<20} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} "
              f"{row['Recall']:>10.4f} {row['F1-Score']:>10.4f} {row['AUC-ROC']:>10.4f}")
    
    print("-" * 80)
    
    # Best strategy for different goals
    print("\n🏆 Best Strategy by Metric:")
    print(f"   - Best Recall: {results_df.loc[results_df['Recall'].idxmax(), 'Strategy']} ({results_df['Recall'].max():.4f})")
    print(f"   - Best Precision: {results_df.loc[results_df['Precision'].idxmax(), 'Strategy']} ({results_df['Precision'].max():.4f})")
    print(f"   - Best F1-Score: {results_df.loc[results_df['F1-Score'].idxmax(), 'Strategy']} ({results_df['F1-Score'].max():.4f})")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("   - For catching most hits (high recall): Use SMOTE or Undersampling")
    print("   - For confident predictions (high precision): Use Class Weight")
    print("   - For balanced performance: Use SMOTE-Tomek or Class Weight")
    
    # Save results
    results_df.to_csv(f'{OUTPUT_DIR}balancing_comparison_results.csv', index=False)
    print(f"\n💾 Results saved to: {OUTPUT_DIR}balancing_comparison_results.csv")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load data
    X, y, features = load_and_prepare_data()
    
    # 1. Analyze imbalance
    class_counts, ratio = analyze_imbalance(y)
    
    # 2. Baseline model
    baseline_metrics, data_splits, scaler = train_baseline(X, y)
    
    # 4. Compare strategies
    results_df, confusion_matrices, test_data = compare_strategies(X, y)
    
    # 5. Visualize
    visualize_comparison(results_df, confusion_matrices)
    
    # 6. Summary
    print_summary(results_df)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 06 - IMBALANCED LEARNING COMPLETED!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return results_df

if __name__ == "__main__":
    results = main()
