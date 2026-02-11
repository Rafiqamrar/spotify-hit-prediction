"""
Phase 03 - Model Comparison
===========================
This script compares multiple ML models for predicting song success:
- Logistic Regression
- Random Forest
- XGBoost
- SVM
- Neural Network (MLP)
- Naive Bayes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = '../data/spotify_cleaned.csv'
OUTPUT_DIR = '../outputs/models/'
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
    """Load data and prepare features/target"""
    print("=" * 60)
    print("PHASE 03 - MODEL COMPARISON")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dataset loaded: {df.shape[0]} rows")
    
    # Select features
    features = [f for f in AUDIO_FEATURES if f in df.columns]
    
    # Add engineered features if available
    engineered = ['energy_dance_ratio', 'mood_score', 'acoustic_electronic']
    features.extend([f for f in engineered if f in df.columns])
    
    print(f"📊 Features used: {len(features)}")
    
    X = df[features].dropna()
    y = df.loc[X.index, 'is_hit']
    
    print(f"   Training samples: {len(X)}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, features

# ============================================
# DEFINE MODELS
# ============================================
def get_models():
    """Return dictionary of models to compare"""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True
        ),
        'Naive Bayes': GaussianNB()
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    return models

# ============================================
# TRAIN AND EVALUATE MODELS
# ============================================
def train_and_evaluate(X, y, models):
    """Train and evaluate all models with cross-validation"""
    print("\n" + "-" * 40)
    print("TRAINING AND EVALUATING MODELS")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store results
    results = []
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        start_time = time.time()
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        train_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc,
            'CV F1 Mean': cv_scores.mean(),
            'CV F1 Std': cv_scores.std(),
            'Train Time (s)': train_time
        })
        
        print(f"   ✅ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {train_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    return results_df, trained_models, (X_test_scaled, y_test), scaler

# ============================================
# VISUALIZE RESULTS
# ============================================
def visualize_results(results_df, trained_models, test_data):
    """Create visualizations for model comparison"""
    print("\n" + "-" * 40)
    print("CREATING VISUALIZATIONS")
    print("-" * 40)
    
    X_test, y_test = test_data
    
    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Model')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 01_metrics_comparison.png")
    
    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}02_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 02_roc_curves.png")
    
    # 3. Confusion Matrices
    n_models = len(trained_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
        axes[i].set_title(f'{name}')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 03_confusion_matrices.png")
    
    # 4. Summary heatmap
    plt.figure(figsize=(12, 6))
    summary_metrics = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']]
    sns.heatmap(summary_metrics, annot=True, cmap='RdYlGn', fmt='.3f', 
                linewidths=0.5, vmin=0, vmax=1)
    plt.title('Model Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 04_performance_heatmap.png")

# ============================================
# PRINT SUMMARY TABLE
# ============================================
def print_summary(results_df):
    """Print formatted summary table"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Sort by F1-Score
    results_sorted = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n📊 Performance Metrics (sorted by F1-Score):")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
    print("-" * 80)
    
    for _, row in results_sorted.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} "
              f"{row['Recall']:>10.4f} {row['F1-Score']:>10.4f} {row['AUC-ROC']:>10.4f}")
    
    print("-" * 80)
    
    # Best model
    best_model = results_sorted.iloc[0]['Model']
    best_f1 = results_sorted.iloc[0]['F1-Score']
    
    print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
    
    # Save results to CSV
    results_df.to_csv(f'{OUTPUT_DIR}model_comparison_results.csv', index=False)
    print(f"\n💾 Results saved to: {OUTPUT_DIR}model_comparison_results.csv")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load and prepare data
    X, y, features = load_and_prepare_data()
    
    # Get models
    models = get_models()
    print(f"\n🤖 Models to compare: {len(models)}")
    for name in models.keys():
        print(f"   - {name}")
    
    # Train and evaluate
    results_df, trained_models, test_data, scaler = train_and_evaluate(X, y, models)
    
    # Visualize results
    visualize_results(results_df, trained_models, test_data)
    
    # Print summary
    print_summary(results_df)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 03 - MODEL COMPARISON COMPLETED!")
    print("=" * 60)
    
    return results_df, trained_models

if __name__ == "__main__":
    results, models = main()
