"""
Phase 04 - Feature Importance & Explainability
===============================================
This script analyzes feature importance using:
- Random Forest feature importance
- Permutation importance
- SHAP values
- LIME explanations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Install with: pip install lime")

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================
# CONFIGURATION
# ============================================
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'spotify_cleaned.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'feature_importance/')
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
    print("PHASE 04 - FEATURE IMPORTANCE & EXPLAINABILITY")
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
# 1. RANDOM FOREST FEATURE IMPORTANCE
# ============================================
def rf_feature_importance(X, y, features):
    """Calculate and visualize Random Forest feature importance"""
    print("\n" + "-" * 40)
    print("1. RANDOM FOREST FEATURE IMPORTANCE")
    print("-" * 40)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
    plt.barh(range(len(importance)), importance['importance'].values, color=colors)
    plt.yticks(range(len(importance)), importance['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(importance['importance'].values):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 01_rf_feature_importance.png")
    print("\n📊 Top 5 Most Important Features:")
    for _, row in importance.head(5).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    return rf, scaler, X_train_scaled, X_test_scaled, y_train, y_test, importance

# ============================================
# 2. PERMUTATION IMPORTANCE
# ============================================
def calc_permutation_importance(model, X_test, y_test, features):
    """Calculate permutation importance"""
    print("\n" + "-" * 40)
    print("2. PERMUTATION IMPORTANCE")
    print("-" * 40)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Create DataFrame
    perm_df = pd.DataFrame({
        'feature': features,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(perm_df)), perm_df['importance_mean'].values, 
             xerr=perm_df['importance_std'].values, color='#667eea', alpha=0.8)
    plt.yticks(range(len(perm_df)), perm_df['feature'].values)
    plt.xlabel('Mean Accuracy Decrease')
    plt.title('Permutation Feature Importance')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}02_permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 02_permutation_importance.png")
    print("\n📊 Top 5 by Permutation Importance:")
    for _, row in perm_df.head(5).iterrows():
        print(f"   - {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
    
    return perm_df

# ============================================
# 3. XGBOOST FEATURE IMPORTANCE (if available)
# ============================================
def xgb_feature_importance(X_train, X_test, y_train, y_test, features):
    """Calculate XGBoost feature importance"""
    if not XGBOOST_AVAILABLE:
        print("\n⚠️ Skipping XGBoost (not installed)")
        return None
    
    print("\n" + "-" * 40)
    print("3. XGBOOST FEATURE IMPORTANCE")
    print("-" * 40)
    
    # Train XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=5, 
                        random_state=RANDOM_STATE, use_label_encoder=False,
                        eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    # Get importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance['importance'].values, color='#f093fb')
    plt.yticks(range(len(importance)), importance['feature'].values)
    plt.xlabel('Feature Importance (Gain)')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_xgb_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 03_xgb_feature_importance.png")
    
    return xgb, importance

# ============================================
# 4. SHAP ANALYSIS
# ============================================
def shap_analysis(model, X_train, X_test, features):
    """Perform SHAP analysis"""
    if not SHAP_AVAILABLE:
        print("\n⚠️ Skipping SHAP analysis (not installed)")
        return None
    
    print("\n" + "-" * 40)
    print("4. SHAP ANALYSIS")
    print("-" * 40)
    
    print("   Computing SHAP values (this may take a moment)...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Use sample for speed
    sample_size = min(1000, len(X_test))
    X_sample = X_test[:sample_size]
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use values for positive class
    
    # 4.1 Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=features, 
                      plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Mean |SHAP|)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_shap_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 04_shap_importance_bar.png")
    
    # 4.2 Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 04_shap_summary.png")
    
    # 4.3 Dependence plots for top features - skip if there are issues
    try:
        mean_shap = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(mean_shap)
        top_3 = [int(sorted_indices[-1]), int(sorted_indices[-2]), int(sorted_indices[-3])]
        
        # Convert X_sample to DataFrame for proper feature handling
        X_sample_df = pd.DataFrame(X_sample, columns=features)
        features_list = list(features)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, idx in enumerate(top_3):
            plt.sca(axes[i])
            feature_name = features_list[idx]
            shap.dependence_plot(feature_name, shap_values, X_sample_df, 
                                feature_names=features_list, show=False, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}04_shap_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 04_shap_dependence.png")
    except Exception as e:
        print(f"⚠️ Skipping dependence plots: {e}")
    
    return shap_values, explainer

# ============================================
# 5. LIME EXPLANATION
# ============================================
def lime_explanation(model, X_train, X_test, y_test, features, scaler):
    """Generate LIME explanations for individual predictions"""
    if not LIME_AVAILABLE:
        print("\n⚠️ Skipping LIME analysis (not installed)")
        return None
    
    print("\n" + "-" * 40)
    print("5. LIME EXPLANATIONS")
    print("-" * 40)
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=features,
        class_names=['Non-Hit', 'Hit'],
        mode='classification'
    )
    
    # Find interesting examples
    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    
    # Find a confident hit prediction
    hit_idx = np.where((predictions == 1) & (probas > 0.7))[0]
    # Find a confident non-hit prediction
    nonhit_idx = np.where((predictions == 0) & (probas < 0.3))[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Explain a hit prediction
    if len(hit_idx) > 0:
        exp = explainer.explain_instance(X_test[hit_idx[0]], model.predict_proba, num_features=10)
        
        # Extract data for plotting
        exp_list = exp.as_list()
        features_exp = [x[0] for x in exp_list]
        values = [x[1] for x in exp_list]
        colors = ['#4ecdc4' if v > 0 else '#ff6b6b' for v in values]
        
        axes[0].barh(range(len(features_exp)), values, color=colors)
        axes[0].set_yticks(range(len(features_exp)))
        axes[0].set_yticklabels(features_exp)
        axes[0].set_xlabel('Contribution to Prediction')
        axes[0].set_title(f'LIME: Why is this predicted as HIT? (prob={probas[hit_idx[0]]:.2f})')
        axes[0].axvline(0, color='black', linewidth=0.5)
    
    # Explain a non-hit prediction
    if len(nonhit_idx) > 0:
        exp = explainer.explain_instance(X_test[nonhit_idx[0]], model.predict_proba, num_features=10)
        
        exp_list = exp.as_list()
        features_exp = [x[0] for x in exp_list]
        values = [x[1] for x in exp_list]
        colors = ['#4ecdc4' if v > 0 else '#ff6b6b' for v in values]
        
        axes[1].barh(range(len(features_exp)), values, color=colors)
        axes[1].set_yticks(range(len(features_exp)))
        axes[1].set_yticklabels(features_exp)
        axes[1].set_xlabel('Contribution to Prediction')
        axes[1].set_title(f'LIME: Why is this predicted as NON-HIT? (prob={probas[nonhit_idx[0]]:.2f})')
        axes[1].axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_lime_explanations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 05_lime_explanations.png")
    
    return explainer

# ============================================
# 6. FEATURE IMPORTANCE COMPARISON
# ============================================
def compare_importance_methods(rf_importance, perm_importance, xgb_importance=None):
    """Compare feature importance across different methods"""
    print("\n" + "-" * 40)
    print("6. IMPORTANCE METHOD COMPARISON")
    print("-" * 40)
    
    # Merge importance DataFrames
    comparison = rf_importance.copy()
    comparison = comparison.rename(columns={'importance': 'Random Forest'})
    comparison = comparison.merge(
        perm_importance[['feature', 'importance_mean']].rename(
            columns={'importance_mean': 'Permutation'}), 
        on='feature'
    )
    
    if xgb_importance is not None:
        comparison = comparison.merge(
            xgb_importance.rename(columns={'importance': 'XGBoost'}),
            on='feature'
        )
    
    # Normalize for comparison
    for col in comparison.columns[1:]:
        comparison[col] = comparison[col] / comparison[col].max()
    
    # Visualize
    comparison_melted = comparison.melt(id_vars='feature', var_name='Method', value_name='Importance')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_melted, x='feature', y='Importance', hue='Method')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importance Comparison Across Methods')
    plt.legend(title='Method')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 06_importance_comparison.png")
    
    # Print consensus top features
    comparison['avg_rank'] = comparison.iloc[:, 1:].rank(ascending=False).mean(axis=1)
    consensus = comparison.sort_values('avg_rank').head(5)
    
    print("\n🏆 Consensus Top 5 Most Important Features:")
    for _, row in consensus.iterrows():
        print(f"   - {row['feature']} (avg rank: {row['avg_rank']:.1f})")
    
    return comparison

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load data
    X, y, features = load_and_prepare_data()
    
    # 1. Random Forest importance
    rf, scaler, X_train, X_test, y_train, y_test, rf_imp = rf_feature_importance(X, y, features)
    
    # 2. Permutation importance
    perm_imp = calc_permutation_importance(rf, X_test, y_test, features)
    
    # 3. XGBoost importance
    xgb_result = xgb_feature_importance(X_train, X_test, y_train, y_test, features)
    xgb_imp = xgb_result[1] if xgb_result else None
    
    # 4. SHAP analysis
    shap_analysis(rf, X_train, X_test, features)
    
    # 5. LIME explanations
    lime_explanation(rf, X_train, X_test, y_test, features, scaler)
    
    # 6. Compare methods
    compare_importance_methods(rf_imp, perm_imp, xgb_imp)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 04 - FEATURE IMPORTANCE COMPLETED!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
