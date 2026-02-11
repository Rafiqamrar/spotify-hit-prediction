"""
Phase 08 - Regression Analysis
==============================
This script predicts the exact popularity score (0-100):
- Regression models comparison
- Regression vs Classification comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================
# CONFIGURATION
# ============================================
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'spotify_cleaned.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'regression/')
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
    """Load data and prepare for regression"""
    print("=" * 60)
    print("PHASE 08 - REGRESSION ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dataset loaded: {df.shape[0]} rows")
    
    # Select features
    features = [f for f in AUDIO_FEATURES if f in df.columns]
    engineered = ['energy_dance_ratio', 'mood_score', 'acoustic_electronic']
    features.extend([f for f in engineered if f in df.columns])
    
    X = df[features].dropna()
    y = df.loc[X.index, 'track_popularity']  # Continuous target
    
    print(f"📊 Target variable statistics:")
    print(f"   Mean: {y.mean():.2f}")
    print(f"   Std: {y.std():.2f}")
    print(f"   Min: {y.min()}, Max: {y.max()}")
    
    return X, y, features, df

# ============================================
# 1. DEFINE REGRESSION MODELS
# ============================================
def get_regression_models():
    """Return dictionary of regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso Regression': Lasso(alpha=0.1, random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        'Neural Network (MLP)': MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500, 
            random_state=RANDOM_STATE, early_stopping=True
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        )
    
    return models

# ============================================
# 2. TRAIN AND EVALUATE REGRESSION MODELS
# ============================================
def train_and_evaluate(X, y, models):
    """Train and evaluate regression models"""
    print("\n" + "-" * 40)
    print("2. TRAINING REGRESSION MODELS")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'CV MAE': cv_mae
        })
        
        print(f"   ✅ MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    
    results_df = pd.DataFrame(results)
    
    return results_df, predictions, (X_test_scaled, y_test), scaler

# ============================================
# 3. VISUALIZE REGRESSION RESULTS
# ============================================
def visualize_results(results_df, predictions, test_data):
    """Visualize regression results"""
    print("\n" + "-" * 40)
    print("3. VISUALIZING RESULTS")
    print("-" * 40)
    
    X_test, y_test = test_data
    
    # 1. Metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    # MAE
    axes[0].bar(results_df['Model'], results_df['MAE'], color=colors)
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('MAE by Model (Lower is Better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # RMSE
    axes[1].bar(results_df['Model'], results_df['RMSE'], color=colors)
    axes[1].set_ylabel('Root Mean Squared Error')
    axes[1].set_title('RMSE by Model (Lower is Better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R²
    axes[2].bar(results_df['Model'], results_df['R²'], color=colors)
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('R² by Model (Higher is Better)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_regression_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 01_regression_metrics.png")
    
    # 2. Actual vs Predicted plots
    best_models = results_df.nsmallest(4, 'MAE')['Model'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(best_models):
        y_pred = predictions[model_name]
        
        axes[i].scatter(y_test, y_pred, alpha=0.3, s=10)
        axes[i].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
        axes[i].set_xlabel('Actual Popularity')
        axes[i].set_ylabel('Predicted Popularity')
        axes[i].set_title(f'{model_name}')
        axes[i].legend()
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 02_actual_vs_predicted.png")
    
    # 3. Residual plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(best_models):
        y_pred = predictions[model_name]
        residuals = y_test - y_pred
        
        axes[i].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[i].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[i].set_xlabel('Predicted Popularity')
        axes[i].set_ylabel('Residuals')
        axes[i].set_title(f'{model_name} Residuals')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_residual_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 03_residual_plots.png")
    
    # 4. Error distribution
    plt.figure(figsize=(12, 6))
    
    for model_name in best_models:
        y_pred = predictions[model_name]
        errors = y_test - y_pred
        plt.hist(errors, bins=50, alpha=0.5, label=model_name)
    
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution by Model')
    plt.legend()
    plt.axvline(0, color='black', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 04_error_distribution.png")

# ============================================
# 4. REGRESSION VS CLASSIFICATION
# ============================================
def regression_vs_classification(results_df, predictions, y_test, threshold=70):
    """Compare regression vs classification approaches"""
    print("\n" + "-" * 40)
    print("4. REGRESSION VS CLASSIFICATION COMPARISON")
    print("-" * 40)
    
    # Convert predictions to binary using threshold
    y_test_binary = (y_test >= threshold).astype(int)
    
    classification_from_regression = []
    
    for model_name, y_pred in predictions.items():
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate classification metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        acc = accuracy_score(y_test_binary, y_pred_binary)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        
        classification_from_regression.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
    
    class_results = pd.DataFrame(classification_from_regression)
    
    print("\n📊 Classification Metrics (from Regression predictions):")
    print(f"   Threshold: popularity >= {threshold} = Hit")
    print(class_results.to_string(index=False))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regression metrics
    results_sorted = results_df.sort_values('MAE')
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))
    
    axes[0].barh(results_sorted['Model'], results_sorted['MAE'], color=colors)
    axes[0].set_xlabel('Mean Absolute Error')
    axes[0].set_title('Regression Performance (MAE)')
    axes[0].invert_yaxis()
    
    # Classification from regression
    class_sorted = class_results.sort_values('F1-Score', ascending=False)
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(class_sorted)))
    
    axes[1].barh(class_sorted['Model'], class_sorted['F1-Score'], color=colors2)
    axes[1].set_xlabel('F1-Score')
    axes[1].set_title('Classification from Regression (F1-Score)')
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}05_regression_vs_classification.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved: 05_regression_vs_classification.png")
    
    # Analysis
    print("\n💡 Key Insights:")
    best_reg = results_df.loc[results_df['MAE'].idxmin(), 'Model']
    best_class = class_results.loc[class_results['F1-Score'].idxmax(), 'Model']
    
    print(f"   - Best regression model: {best_reg}")
    print(f"   - Best for classification (via regression): {best_class}")
    print(f"\n   Regression predicts exact popularity (more informative)")
    print(f"   Classification predicts binary hit/non-hit (simpler decision)")
    
    return class_results

# ============================================
# 5. FEATURE IMPORTANCE (REGRESSION)
# ============================================
def regression_feature_importance(X, y, features):
    """Feature importance for regression"""
    print("\n" + "-" * 40)
    print("5. FEATURE IMPORTANCE FOR REGRESSION")
    print("-" * 40)
    
    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, 
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
    plt.barh(range(len(importance)), importance['Importance'].values, color=colors)
    plt.yticks(range(len(importance)), importance['Feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Popularity Prediction (Regression)')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06_regression_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 06_regression_feature_importance.png")
    
    print("\n📊 Top 5 Most Important Features:")
    for _, row in importance.head(5).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")
    
    return importance

# ============================================
# 6. PRINT SUMMARY
# ============================================
def print_summary(results_df):
    """Print regression summary"""
    print("\n" + "=" * 80)
    print("REGRESSION MODELS SUMMARY")
    print("=" * 80)
    
    print("\n📊 Results Table (sorted by MAE):")
    results_sorted = results_df.sort_values('MAE')
    
    print("-" * 70)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'CV MAE':>10}")
    print("-" * 70)
    
    for _, row in results_sorted.iterrows():
        print(f"{row['Model']:<25} {row['MAE']:>10.2f} {row['RMSE']:>10.2f} "
              f"{row['R²']:>10.4f} {row['CV MAE']:>10.2f}")
    
    print("-" * 70)
    
    # Best model
    best = results_sorted.iloc[0]
    print(f"\n🏆 Best Model: {best['Model']}")
    print(f"   MAE: {best['MAE']:.2f} (average error in popularity points)")
    print(f"   R²: {best['R²']:.4f} (variance explained)")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    print(f"   - On average, predictions are off by ~{best['MAE']:.1f} popularity points")
    print(f"   - The model explains {best['R²']*100:.1f}% of the variance in popularity")
    
    # Save results
    results_df.to_csv(f'{OUTPUT_DIR}regression_results.csv', index=False)
    print(f"\n💾 Results saved to: {OUTPUT_DIR}regression_results.csv")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Load data
    X, y, features, df = load_and_prepare_data()
    
    # Get models
    models = get_regression_models()
    print(f"\n🤖 Models to compare: {len(models)}")
    
    # Train and evaluate
    results_df, predictions, test_data, scaler = train_and_evaluate(X, y, models)
    
    # Visualize
    visualize_results(results_df, predictions, test_data)
    
    # Regression vs Classification
    X_test, y_test = test_data
    class_results = regression_vs_classification(results_df, predictions, y_test)
    
    # Feature importance
    regression_feature_importance(X, y, features)
    
    # Summary
    print_summary(results_df)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 08 - REGRESSION ANALYSIS COMPLETED!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return results_df

if __name__ == "__main__":
    results = main()
