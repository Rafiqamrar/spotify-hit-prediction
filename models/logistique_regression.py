# 02_logistique_regression.py
# =====================================================
# Phase 02 : Logistic Regression Model for Track Popularity
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. LOAD AND PREPARE DATA
# =====================================================
print("📂 Loading dataset...")
df = pd.read_csv("../data/spotify_songs.csv")

print(f"Original shape: {df.shape}")

# Remove missing values
df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])
print(f"After removing missing values: {df.shape}")

# Remove duplicates based on track_id
df = df.drop_duplicates(subset=["track_id"])
print(f"After removing duplicates: {df.shape}")

# =====================================================
# 2. DEFINE TARGET VARIABLE (BINARY CLASSIFICATION)
# =====================================================
# Convert popularity to binary: 1 if popular (>= 50), 0 otherwise
df['popularity_binary'] = (df['track_popularity'] >= 50).astype(int)

print("\n📊 Target variable distribution:")
print(df['popularity_binary'].value_counts())
print(f"Class balance: {df['popularity_binary'].value_counts(normalize=True)}")

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================
print("\n🔧 Feature Engineering...")

# Select audio features for the model
audio_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

# Categorical features
categorical_features = ["playlist_genre"]

# Create X (features) and y (target)
X = df[audio_features + categorical_features].copy()
y = df['popularity_binary'].copy()

# Encode categorical variables
le = LabelEncoder()
X['playlist_genre'] = le.fit_transform(X['playlist_genre'])

print(f"Features shape: {X.shape}")
print(f"Features used: {X.columns.tolist()}")

# =====================================================
# 4. SPLIT DATA
# =====================================================
print("\n📋 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# =====================================================
# 5. STANDARDIZE FEATURES
# =====================================================
print("\n⚙️ Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 6. TRAIN LOGISTIC REGRESSION MODEL
# =====================================================
print("\n🚀 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Model trained successfully!")

# =====================================================
# 7. MAKE PREDICTIONS
# =====================================================
print("\n🎯 Making predictions...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

# =====================================================
# 8. EVALUATE MODEL
# =====================================================
print("\n📈 MODEL EVALUATION\n")

print("=" * 60)
print("TRAINING SET METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Precision: {precision_score(y_train, y_pred_train):.4f}")
print(f"Recall:    {recall_score(y_train, y_pred_train):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_pred_train):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_train, y_pred_proba_train):.4f}")

print("\n" + "=" * 60)
print("TEST SET METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_test):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_test):.4f}")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (TEST SET)")
print("=" * 60)
print(classification_report(y_test, y_pred_test, target_names=["Not Popular", "Popular"]))

# =====================================================
# 9. CONFUSION MATRIX
# =====================================================
print("\n" + "=" * 60)
print("CONFUSION MATRIX (TEST SET)")
print("=" * 60)
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# =====================================================
# 10. FEATURE IMPORTANCE (COEFFICIENTS)
# =====================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Model Coefficients)")
print("=" * 60)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(feature_importance)

# =====================================================
# 11. VISUALIZATIONS
# =====================================================

# --- Confusion Matrix Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Popular', 'Popular'],
            yticklabels=['Not Popular', 'Popular'])
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Feature Importance Bar Plot ---
plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in feature_importance['coefficient']]
plt.barh(feature_importance['feature'], feature_importance['coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Prediction Probability Distribution ---
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba_test[y_test == 0], bins=30, alpha=0.6, label='Not Popular (Actual)')
plt.hist(y_pred_proba_test[y_test == 1], bins=30, alpha=0.6, label='Popular (Actual)')
plt.xlabel('Predicted Probability of Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# 12. CONCLUSION
# =====================================================
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print(f"""
✅ Model successfully trained on {len(X_train)} samples
📊 Test set performance:
   - Accuracy: {accuracy_score(y_test, y_pred_test):.2%}
   - ROC-AUC: {roc_auc_score(y_test, y_pred_proba_test):.4f}

🔍 Key insights:
   - Most important features: {', '.join(feature_importance['feature'].head(3).tolist())}
   - The model can classify tracks as popular/unpopular with reasonable accuracy
   - Consider ensemble methods or neural networks for better performance

📁 Generated files:
   - confusion_matrix.png
   - roc_curve.png
   - feature_importance.png
   - probability_distribution.png
""")
