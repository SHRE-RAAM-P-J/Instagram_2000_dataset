# =============================================================================
# SR NEXT Self-Guided Internship — Phase 2
# ML Models: Automation Detection + Engagement Rate Prediction
# Student: SHRE RAAM P J
# =============================================================================
# INSTALL REQUIRED LIBRARIES:
#   pip install pandas numpy scikit-learn openpyxl matplotlib seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so it works without a display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, mean_absolute_error, r2_score, mean_squared_error
)

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
# Load the cleaned & scored dataset produced by phase2_analysis.py

print("=" * 60)
print("SR NEXT — Phase 2 ML Models")
print("=" * 60)

df = pd.read_csv('SR_NEXT_Phase2_Ranked_Influencers.csv')
print(f"\nDataset loaded: {len(df):,} influencers")
print(f"Columns: {df.columns.tolist()}")

# =============================================================================
# STEP 2 — PREPARE FEATURES FOR ML
# =============================================================================
# ML models need numbers — convert categories to numbers using Label Encoding

print("\n--- Preparing Features ---")

# Encode Broad_Category as numbers (e.g. Fashion & Beauty → 3)
le_cat = LabelEncoder()
df['Category_Encoded'] = le_cat.fit_transform(df['Broad_Category'].astype(str))

print(f"Category encoding: {dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_)))}")

# Feature set used for BOTH models
# These are the inputs the model learns from
FEATURES = ['Followers', 'Engagement Rate', 'Posts_Per_Month',
            'Hashtag_Count', 'Category_Encoded']

X = df[FEATURES].copy()

# Fill any remaining NaNs just in case
X = X.fillna(X.median())

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features used: {FEATURES}")


# =============================================================================
# MODEL 1 — CLASSIFICATION
# Predict Automation Likelihood: Low / Medium / High
# Algorithm: Random Forest Classifier
# =============================================================================

print("\n" + "=" * 60)
print("MODEL 1: Automation Likelihood Classifier")
print("Algorithm: Random Forest Classifier")
print("=" * 60)

# Target variable — what we want to predict
y_class = df['Automation_Likelihood']

print(f"\nClass distribution:")
print(y_class.value_counts())

# Split data: 80% for training, 20% for testing
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\nTraining set size : {len(X_train_c)}")
print(f"Test set size     : {len(X_test_c)}")

# ── Train Random Forest ───────────────────────────────────────────────────────
# Random Forest = many decision trees working together (ensemble method)
# n_estimators=100 means 100 trees; each tree votes and majority wins

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'   # handles imbalanced classes (Low >> High)
)

rf_classifier.fit(X_train_c, y_train_c)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred_c = rf_classifier.predict(X_test_c)
accuracy  = accuracy_score(y_test_c, y_pred_c)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c))

# Cross-validation — test on 5 different data splits to confirm accuracy
cv_scores = cross_val_score(rf_classifier, X, y_class, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# ── Feature Importance ────────────────────────────────────────────────────────
# Which features does the model rely on most?
importances_c = pd.Series(rf_classifier.feature_importances_, index=FEATURES)
print(f"\nFeature Importances (what the model looks at most):")
print(importances_c.sort_values(ascending=False).round(4))

# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test_c, y_pred_c, labels=['High','Medium','Low'])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['High','Medium','Low'],
            yticklabels=['High','Medium','Low'])
plt.title('Confusion Matrix — Automation Likelihood Classifier\n(Random Forest)', fontsize=13, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_automation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nConfusion matrix saved as: confusion_matrix_automation.png")

# ── Feature Importance Plot ───────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
importances_c.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importances — Automation Classifier', fontsize=12, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_automation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance saved as: feature_importance_automation.png")

# ── Apply model predictions back to full dataset ──────────────────────────────
df['ML_Automation_Prediction'] = rf_classifier.predict(X)
df['ML_Automation_Confidence'] = rf_classifier.predict_proba(X).max(axis=1).round(3)

print(f"\nML Predictions added to dataset:")
print(df['ML_Automation_Prediction'].value_counts())


# =============================================================================
# MODEL 2 — REGRESSION
# Predict Engagement Rate from follower count, posting freq, category
# Algorithm: Random Forest Regressor + Linear Regression (compare both)
# =============================================================================

print("\n" + "=" * 60)
print("MODEL 2: Engagement Rate Predictor")
print("Algorithm: Random Forest Regressor vs Linear Regression")
print("=" * 60)

# Target: Engagement Rate
# Cap at 99th percentile to remove extreme outliers from training
er_cap     = df['Engagement Rate'].quantile(0.99)
df_reg     = df[df['Engagement Rate'] <= er_cap].copy()
y_reg      = df_reg['Engagement Rate']
X_reg      = df_reg[FEATURES].fillna(df_reg[FEATURES].median())

print(f"\nRegression dataset size: {len(df_reg)} (removed {len(df)-len(df_reg)} outliers above {er_cap:.3f})")
print(f"ER range: {y_reg.min():.4f} — {y_reg.max():.4f}")
print(f"Mean ER : {y_reg.mean():.4f} ({y_reg.mean()*100:.2f}%)")

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ── Random Forest Regressor ───────────────────────────────────────────────────
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_regressor.fit(X_train_r, y_train_r)
y_pred_rf = rf_regressor.predict(X_test_r)

mae_rf = mean_absolute_error(y_test_r, y_pred_rf)
r2_rf  = r2_score(y_test_r, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_r, y_pred_rf))

print(f"\nRandom Forest Regressor Results:")
print(f"   R² Score (accuracy of fit) : {r2_rf:.4f}  (1.0 = perfect)")
print(f"   MAE (avg prediction error) : {mae_rf:.4f} ({mae_rf*100:.2f}%)")
print(f"   RMSE                       : {rmse_rf:.4f}")

# ── Linear Regression ─────────────────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train_r)
X_test_s  = scaler.transform(X_test_r)

lr_regressor = LinearRegression()
lr_regressor.fit(X_train_s, y_train_r)
y_pred_lr = lr_regressor.predict(X_test_s)

mae_lr = mean_absolute_error(y_test_r, y_pred_lr)
r2_lr  = r2_score(y_test_r, y_pred_lr)

print(f"\nLinear Regression Results:")
print(f"   R² Score : {r2_lr:.4f}")
print(f"   MAE      : {mae_lr:.4f} ({mae_lr*100:.2f}%)")

print(f"\nBetter model: {'Random Forest' if r2_rf > r2_lr else 'Linear Regression'}")

# ── Feature Importance for Regression ────────────────────────────────────────
importances_r = pd.Series(rf_regressor.feature_importances_, index=FEATURES)
print(f"\nFeature Importances (ER prediction):")
print(importances_r.sort_values(ascending=False).round(4))

# ── Actual vs Predicted Plot ──────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_rf, alpha=0.4, color='steelblue', s=20, label='RF Predictions')
plt.plot([0, y_test_r.max()], [0, y_test_r.max()], 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel('Actual Engagement Rate')
plt.ylabel('Predicted Engagement Rate')
plt.title(f'Actual vs Predicted Engagement Rate\nRandom Forest (R² = {r2_rf:.3f})', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_er.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nActual vs Predicted plot saved as: actual_vs_predicted_er.png")

# ── Feature Importance Regression Plot ───────────────────────────────────────
plt.figure(figsize=(8, 4))
importances_r.sort_values().plot(kind='barh', color='darkorange')
plt.title('Feature Importances — Engagement Rate Predictor', fontsize=12, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_er.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance saved as: feature_importance_er.png")

# Apply predictions back to dataset
X_full_s = scaler.transform(X_reg[FEATURES].fillna(X_reg[FEATURES].median()))
df.loc[df_reg.index, 'ML_ER_Prediction'] = rf_regressor.predict(X_reg).round(4)


# =============================================================================
# STEP 3 — EXPORT FINAL DATASET WITH ML PREDICTIONS
# =============================================================================

print("\n" + "=" * 60)
print("EXPORTING FINAL DATASET WITH ML PREDICTIONS")
print("=" * 60)

output_cols = [
    'Rank', 'Username', 'Followers', 'Broad_Category', 'Category',
    'Engagement Rate', 'Posts_Per_Month', 'Hashtag_Count',
    'Influence_Score', 'Tier',
    'Automation_Likelihood',       # rule-based (original)
    'ML_Automation_Prediction',    # ML model prediction
    'ML_Automation_Confidence',    # how confident the model is (0-1)
    'ML_ER_Prediction',            # predicted engagement rate
    'Auto_Reason', 'Contact'
]

# Keep only columns that exist
output_cols = [c for c in output_cols if c in df.columns]
df_out = df[output_cols].fillna('')
df_out.to_csv('SR_NEXT_Phase2_ML_Results.csv', index=False)
print(f"ML results CSV saved: SR_NEXT_Phase2_ML_Results.csv")
print(f"   Rows: {len(df_out):,}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2 ML MODELS — FINAL SUMMARY")
print("=" * 60)
print(f"\nMODEL 1 — Automation Classifier (Random Forest)")
print(f"  Accuracy          : {accuracy*100:.2f}%")
print(f"  CV Accuracy       : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"  Top feature       : {importances_c.idxmax()}")

print(f"\nMODEL 2 — Engagement Rate Predictor")
print(f"  Random Forest R²  : {r2_rf:.4f}")
print(f"  Random Forest MAE : {mae_rf*100:.2f}%")
print(f"  Linear Reg R²     : {r2_lr:.4f}")
print(f"  Top feature       : {importances_r.idxmax()}")

print(f"\nOUTPUT FILES GENERATED:")
print(f"  SR_NEXT_Phase2_ML_Results.csv")
print(f"  confusion_matrix_automation.png")
print(f"  feature_importance_automation.png")
print(f"  actual_vs_predicted_er.png")
print(f"  feature_importance_er.png")

print("\nAll ML models complete!")
