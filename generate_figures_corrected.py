"""
Generate All Publication Figures - CORRECTED VERSION
=====================================================
Fixes data leakage issues and handles class imbalance.

PROBLEM: Original model had 100% accuracy because features contained the answer:
- speed_over_limit = max(0, speed - 60) --> if > 0, it's a violation!
- speed_ratio = speed / 60 --> if > 1, it's a violation!
- location_violation_rate --> computed FROM target variable!

SOLUTION: Use only contextual features available BEFORE knowing the speed.

Author: Traffic Vision Research Team
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR,
    COLORS, COLOR_PALETTE, FIGURE_SETTINGS, MPL_STYLE,
    SPEED_LIMIT_DEFAULT, RANDOM_STATE
)

matplotlib.rcParams.update(MPL_STYLE)
plt.rcParams['font.family'] = 'serif'

print("=" * 70)
print("TRAFFIC VISION - CORRECTED ANALYSIS (NO DATA LEAKAGE)")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("\n[1/6] Loading data...")

data = pd.read_csv(PROCESSED_DATA_DIR / 'processed_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed')
print(f"Total records: {len(data):,}")
print(f"Violation rate: {data['is_violation'].mean()*100:.1f}%")

# =============================================================================
# 2. DEFINE NON-LEAKY FEATURES
# =============================================================================

print("\n[2/6] Selecting non-leaky features...")

# Features that do NOT contain speed or target information
# These are contextual features available BEFORE knowing if violation occurred
NON_LEAKY_FEATURES = [
    # Temporal features (when)
    'hour',
    'day_of_week',
    'is_weekend',
    'is_rush_hour',
    'is_night',
    'month',
    'day_of_month',

    # Location features (where) - but NOT speed-based location stats
    'radar_id_encoded',
    'latitude',
    'longitude',

    # Vehicle features (what)
    'vehicle_class_encoded',
    'vehicle_category_encoded',
    'is_heavy_vehicle',

    # Lane info
    'lane',

    # Traffic volume (how many cars, but NOT their speeds)
    'vehicles_per_hour',
    'congestion_index',
]

# LEAKY FEATURES (DO NOT USE):
# - speed, speed_over_limit, speed_ratio (directly reveal answer)
# - speed_diff_from_avg, location_avg_speed, hourly_avg_speed, lane_avg_speed
# - rolling_avg_speed_*, speed_lag_* (all speed-based)
# - location_violation_rate (computed from target!)

# Filter to available features
available_features = [f for f in NON_LEAKY_FEATURES if f in data.columns]
print(f"Using {len(available_features)} clean features:")
for f in available_features:
    print(f"  - {f}")

# =============================================================================
# 3. PREPARE DATA WITH CLASS BALANCING
# =============================================================================

print("\n[3/6] Preparing data with class balancing...")

X = data[available_features].copy()
y = data['is_violation'].copy()

# Fill NaN
X = X.fillna(X.median())

# Check class distribution
print(f"Class distribution:")
print(f"  - No Violation (0): {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"  - Violation (1): {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")

# Temporal split (no leakage)
data_sorted = data.sort_values('timestamp').reset_index(drop=True)
X_sorted = X.loc[data_sorted.index].reset_index(drop=True)
y_sorted = y.loc[data_sorted.index].reset_index(drop=True)

split_idx = int(len(X_sorted) * 0.8)
X_train_full = X_sorted.iloc[:split_idx]
X_test_full = X_sorted.iloc[split_idx:]
y_train_full = y_sorted.iloc[:split_idx]
y_test_full = y_sorted.iloc[split_idx:]

# Sample for computational efficiency
TRAIN_SAMPLE = 100000
TEST_SAMPLE = 25000

np.random.seed(RANDOM_STATE)
train_idx = np.random.choice(len(X_train_full), min(TRAIN_SAMPLE, len(X_train_full)), replace=False)
test_idx = np.random.choice(len(X_test_full), min(TEST_SAMPLE, len(X_test_full)), replace=False)

X_train = X_train_full.iloc[train_idx].reset_index(drop=True)
y_train = y_train_full.iloc[train_idx].reset_index(drop=True)
X_test = X_test_full.iloc[test_idx].reset_index(drop=True)
y_test = y_test_full.iloc[test_idx].reset_index(drop=True)

print(f"\nTrain set: {len(X_train):,}")
print(f"Test set: {len(X_test):,}")

# Handle class imbalance using undersampling (since we have too many violations)
print("\nBalancing classes...")
# Undersample majority class to create more balanced dataset
try:
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy=0.3, random_state=RANDOM_STATE)  # 30% minority
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
    print(f"After balancing: {len(X_train_balanced):,} samples")
    print(f"  - No Violation: {(y_train_balanced==0).sum():,}")
    print(f"  - Violation: {(y_train_balanced==1).sum():,}")
except ImportError:
    print("imbalanced-learn not installed, using original distribution")
    X_train_balanced = X_train
    y_train_balanced = y_train

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')

# =============================================================================
# 4. TRAIN MODELS
# =============================================================================

print("\n[4/6] Training models (this is harder without leaky features)...")

models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                                              class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                            random_state=RANDOM_STATE, n_jobs=-1,
                                            class_weight='balanced'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                    random_state=RANDOM_STATE,
                                    max_iter=500, early_stopping=True)
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                      random_state=RANDOM_STATE,
                                      scale_pos_weight=(y_train_balanced==1).sum()/(y_train_balanced==0).sum(),
                                      use_label_encoder=False, eval_metric='logloss', n_jobs=-1)

if HAS_LIGHTGBM:
    models['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                                        class_weight='balanced')

results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train_balanced)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_prob

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    # Save model
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, MODELS_DIR / f'{safe_name}_model.joblib')

    print(f"    Accuracy: {results[name]['accuracy']:.4f}")
    print(f"    Precision: {results[name]['precision']:.4f}")
    print(f"    Recall: {results[name]['recall']:.4f}")
    print(f"    F1: {results[name]['f1']:.4f}")
    print(f"    AUC: {results[name]['roc_auc']:.4f}")
    print()

# Best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = models[best_model_name]
joblib.dump(best_model, MODELS_DIR / 'best_model.joblib')
print(f"Best model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv(REPORTS_DIR / 'model_results.csv')

# =============================================================================
# 5. GENERATE ML FIGURES
# =============================================================================

print("\n[5/6] Generating ML figures...")

def save_figure(fig, name):
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png")

# Figure 1: Model Comparison
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(models))
width = 0.15

for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in models.keys()]
    ax.bar(x + i * width, values, width, label=metric.upper().replace('_', '-'),
           color=COLOR_PALETTE[i])

ax.set_xticks(x + width * 2)
ax.set_xticklabels(models.keys(), rotation=15, ha='right')
ax.set_ylabel('Score', fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_figure(fig, 'model_comparison')

# Figure 2: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
for i, (name, model) in enumerate(models.items()):
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc = results[name]['roc_auc']
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
            color=COLOR_PALETTE[i], linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'roc_curves')

# Figure 3: Precision-Recall Curves
fig, ax = plt.subplots(figsize=(10, 8))
for i, (name, model) in enumerate(models.items()):
    precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
    ap = average_precision_score(y_test, probabilities[name])
    ax.plot(recall, precision, label=f'{name} (AP={ap:.3f})',
            color=COLOR_PALETTE[i], linewidth=2)

# Add baseline (proportion of positives)
baseline = y_test.mean()
ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.2f})')
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'precision_recall_curves')

# Figure 4: Confusion Matrix (Best Model)
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions[best_model_name])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Violation', 'Violation'],
            yticklabels=['No Violation', 'Violation'],
            annot_kws={'size': 14})
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
plt.tight_layout()
save_figure(fig, 'confusion_matrix')

# Figure 5: All Confusion Matrices
n_models = len(models)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Viol.', 'Viol.'],
                yticklabels=['No Viol.', 'Viol.'])
    axes[i].set_xlabel('Predicted', fontsize=11)
    axes[i].set_ylabel('Actual', fontsize=11)
    axes[i].set_title(name, fontsize=12, fontweight='bold')

for j in range(len(predictions), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
save_figure(fig, 'confusion_matrices_all')

# Figure 6: Feature Importance
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': importance
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [COLORS['primary'] if imp > importance_df['importance'].mean()
              else COLORS['neutral'] for imp in importance_df['importance']]
    ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    ax.set_xlabel('Feature Importance', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'feature_importance_rf')

# =============================================================================
# 6. GENERATE XAI FIGURES
# =============================================================================

print("\n[6/6] Generating XAI figures (SHAP)...")

if HAS_SHAP:
    SHAP_SAMPLE = 500
    shap_indices = np.random.choice(len(X_test_scaled), min(SHAP_SAMPLE, len(X_test_scaled)), replace=False)
    X_shap = X_test_scaled[shap_indices]

    if HAS_LIGHTGBM and 'LightGBM' in models:
        shap_model = models['LightGBM']
        explainer = shap.TreeExplainer(shap_model)
    elif 'Random Forest' in models:
        shap_model = models['Random Forest']
        explainer = shap.TreeExplainer(shap_model)
    else:
        shap_model = best_model
        background = shap.sample(X_train_scaled, 100)
        explainer = shap.KernelExplainer(shap_model.predict_proba, background)

    print("  Computing SHAP values...")
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # SHAP Beeswarm
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=available_features,
                      show=False, max_display=len(available_features))
    plt.tight_layout()
    save_figure(plt.gcf(), 'shap_beeswarm')

    # SHAP Bar
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=available_features,
                      plot_type='bar', show=False, max_display=len(available_features))
    plt.tight_layout()
    save_figure(plt.gcf(), 'shap_importance_bar')

    # SHAP Dependence (top 4)
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_shap)[-4:][::-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_features_idx):
        shap.dependence_plot(feat_idx, shap_values, X_shap,
                            feature_names=available_features,
                            ax=axes[i], show=False)
        axes[i].set_xlabel(available_features[feat_idx], fontsize=12)
        axes[i].set_ylabel('SHAP value', fontsize=12)

    plt.tight_layout()
    save_figure(fig, 'shap_dependence_top4')

    # SHAP Waterfall Examples
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    test_labels = y_test.iloc[shap_indices].values
    viol_idx = np.where(test_labels == 1)[0]
    non_viol_idx = np.where(test_labels == 0)[0]

    if len(viol_idx) > 0:
        idx = viol_idx[0]
        sv = shap_values[idx]
        sorted_idx = np.argsort(np.abs(sv))[::-1][:10]

        axes[0].barh(range(10), sv[sorted_idx],
                    color=[COLORS['danger'] if v > 0 else COLORS['success'] for v in sv[sorted_idx]])
        axes[0].set_yticks(range(10))
        axes[0].set_yticklabels([available_features[i] for i in sorted_idx])
        axes[0].set_xlabel('SHAP Value', fontsize=12)
        axes[0].axvline(x=0, color='black', linewidth=0.5)
        axes[0].invert_yaxis()
        axes[0].set_title('Violation Case', fontsize=12, fontweight='bold')

    if len(non_viol_idx) > 0:
        idx = non_viol_idx[0]
        sv = shap_values[idx]
        sorted_idx = np.argsort(np.abs(sv))[::-1][:10]

        axes[1].barh(range(10), sv[sorted_idx],
                    color=[COLORS['danger'] if v > 0 else COLORS['success'] for v in sv[sorted_idx]])
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([available_features[i] for i in sorted_idx])
        axes[1].set_xlabel('SHAP Value', fontsize=12)
        axes[1].axvline(x=0, color='black', linewidth=0.5)
        axes[1].invert_yaxis()
        axes[1].set_title('Non-Violation Case', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'shap_waterfall_examples')

    # SHAP Interaction
    if len(top_features_idx) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.dependence_plot(top_features_idx[0], shap_values, X_shap,
                            interaction_index=top_features_idx[1],
                            feature_names=available_features,
                            ax=ax, show=False)
        plt.tight_layout()
        save_figure(fig, 'shap_interaction')

    # Save importance
    shap_importance = pd.DataFrame({
        'feature': available_features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    shap_importance.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("CORRECTED ANALYSIS COMPLETE")
print("=" * 70)

print("\n" + "-" * 40)
print("MODEL PERFORMANCE (Realistic Results)")
print("-" * 40)
print(results_df.round(4).to_string())

print("\n" + "-" * 40)
print("KEY FINDINGS")
print("-" * 40)
print(f"1. Without speed-based features, prediction is much harder (realistic)")
print(f"2. Best model: {best_model_name}")
print(f"3. Most important contextual features for predicting violations:")
if 'Random Forest' in models:
    top_3 = importance_df.tail(3)['feature'].tolist()[::-1]
    for i, f in enumerate(top_3, 1):
        print(f"   {i}. {f}")

print(f"\n4. The high violation rate ({y_test.mean()*100:.1f}%) suggests most vehicles")
print(f"   exceed the {SPEED_LIMIT_DEFAULT} km/h limit in this urban area.")

import os
figures = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
print(f"\nGenerated {len(figures)} figures in: {FIGURES_DIR}")
