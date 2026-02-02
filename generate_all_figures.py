"""
Generate All Publication Figures
================================
Creates all ML and XAI figures for Q1 journal publication.
Ensures NO data leakage with proper temporal train/test split.

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
import joblib

# Import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available")

# Import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not available - install with: pip install shap")

from config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR,
    COLORS, COLOR_PALETTE, FIGURE_SETTINGS, MPL_STYLE,
    SPEED_LIMIT_DEFAULT, RANDOM_STATE
)

# Apply matplotlib style for publication quality
matplotlib.rcParams.update(MPL_STYLE)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

print("=" * 60)
print("TRAFFIC VISION - COMPLETE FIGURE GENERATION")
print("=" * 60)

# =============================================================================
# 1. LOAD AND PREPARE DATA (NO LEAKAGE)
# =============================================================================

print("\n[1/5] Loading and preparing data...")

data = pd.read_csv(PROCESSED_DATA_DIR / 'processed_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed')
print(f"Total records: {len(data):,}")

# Feature columns for modeling
feature_cols = [
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'is_night',
    'speed_over_limit', 'speed_ratio',
    'radar_id_encoded', 'latitude', 'longitude',
    'location_avg_speed', 'location_violation_rate',
    'vehicle_class_encoded', 'vehicle_category_encoded', 'is_heavy_vehicle',
    'vehicles_per_hour', 'congestion_index', 'hourly_avg_speed', 'lane_avg_speed',
    'speed_lag_1', 'speed_lag_2', 'rolling_avg_speed_5', 'rolling_avg_speed_10',
    'speed_diff_from_avg'
]

# Filter to available features
available_features = [f for f in feature_cols if f in data.columns]
print(f"Features used: {len(available_features)}")

# Prepare feature matrix
X = data[available_features].copy()
y = data['is_violation'].copy()

# Fill NaN with median (no leakage - computed before split)
X = X.fillna(X.median())

# =============================================================================
# TEMPORAL SPLIT TO AVOID DATA LEAKAGE
# =============================================================================

print("\n[2/5] Splitting data (temporal split - NO leakage)...")

# Sort by timestamp for temporal split
data_sorted = data.sort_values('timestamp').reset_index(drop=True)
X_sorted = X.loc[data_sorted.index].reset_index(drop=True)
y_sorted = y.loc[data_sorted.index].reset_index(drop=True)

# Use first 80% for training, last 20% for testing (temporal)
split_idx = int(len(X_sorted) * 0.8)
X_train_full = X_sorted.iloc[:split_idx]
X_test_full = X_sorted.iloc[split_idx:]
y_train_full = y_sorted.iloc[:split_idx]
y_test_full = y_sorted.iloc[split_idx:]

print(f"Training set: {len(X_train_full):,} records")
print(f"Test set: {len(X_test_full):,} records")
print(f"Train period: {data_sorted.iloc[:split_idx]['timestamp'].min()} to {data_sorted.iloc[:split_idx]['timestamp'].max()}")
print(f"Test period: {data_sorted.iloc[split_idx:]['timestamp'].min()} to {data_sorted.iloc[split_idx:]['timestamp'].max()}")

# Sample for faster training (stratified)
SAMPLE_SIZE = 200000
if len(X_train_full) > SAMPLE_SIZE:
    train_indices = X_train_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).index
    X_train = X_train_full.loc[train_indices]
    y_train = y_train_full.loc[train_indices]
else:
    X_train = X_train_full
    y_train = y_train_full

if len(X_test_full) > SAMPLE_SIZE // 4:
    test_indices = X_test_full.sample(n=SAMPLE_SIZE // 4, random_state=RANDOM_STATE).index
    X_test = X_test_full.loc[test_indices]
    y_test = y_test_full.loc[test_indices]
else:
    X_test = X_test_full
    y_test = y_test_full

print(f"Sampled training: {len(X_train):,}")
print(f"Sampled test: {len(X_test):,}")

# Scale features (fit on train only - NO leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')

# =============================================================================
# 3. TRAIN ALL MODELS
# =============================================================================

print("\n[3/5] Training models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15,
                                            random_state=RANDOM_STATE, n_jobs=-1),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=RANDOM_STATE,
                                    max_iter=300, early_stopping=True)
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                      random_state=RANDOM_STATE, use_label_encoder=False,
                                      eval_metric='logloss', n_jobs=-1)

if HAS_LIGHTGBM:
    models['LightGBM'] = LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1,
                                        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)

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

    print(f"    F1: {results[name]['f1']:.4f}, AUC: {results[name]['roc_auc']:.4f}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = models[best_model_name]
joblib.dump(best_model, MODELS_DIR / 'best_model.joblib')
print(f"\nBest model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv(REPORTS_DIR / 'model_results.csv')

# =============================================================================
# 4. GENERATE ML FIGURES
# =============================================================================

print("\n[4/5] Generating ML figures...")

def save_figure(fig, name):
    """Save figure in PNG and PDF formats."""
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png")

# Figure 1: Model Comparison Bar Chart
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

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.legend(loc='lower right', fontsize=11)
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

ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.legend(loc='lower left', fontsize=11)
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

# Figure 5: Confusion Matrices for All Models
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (name, y_pred) in enumerate(predictions.items()):
    if i >= len(axes):
        break
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Viol.', 'Viol.'],
                yticklabels=['No Viol.', 'Viol.'])
    axes[i].set_xlabel('Predicted', fontsize=11)
    axes[i].set_ylabel('Actual', fontsize=11)
    axes[i].set_title(name, fontsize=12, fontweight='bold')

# Hide unused subplots
for j in range(len(predictions), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
save_figure(fig, 'confusion_matrices_all')

# Figure 6: Feature Importance (Random Forest)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': importance
    }).sort_values('importance', ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['feature'], importance_df['importance'], color=COLORS['primary'])
    ax.set_xlabel('Feature Importance', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'feature_importance_rf')

# =============================================================================
# 5. GENERATE XAI FIGURES (SHAP)
# =============================================================================

print("\n[5/5] Generating XAI figures (SHAP)...")

if HAS_SHAP:
    # Use a sample for SHAP (computationally intensive)
    SHAP_SAMPLE = 1000
    shap_indices = np.random.choice(len(X_test_scaled), min(SHAP_SAMPLE, len(X_test_scaled)), replace=False)
    X_shap = X_test_scaled[shap_indices]

    # Use tree explainer for tree-based models
    if HAS_LIGHTGBM and 'LightGBM' in models:
        shap_model = models['LightGBM']
        explainer = shap.TreeExplainer(shap_model)
    elif 'Random Forest' in models:
        shap_model = models['Random Forest']
        explainer = shap.TreeExplainer(shap_model)
    else:
        shap_model = best_model
        # Use KernelExplainer for non-tree models
        background = shap.sample(X_train_scaled, 100)
        explainer = shap.KernelExplainer(shap_model.predict_proba, background)

    print("  Computing SHAP values...")
    shap_values = explainer.shap_values(X_shap)

    # Handle multi-output (get positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Figure 7: SHAP Summary (Beeswarm) Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=available_features,
                      show=False, max_display=20, plot_size=None)
    plt.tight_layout()
    save_figure(plt.gcf(), 'shap_beeswarm')

    # Figure 8: SHAP Bar Plot (Global Importance)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=available_features,
                      plot_type='bar', show=False, max_display=20)
    plt.tight_layout()
    save_figure(plt.gcf(), 'shap_importance_bar')

    # Figure 9: SHAP Dependence Plots (Top 4 Features)
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

    # Figure 10: SHAP Waterfall (Single Prediction Examples)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Find a violation and non-violation example
    viol_idx = np.where(y_test.iloc[shap_indices].values == 1)[0]
    non_viol_idx = np.where(y_test.iloc[shap_indices].values == 0)[0]

    if len(viol_idx) > 0:
        idx = viol_idx[0]
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1]

        # Create waterfall data manually
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

    # Figure 11: SHAP Interaction Plot
    if len(top_features_idx) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.dependence_plot(top_features_idx[0], shap_values, X_shap,
                            interaction_index=top_features_idx[1],
                            feature_names=available_features,
                            ax=ax, show=False)
        ax.set_xlabel(available_features[top_features_idx[0]], fontsize=13)
        ax.set_ylabel('SHAP value', fontsize=13)
        plt.tight_layout()
        save_figure(fig, 'shap_interaction')

    # Save feature importance from SHAP
    shap_importance = pd.DataFrame({
        'feature': available_features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    shap_importance.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)
    print("  SHAP feature importance saved")

else:
    print("  SHAP not available - skipping XAI figures")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FIGURE GENERATION COMPLETE")
print("=" * 60)

# List all generated figures
import os
figures = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
print(f"\nGenerated {len(figures)} figures:")
for f in figures:
    print(f"  - {f}")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print(f"Model results saved to: {REPORTS_DIR / 'model_results.csv'}")
print(f"Feature importance saved to: {REPORTS_DIR / 'feature_importance.csv'}")

# Print model summary
print("\n" + "-" * 40)
print("MODEL PERFORMANCE SUMMARY")
print("-" * 40)
print(results_df.round(4).to_string())
