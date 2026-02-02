"""
Traffic Vision Analysis Pipeline
================================
Complete ML + XAI + LLM pipeline for traffic speed violation analysis.
Designed for Q1 journal publication quality research.

Author: Mahbub Hassan
        Graduate Student & Non-ASEAN Scholar
        Department of Civil Engineering, Faculty of Engineering
        Chulalongkorn University, Bangkok, Thailand
        Email: 6870376421@student.chula.ac.th
        GitHub: https://github.com/mahbubchula

Version: 1.0.0
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, PROCESSED_DATA_DIR,
    RAW_DATA_FOLDERS, CSV_COLUMNS, VEHICLE_CLASSES, LANE_POSITIONS,
    RADAR_LOCATIONS, SPEED_LIMIT_DEFAULT, SPEED_MIN_VALID, SPEED_MAX_VALID,
    RUSH_HOUR_MORNING, RUSH_HOUR_EVENING, WEEKEND_DAYS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, HYPERPARAMETER_GRIDS,
    COLORS, COLOR_PALETTE, FIGURE_SETTINGS, MPL_STYLE, SHAP_SETTINGS,
    GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
    PROMPT_TEMPLATES, LOG_SETTINGS,
    get_vehicle_name, get_vehicle_category, get_lane_name, get_radar_location,
    is_violation, get_violation_severity
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_SETTINGS["level"]),
    format=LOG_SETTINGS["format"],
    datefmt=LOG_SETTINGS["date_format"],
    handlers=[
        logging.FileHandler(LOG_SETTINGS["log_file"]),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Apply matplotlib style
matplotlib.rcParams.update(MPL_STYLE)

# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

class DataProcessor:
    """Handles data loading, cleaning, and feature engineering."""

    def __init__(self, data_dir: Path = DATA_DIR):
        """Initialize data processor.

        Args:
            data_dir: Directory containing raw data folders
        """
        self.data_dir = data_dir
        self.cleaning_stats: Dict[str, Any] = {}
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

    def load_all_data(self) -> pd.DataFrame:
        """Load all CSV files from data folders.

        Returns:
            Combined DataFrame with all data
        """
        logger.info("Loading data from all folders...")
        all_data = []

        for folder in RAW_DATA_FOLDERS:
            folder_path = self.data_dir / folder
            if not folder_path.exists():
                logger.warning(f"Folder not found: {folder_path}")
                continue

            csv_files = list(folder_path.glob("*.csv"))
            logger.info(f"Found {len(csv_files)} CSV files in {folder}")

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['source_folder'] = folder
                    df['source_file'] = csv_file.name
                    all_data.append(df)
                    logger.debug(f"Loaded {len(df)} rows from {csv_file.name}")
                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")

        if not all_data:
            raise ValueError("No data files found!")

        self.raw_data = pd.concat(all_data, ignore_index=True)
        self.cleaning_stats['original_rows'] = len(self.raw_data)
        logger.info(f"Total rows loaded: {len(self.raw_data):,}")

        return self.raw_data

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean and validate data.

        Args:
            df: DataFrame to clean (uses raw_data if None)

        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()

        logger.info("Cleaning data...")
        initial_rows = len(df)

        # Rename columns to standard names
        column_mapping = {v: k for k, v in CSV_COLUMNS.items()}
        df = df.rename(columns=column_mapping)

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        invalid_timestamps = df['timestamp'].isna().sum()
        df = df.dropna(subset=['timestamp'])
        logger.info(f"Removed {invalid_timestamps} rows with invalid timestamps")

        # Filter unrealistic speeds
        speed_before = len(df)
        df = df[(df['speed'] >= SPEED_MIN_VALID) & (df['speed'] <= SPEED_MAX_VALID)]
        speed_filtered = speed_before - len(df)
        logger.info(f"Removed {speed_filtered} rows with unrealistic speeds")

        # Filter invalid vehicle classes
        valid_classes = list(VEHICLE_CLASSES.keys())
        class_before = len(df)
        df = df[df['vehicle_class'].isin(valid_classes)]
        class_filtered = class_before - len(df)
        logger.info(f"Removed {class_filtered} rows with invalid vehicle classes")

        # Remove duplicates
        dup_before = len(df)
        df = df.drop_duplicates(subset=['device_id', 'timestamp', 'lane', 'speed'])
        duplicates_removed = dup_before - len(df)
        logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Record cleaning statistics
        final_rows = len(df)
        self.cleaning_stats.update({
            'cleaned_rows': final_rows,
            'invalid_timestamps': invalid_timestamps,
            'speed_outliers': speed_filtered,
            'invalid_classes': class_filtered,
            'duplicates': duplicates_removed,
            'total_removed': initial_rows - final_rows,
            'removal_percentage': (initial_rows - final_rows) / initial_rows * 100
        })

        logger.info(f"Cleaning complete: {final_rows:,} rows remaining "
                   f"({self.cleaning_stats['removal_percentage']:.2f}% removed)")

        self.cleaned_data = df
        return df

    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create features for machine learning.

        Args:
            df: DataFrame to process (uses cleaned_data if None)

        Returns:
            DataFrame with engineered features
        """
        if df is None:
            df = self.cleaned_data.copy()
        else:
            df = df.copy()

        logger.info("Engineering features...")

        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['day_of_week'].isin(WEEKEND_DAYS).astype(int)
        df['is_rush_hour'] = (
            ((df['hour'] >= RUSH_HOUR_MORNING[0]) & (df['hour'] <= RUSH_HOUR_MORNING[1])) |
            ((df['hour'] >= RUSH_HOUR_EVENING[0]) & (df['hour'] <= RUSH_HOUR_EVENING[1]))
        ).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

        # Speed features
        df['is_violation'] = (df['speed'] > SPEED_LIMIT_DEFAULT).astype(int)
        df['speed_over_limit'] = np.maximum(0, df['speed'] - SPEED_LIMIT_DEFAULT)
        df['violation_severity'] = df['speed'].apply(
            lambda x: get_violation_severity(x, SPEED_LIMIT_DEFAULT)
        )
        df['speed_ratio'] = df['speed'] / SPEED_LIMIT_DEFAULT
        df['speed_category'] = pd.cut(
            df['speed'],
            bins=[0, 40, 60, 80, 100, 200],
            labels=['Very Slow', 'Normal', 'Fast', 'Very Fast', 'Extreme']
        )

        # Location features
        df['latitude'] = df['device_id'].apply(lambda x: get_radar_location(x)[0])
        df['longitude'] = df['device_id'].apply(lambda x: get_radar_location(x)[1])
        df['location_name'] = df['device_id'].apply(lambda x: get_radar_location(x)[2])

        # Encode radar ID
        le_radar = LabelEncoder()
        df['radar_id_encoded'] = le_radar.fit_transform(df['device_id'])

        # Location-based aggregations
        location_stats = df.groupby('device_id').agg({
            'speed': ['mean', 'std'],
            'is_violation': 'mean'
        }).reset_index()
        location_stats.columns = ['device_id', 'location_avg_speed',
                                  'location_speed_std', 'location_violation_rate']
        df = df.merge(location_stats, on='device_id', how='left')

        # Vehicle features
        df['vehicle_name'] = df['vehicle_class'].apply(get_vehicle_name)
        df['vehicle_category'] = df['vehicle_class'].apply(get_vehicle_category)
        df['is_heavy_vehicle'] = (df['vehicle_category'] == 'heavy').astype(int)

        # Encode vehicle class
        le_vehicle = LabelEncoder()
        df['vehicle_class_encoded'] = le_vehicle.fit_transform(df['vehicle_class'])
        le_category = LabelEncoder()
        df['vehicle_category_encoded'] = le_category.fit_transform(df['vehicle_category'])

        # Lane features
        df['lane_name'] = df['lane'].apply(get_lane_name)

        # Lane-based aggregations
        lane_stats = df.groupby(['device_id', 'lane']).agg({
            'speed': 'mean'
        }).reset_index()
        lane_stats.columns = ['device_id', 'lane', 'lane_avg_speed']
        df = df.merge(lane_stats, on=['device_id', 'lane'], how='left')

        # Traffic flow features (hourly aggregations)
        df['_date'] = df['timestamp'].dt.date
        df['_hour'] = df['timestamp'].dt.hour

        hourly_stats = df.groupby(['_date', '_hour', 'device_id']).agg(
            vehicles_per_hour=('speed', 'count'),
            hourly_avg_speed=('speed', 'mean')
        ).reset_index()
        hourly_stats.columns = ['_date', '_hour', 'device_id',
                               'vehicles_per_hour', 'hourly_avg_speed']

        # Merge back (approximate - for same hour)
        df = df.merge(hourly_stats, on=['_date', '_hour', 'device_id'], how='left')
        df = df.drop(columns=['_date', '_hour'])

        # Congestion index (normalized vehicle count)
        max_vehicles = df['vehicles_per_hour'].max()
        df['congestion_index'] = df['vehicles_per_hour'] / max_vehicles if max_vehicles > 0 else 0

        # Lag features (sorted by device and time)
        df = df.sort_values(['device_id', 'timestamp'])
        for lag in [1, 2, 3]:
            df[f'speed_lag_{lag}'] = df.groupby('device_id')['speed'].shift(lag)

        # Rolling average speeds
        df['rolling_avg_speed_5'] = df.groupby('device_id')['speed'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['rolling_avg_speed_10'] = df.groupby('device_id')['speed'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

        # Speed difference from average
        df['speed_diff_from_avg'] = df['speed'] - df['location_avg_speed']

        # Fill NaN values in lag features
        lag_cols = [c for c in df.columns if 'lag' in c or 'rolling' in c]
        for col in lag_cols:
            df[col] = df[col].fillna(df['speed'])

        # Drop temporary columns
        df = df.drop(columns=['date'], errors='ignore')

        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")

        self.processed_data = df
        return df

    def save_processed_data(self, filename: str = "processed_data.csv"):
        """Save processed data to CSV.

        Args:
            filename: Output filename
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save. Run engineer_features first.")

        output_path = PROCESSED_DATA_DIR / filename
        self.processed_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        # Save cleaning statistics
        stats_path = PROCESSED_DATA_DIR / "cleaning_stats.txt"
        with open(stats_path, 'w') as f:
            f.write("Data Cleaning Statistics\n")
            f.write("=" * 40 + "\n")
            for key, value in self.cleaning_stats.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Cleaning statistics saved to {stats_path}")

    def get_feature_matrix(self, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Get feature matrix and target for modeling.

        Args:
            features: List of feature column names

        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Run engineer_features first.")

        # Filter to available features
        available_features = [f for f in features if f in self.processed_data.columns]
        missing = set(features) - set(available_features)
        if missing:
            logger.warning(f"Missing features: {missing}")

        X = self.processed_data[available_features].copy()
        y = self.processed_data['is_violation'].copy()

        # Handle any remaining NaN values
        X = X.fillna(X.median())

        return X, y


# =============================================================================
# MACHINE LEARNING MODELS
# =============================================================================

class ViolationPredictor:
    """Machine learning models for violation prediction."""

    def __init__(self):
        """Initialize predictor with models."""
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.scaler = StandardScaler()
        self.results: Dict[str, Dict] = {}
        self.feature_names: List[str] = []

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split data for training.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.feature_names = list(X.columns)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all classification models.

        Returns:
            Dictionary of model name -> model object
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            'XGBoost': None,  # Will initialize if available
            'LightGBM': None,  # Will initialize if available
            'Neural Network': MLPClassifier(
                random_state=RANDOM_STATE, max_iter=500, early_stopping=True
            )
        }

        # Try importing XGBoost
        try:
            from xgboost import XGBClassifier
            self.models['XGBoost'] = XGBClassifier(
                random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric='logloss', n_jobs=-1
            )
        except ImportError:
            logger.warning("XGBoost not installed. Skipping.")
            del self.models['XGBoost']

        # Try importing LightGBM
        try:
            from lightgbm import LGBMClassifier
            self.models['LightGBM'] = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )
        except ImportError:
            logger.warning("LightGBM not installed. Skipping.")
            del self.models['LightGBM']

        return self.models

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Dict]:
        """Train all models and evaluate performance.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Dictionary of model results
        """
        logger.info("Training and evaluating models...")

        for name, model in self.models.items():
            if model is None:
                continue

            logger.info(f"Training {name}...")

            try:
                # Hyperparameter tuning
                if tune_hyperparameters and name.lower().replace(' ', '_') in HYPERPARAMETER_GRIDS:
                    grid_key = name.lower().replace(' ', '_')
                    grid = HYPERPARAMETER_GRIDS[grid_key]

                    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                        random_state=RANDOM_STATE)

                    grid_search = GridSearchCV(
                        model, grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    logger.info(f"Best params for {name}: {best_params}")
                else:
                    model.fit(X_train, y_train)
                    best_params = {}

                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=CV_FOLDS, scoring='f1'
                )

                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'best_params': best_params
                }

                self.results[name] = metrics
                self.models[name] = model

                auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A'
                logger.info(f"{name} - F1: {metrics['f1']:.4f}, AUC: {auc_str}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                self.results[name] = {'error': str(e)}

        # Select best model based on F1 score
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if valid_results:
            self.best_model_name = max(valid_results, key=lambda x: valid_results[x]['f1'])
            self.best_model = self.models[self.best_model_name]
            logger.info(f"Best model: {self.best_model_name} "
                       f"(F1: {self.results[self.best_model_name]['f1']:.4f})")

        return self.results

    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.models.items():
            if model is None:
                continue
            safe_name = name.lower().replace(' ', '_')
            model_path = MODELS_DIR / f"{safe_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")

        # Save scaler
        scaler_path = MODELS_DIR / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)

        # Save best model separately
        if self.best_model is not None:
            best_path = MODELS_DIR / "best_model.joblib"
            joblib.dump(self.best_model, best_path)

            # Save model info
            info_path = MODELS_DIR / "best_model_info.txt"
            with open(info_path, 'w') as f:
                f.write(f"Best Model: {self.best_model_name}\n")
                f.write(f"Metrics:\n")
                for k, v in self.results[self.best_model_name].items():
                    f.write(f"  {k}: {v}\n")

    def load_model(self, model_name: str = "best") -> Any:
        """Load a saved model.

        Args:
            model_name: Name of model to load or 'best'

        Returns:
            Loaded model
        """
        if model_name == "best":
            model_path = MODELS_DIR / "best_model.joblib"
        else:
            safe_name = model_name.lower().replace(' ', '_')
            model_path = MODELS_DIR / f"{safe_name}_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        self.scaler = joblib.load(MODELS_DIR / "scaler.joblib")

        return model

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        model: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data.

        Args:
            X: Feature matrix
            model: Model to use (uses best_model if None)

        Returns:
            Tuple of (predictions, probabilities)
        """
        if model is None:
            model = self.best_model

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities


# =============================================================================
# EXPLAINABILITY (SHAP + LIME)
# =============================================================================

class ModelExplainer:
    """Model explainability using SHAP and rule extraction."""

    def __init__(self, model: Any, feature_names: List[str]):
        """Initialize explainer.

        Args:
            model: Trained model to explain
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None

    def compute_shap_values(
        self,
        X: np.ndarray,
        sample_size: int = SHAP_SETTINGS['sample_size']
    ) -> np.ndarray:
        """Compute SHAP values for model explanation.

        Args:
            X: Feature matrix
            sample_size: Number of samples for SHAP calculation

        Returns:
            SHAP values array
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed. Install with: pip install shap")
            return None

        logger.info("Computing SHAP values...")

        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Create appropriate explainer based on model type
        model_type = type(self.model).__name__

        if 'XGB' in model_type or 'LGBM' in model_type or 'Forest' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)

            # Handle multi-output format
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Positive class
        else:
            # Use KernelExplainer for other models
            background = shap.sample(X_sample, min(100, len(X_sample)))
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
            self.shap_values = self.explainer.shap_values(X_sample[:100])
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]

        logger.info(f"SHAP values computed. Shape: {self.shap_values.shape}")

        # Store X_sample for plotting
        self.X_sample = X_sample

        return self.shap_values

    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values.

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values first.")

        importance = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_shap_summary(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Create SHAP summary plot (beeswarm).

        Args:
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed")
            return None

        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size'])

        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=SHAP_SETTINGS['max_display'],
            plot_size=None
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=FIGURE_SETTINGS['dpi'], bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")

        return fig

    def plot_feature_dependence(
        self,
        feature: str,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create SHAP dependence plot for a feature.

        Args:
            feature: Feature name
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        try:
            import shap
        except ImportError:
            return None

        if feature not in self.feature_names:
            logger.warning(f"Feature {feature} not found")
            return None

        feature_idx = self.feature_names.index(feature)

        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size_small'])

        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            show=False,
            ax=ax
        )

        ax.set_xlabel(feature, fontsize=FIGURE_SETTINGS['font_size_labels'])
        ax.set_ylabel('SHAP value', fontsize=FIGURE_SETTINGS['font_size_labels'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=FIGURE_SETTINGS['dpi'], bbox_inches='tight')
            logger.info(f"Dependence plot saved to {save_path}")

        return fig

    def explain_prediction(
        self,
        X_single: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, Dict]:
        """Explain a single prediction with waterfall plot.

        Args:
            X_single: Single sample to explain
            save_path: Path to save figure

        Returns:
            Tuple of (figure, contribution dict)
        """
        try:
            import shap
        except ImportError:
            return None, {}

        # Compute SHAP values for single prediction
        if 'Tree' in type(self.explainer).__name__:
            sv = self.explainer.shap_values(X_single.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1][0]
            else:
                sv = sv[0]
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1]
        else:
            sv = self.explainer.shap_values(X_single.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1][0]
            else:
                sv = sv[0]
            expected_value = self.explainer.expected_value[1]

        # Create contribution dictionary
        contributions = dict(zip(self.feature_names, sv))

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size'])

        # Sort by absolute contribution
        sorted_idx = np.argsort(np.abs(sv))[::-1][:15]

        features_plot = [self.feature_names[i] for i in sorted_idx]
        values_plot = [sv[i] for i in sorted_idx]

        colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in values_plot]

        ax.barh(range(len(features_plot)), values_plot, color=colors)
        ax.set_yticks(range(len(features_plot)))
        ax.set_yticklabels(features_plot)
        ax.set_xlabel('SHAP value (impact on prediction)',
                     fontsize=FIGURE_SETTINGS['font_size_labels'])
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=FIGURE_SETTINGS['dpi'], bbox_inches='tight')

        return fig, contributions

    def extract_rules(self, max_rules: int = 10) -> List[str]:
        """Extract human-readable rules from tree-based models.

        Args:
            max_rules: Maximum number of rules to extract

        Returns:
            List of rule strings
        """
        rules = []
        model_type = type(self.model).__name__

        if 'Forest' in model_type or 'XGB' in model_type:
            # Get feature importance and top features
            importance_df = self.get_feature_importance()
            top_features = importance_df.head(5)['feature'].tolist()

            # Generate rule templates based on feature importance
            for feat in top_features:
                idx = self.feature_names.index(feat) if feat in self.feature_names else -1
                if idx >= 0 and self.shap_values is not None:
                    mean_shap = self.shap_values[:, idx].mean()
                    direction = "increases" if mean_shap > 0 else "decreases"
                    rules.append(f"Higher {feat} {direction} violation probability")

        return rules[:max_rules]


# =============================================================================
# LLM INTEGRATION (Groq)
# =============================================================================

class LLMAssistant:
    """LLM-powered assistant using Groq API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM assistant.

        Args:
            api_key: Groq API key (uses environment variable if None)
        """
        self.api_key = api_key or GROQ_API_KEY
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Groq client."""
        if not self.api_key:
            logger.warning("Groq API key not set. LLM features disabled.")
            return

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
        except ImportError:
            logger.error("Groq library not installed. Install with: pip install groq")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")

    def _call_llm(self, prompt: str, max_tokens: int = GROQ_MAX_TOKENS) -> str:
        """Call Groq LLM API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum response tokens

        Returns:
            Generated response text
        """
        if self.client is None:
            return "LLM not available. Please set GROQ_API_KEY."

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=GROQ_TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return f"Error generating response: {e}"

    def generate_weekly_report(
        self,
        data: pd.DataFrame,
        insights: Dict[str, Any]
    ) -> str:
        """Generate a weekly traffic report.

        Args:
            data: Processed traffic data
            insights: Dictionary of key insights

        Returns:
            Generated report text
        """
        # Prepare report data
        date_range = f"{data['timestamp'].min().date()} to {data['timestamp'].max().date()}"
        total_vehicles = len(data)
        total_violations = data['is_violation'].sum()
        violation_rate = (total_violations / total_vehicles) * 100
        avg_speed = data['speed'].mean()
        max_speed = data['speed'].max()

        # Find peak violation hour
        hourly_violations = data[data['is_violation'] == 1].groupby('hour').size()
        peak_hour = hourly_violations.idxmax() if len(hourly_violations) > 0 else "N/A"

        # Find top violation location
        location_violations = data[data['is_violation'] == 1].groupby('location_name').size()
        top_location = location_violations.idxmax() if len(location_violations) > 0 else "N/A"

        # Format key findings
        key_findings = "\n".join([
            f"- {k}: {v}" for k, v in insights.items()
        ]) if insights else "No specific findings available."

        prompt = PROMPT_TEMPLATES['weekly_report'].format(
            date_range=date_range,
            total_vehicles=total_vehicles,
            total_violations=total_violations,
            violation_rate=violation_rate,
            avg_speed=avg_speed,
            max_speed=max_speed,
            peak_violation_hour=f"{peak_hour}:00" if peak_hour != "N/A" else "N/A",
            top_violation_location=top_location,
            key_findings=key_findings
        )

        return self._call_llm(prompt)

    def explain_prediction(
        self,
        prediction: int,
        confidence: float,
        feature_values: Dict[str, float],
        shap_contributions: Dict[str, float]
    ) -> str:
        """Generate natural language explanation for a prediction.

        Args:
            prediction: Predicted class (0 or 1)
            confidence: Prediction confidence
            feature_values: Dictionary of feature values
            shap_contributions: SHAP contribution values

        Returns:
            Explanation text
        """
        pred_text = "Violation" if prediction == 1 else "No Violation"

        # Format feature values
        feature_str = "\n".join([
            f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}"
            for k, v in list(feature_values.items())[:10]
        ])

        # Format SHAP contributions (top 5)
        sorted_shap = sorted(shap_contributions.items(),
                            key=lambda x: abs(x[1]), reverse=True)[:5]
        shap_str = "\n".join([
            f"- {k}: {v:+.4f} ({'increases' if v > 0 else 'decreases'} violation probability)"
            for k, v in sorted_shap
        ])

        prompt = PROMPT_TEMPLATES['prediction_explanation'].format(
            prediction=pred_text,
            confidence=confidence * 100,
            feature_values=feature_str,
            shap_contributions=shap_str
        )

        return self._call_llm(prompt)

    def generate_location_recommendations(
        self,
        location_name: str,
        location_data: pd.DataFrame
    ) -> str:
        """Generate recommendations for a specific location.

        Args:
            location_name: Name of the location
            location_data: Traffic data for the location

        Returns:
            Recommendations text
        """
        date_range = f"{location_data['timestamp'].min().date()} to {location_data['timestamp'].max().date()}"
        avg_speed = location_data['speed'].mean()
        violation_rate = location_data['is_violation'].mean() * 100

        # Peak hours
        hourly_violations = location_data[location_data['is_violation'] == 1].groupby('hour').size()
        top_hours = hourly_violations.nlargest(3).index.tolist()
        peak_hours = ", ".join([f"{h}:00" for h in top_hours]) if top_hours else "N/A"

        # Vehicle types
        vehicle_violations = location_data[location_data['is_violation'] == 1].groupby('vehicle_name').size()
        top_vehicles = vehicle_violations.nlargest(2).index.tolist()
        vehicle_types = ", ".join(top_vehicles) if top_vehicles else "All types"

        # Historical patterns
        daily_violations = location_data.groupby(location_data['timestamp'].dt.dayofweek)['is_violation'].mean()
        weekend_avg = daily_violations.loc[[5, 6]].mean() if 5 in daily_violations.index else 0
        weekday_avg = daily_violations.loc[daily_violations.index < 5].mean()

        patterns = f"Weekend violation rate: {weekend_avg*100:.1f}%\nWeekday violation rate: {weekday_avg*100:.1f}%"

        prompt = PROMPT_TEMPLATES['location_recommendation'].format(
            location_name=location_name,
            date_range=date_range,
            avg_speed=avg_speed,
            violation_rate=violation_rate,
            peak_hours=peak_hours,
            vehicle_types=vehicle_types,
            patterns=patterns
        )

        return self._call_llm(prompt)

    def answer_query(self, question: str, context: str) -> str:
        """Answer a natural language query about the data.

        Args:
            question: User's question
            context: Data context to use for answering

        Returns:
            Answer text
        """
        prompt = PROMPT_TEMPLATES['query_response'].format(
            question=question,
            context=context
        )

        return self._call_llm(prompt)


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Publication-quality visualizations."""

    def __init__(self):
        """Initialize visualizer with style settings."""
        matplotlib.rcParams.update(MPL_STYLE)
        self.colors = COLOR_PALETTE

    def _save_figure(self, fig: plt.Figure, filename: str):
        """Save figure in publication format.

        Args:
            fig: Matplotlib figure
            filename: Output filename (without extension)
        """
        # Save as PNG
        png_path = FIGURES_DIR / f"{filename}.png"
        fig.savefig(png_path, dpi=FIGURE_SETTINGS['dpi'],
                   bbox_inches='tight', facecolor='white')

        # Also save as PDF for vector format
        pdf_path = FIGURES_DIR / f"{filename}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

        logger.info(f"Figure saved: {png_path}")

    def plot_temporal_patterns(self, data: pd.DataFrame) -> plt.Figure:
        """Create temporal pattern visualization.

        Args:
            data: Processed traffic data

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SETTINGS['figure_size_large'])

        # Hourly speed pattern
        hourly = data.groupby('hour').agg({
            'speed': 'mean',
            'is_violation': 'mean'
        }).reset_index()

        axes[0, 0].plot(hourly['hour'], hourly['speed'],
                       color=self.colors[0], linewidth=2, marker='o')
        axes[0, 0].axhline(y=SPEED_LIMIT_DEFAULT, color=self.colors[4],
                          linestyle='--', label='Speed Limit')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Speed (km/h)')
        axes[0, 0].legend()

        # Daily pattern
        daily = data.groupby('day_of_week').agg({
            'speed': 'mean',
            'is_violation': 'mean'
        }).reset_index()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        axes[0, 1].bar(daily['day_of_week'], daily['is_violation'] * 100,
                      color=self.colors[1])
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(days)
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Violation Rate (%)')

        # Monthly pattern
        monthly = data.groupby('month').agg({
            'speed': 'mean',
            'is_violation': 'sum'
        }).reset_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        ax2 = axes[1, 0]
        x = monthly['month'] - 1
        ax2.bar(x, monthly['is_violation'], color=self.colors[2])
        ax2.set_xticks(range(12))
        ax2.set_xticklabels([months[i] for i in range(12)], rotation=45)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Total Violations')

        # Hour vs Day heatmap
        pivot = data.pivot_table(
            values='is_violation',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        )

        sns.heatmap(pivot, ax=axes[1, 1], cmap='YlOrRd',
                   cbar_kws={'label': 'Violation Rate'})
        axes[1, 1].set_yticklabels(days, rotation=0)
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Day of Week')

        plt.tight_layout()
        self._save_figure(fig, 'temporal_patterns')

        return fig

    def plot_vehicle_distribution(self, data: pd.DataFrame) -> plt.Figure:
        """Create vehicle type distribution visualization.

        Args:
            data: Processed traffic data

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SETTINGS['figure_size'])

        # Vehicle type counts
        vehicle_counts = data['vehicle_name'].value_counts()

        axes[0].barh(vehicle_counts.index, vehicle_counts.values,
                    color=self.colors[:len(vehicle_counts)])
        axes[0].set_xlabel('Count')
        axes[0].set_ylabel('Vehicle Type')

        # Violation rate by vehicle type
        vehicle_violations = data.groupby('vehicle_name')['is_violation'].mean() * 100

        axes[1].barh(vehicle_violations.index, vehicle_violations.values,
                    color=self.colors[:len(vehicle_violations)])
        axes[1].set_xlabel('Violation Rate (%)')
        axes[1].set_ylabel('Vehicle Type')

        plt.tight_layout()
        self._save_figure(fig, 'vehicle_distribution')

        return fig

    def plot_speed_distribution(self, data: pd.DataFrame) -> plt.Figure:
        """Create speed distribution visualization.

        Args:
            data: Processed traffic data

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SETTINGS['figure_size'])

        # Overall speed distribution
        axes[0].hist(data['speed'], bins=50, color=self.colors[0],
                    edgecolor='white', alpha=0.7)
        axes[0].axvline(x=SPEED_LIMIT_DEFAULT, color=self.colors[4],
                       linestyle='--', linewidth=2, label='Speed Limit')
        axes[0].set_xlabel('Speed (km/h)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()

        # Speed by location (top 5)
        top_locations = data['location_name'].value_counts().head(5).index
        location_data = data[data['location_name'].isin(top_locations)]

        for i, loc in enumerate(top_locations):
            loc_speeds = location_data[location_data['location_name'] == loc]['speed']
            axes[1].hist(loc_speeds, bins=30, alpha=0.6,
                        label=loc[:15], color=self.colors[i])

        axes[1].axvline(x=SPEED_LIMIT_DEFAULT, color=self.colors[4],
                       linestyle='--', linewidth=2)
        axes[1].set_xlabel('Speed (km/h)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, 'speed_distribution')

        return fig

    def plot_violation_heatmap(self, data: pd.DataFrame) -> plt.Figure:
        """Create violation heatmap (hour vs day).

        Args:
            data: Processed traffic data

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size'])

        pivot = data.pivot_table(
            values='is_violation',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        ) * 100

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
               'Friday', 'Saturday', 'Sunday']

        sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=False,
                   cbar_kws={'label': 'Violation Rate (%)'})

        ax.set_yticklabels(days, rotation=0)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')

        plt.tight_layout()
        self._save_figure(fig, 'violation_heatmap')

        return fig

    def plot_model_comparison(self, results: Dict[str, Dict]) -> plt.Figure:
        """Create model performance comparison chart.

        Args:
            results: Dictionary of model results

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size'])

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        models = [m for m in results.keys() if 'error' not in results[m]]

        x = np.arange(len(models))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) or 0 for m in models]
            ax.bar(x + i * width, values, width, label=metric.upper(),
                  color=self.colors[i])

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')

        plt.tight_layout()
        self._save_figure(fig, 'model_comparison')

        return fig

    def plot_roc_curves(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> plt.Figure:
        """Create ROC curves for all models.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size_small'])

        for i, (name, model) in enumerate(models.items()):
            if model is None or not hasattr(model, 'predict_proba'):
                continue

            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)

            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
                   color=self.colors[i], linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        self._save_figure(fig, 'roc_curves')

        return fig

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str = "Best Model"
    ) -> plt.Figure:
        """Create confusion matrix visualization.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model for labeling

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size_small'])

        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Violation', 'Violation'],
                   yticklabels=['No Violation', 'Violation'])

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        plt.tight_layout()
        self._save_figure(fig, 'confusion_matrix')

        return fig

    def plot_study_area_map(self, data: pd.DataFrame) -> plt.Figure:
        """Create study area map with radar locations.

        Args:
            data: Processed traffic data

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SETTINGS['figure_size'])

        # Get unique locations with counts
        location_stats = data.groupby(['device_id', 'latitude', 'longitude', 'location_name']).agg({
            'speed': 'count',
            'is_violation': 'mean'
        }).reset_index()
        location_stats.columns = ['device_id', 'lat', 'lon', 'name', 'count', 'violation_rate']

        # Plot scatter
        scatter = ax.scatter(
            location_stats['lon'],
            location_stats['lat'],
            s=location_stats['count'] / 1000,  # Size by count
            c=location_stats['violation_rate'],  # Color by violation rate
            cmap='YlOrRd',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Violation Rate')

        # Add labels
        for _, row in location_stats.iterrows():
            ax.annotate(
                row['name'][:10],
                (row['lon'], row['lat']),
                fontsize=8,
                ha='center',
                va='bottom'
            )

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.tight_layout()
        self._save_figure(fig, 'study_area_map')

        return fig

    def create_all_figures(
        self,
        data: pd.DataFrame,
        models: Dict[str, Any],
        results: Dict[str, Dict],
        X_test: np.ndarray,
        y_test: pd.Series,
        y_pred: np.ndarray,
        explainer: ModelExplainer
    ):
        """Create all publication figures.

        Args:
            data: Processed data
            models: Trained models
            results: Model results
            X_test: Test features
            y_test: Test labels
            y_pred: Predictions
            explainer: Model explainer
        """
        logger.info("Creating all publication figures...")

        # Basic visualizations
        self.plot_study_area_map(data)
        self.plot_temporal_patterns(data)
        self.plot_vehicle_distribution(data)
        self.plot_speed_distribution(data)
        self.plot_violation_heatmap(data)

        # Model visualizations
        self.plot_model_comparison(results)
        self.plot_roc_curves(models, X_test, y_test)
        self.plot_confusion_matrix(y_test, y_pred)

        # SHAP visualizations
        if explainer.shap_values is not None:
            explainer.plot_shap_summary(FIGURES_DIR / 'shap_importance.png')

            # Plot top 3 feature dependence
            importance_df = explainer.get_feature_importance()
            top_features = importance_df.head(3)['feature'].tolist()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, feat in enumerate(top_features):
                if feat in explainer.feature_names:
                    feat_idx = explainer.feature_names.index(feat)
                    axes[i].scatter(
                        explainer.X_sample[:, feat_idx],
                        explainer.shap_values[:, feat_idx],
                        alpha=0.3,
                        color=self.colors[i]
                    )
                    axes[i].set_xlabel(feat)
                    axes[i].set_ylabel('SHAP value')

            plt.tight_layout()
            fig.savefig(FIGURES_DIR / 'shap_dependence.png',
                       dpi=FIGURE_SETTINGS['dpi'], bbox_inches='tight')

        logger.info(f"All figures saved to {FIGURES_DIR}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(tune_hyperparameters: bool = False):
    """Run the complete analysis pipeline.

    Args:
        tune_hyperparameters: Whether to perform hyperparameter tuning

    Returns:
        Dictionary containing all results and objects
    """
    logger.info("=" * 60)
    logger.info("TRAFFIC VISION ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # 1. Data Processing
    logger.info("\n[1/6] DATA PROCESSING")
    processor = DataProcessor()
    processor.load_all_data()
    processor.clean_data()
    processor.engineer_features()
    processor.save_processed_data()

    # 2. Prepare features for modeling
    logger.info("\n[2/6] FEATURE PREPARATION")
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

    X, y = processor.get_feature_matrix(feature_cols)
    logger.info(f"Feature matrix shape: {X.shape}")

    # 3. Train Models
    logger.info("\n[3/6] MODEL TRAINING")
    predictor = ViolationPredictor()
    predictor.initialize_models()
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    results = predictor.train_and_evaluate(
        X_train, X_test, y_train, y_test,
        tune_hyperparameters=tune_hyperparameters
    )
    predictor.save_models()

    # Get predictions from best model
    y_pred, y_prob = predictor.predict(X.iloc[y_test.index], predictor.best_model)

    # 4. Model Explanation
    logger.info("\n[4/6] MODEL EXPLANATION (SHAP)")
    explainer = ModelExplainer(predictor.best_model, predictor.feature_names)
    explainer.compute_shap_values(X_test)

    # Save feature importance
    importance_df = explainer.get_feature_importance()
    importance_df.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)

    # Extract rules
    rules = explainer.extract_rules()
    with open(REPORTS_DIR / 'model_rules.txt', 'w') as f:
        f.write("Extracted Rules from Best Model\n")
        f.write("=" * 40 + "\n")
        for i, rule in enumerate(rules, 1):
            f.write(f"{i}. {rule}\n")

    # 5. Visualizations
    logger.info("\n[5/6] CREATING VISUALIZATIONS")
    visualizer = Visualizer()
    visualizer.create_all_figures(
        processor.processed_data,
        predictor.models,
        results,
        X_test,
        y_test,
        y_pred,
        explainer
    )

    # 6. Save results
    logger.info("\n[6/6] SAVING RESULTS")

    # Model comparison table
    results_df = pd.DataFrame(results).T
    results_df.to_csv(REPORTS_DIR / 'model_results.csv')

    # Statistics summary
    stats_summary = {
        'Total Records': len(processor.processed_data),
        'Total Violations': processor.processed_data['is_violation'].sum(),
        'Violation Rate': processor.processed_data['is_violation'].mean() * 100,
        'Average Speed': processor.processed_data['speed'].mean(),
        'Max Speed': processor.processed_data['speed'].max(),
        'Unique Locations': processor.processed_data['device_id'].nunique(),
        'Date Range': f"{processor.processed_data['timestamp'].min()} to {processor.processed_data['timestamp'].max()}",
        'Best Model': predictor.best_model_name,
        'Best F1 Score': results[predictor.best_model_name]['f1']
    }

    with open(REPORTS_DIR / 'statistics_summary.txt', 'w') as f:
        f.write("Traffic Violation Analysis - Statistical Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats_summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed data: {PROCESSED_DATA_DIR}")
    logger.info(f"Models: {MODELS_DIR}")
    logger.info(f"Figures: {FIGURES_DIR}")
    logger.info(f"Reports: {REPORTS_DIR}")

    return {
        'processor': processor,
        'predictor': predictor,
        'explainer': explainer,
        'visualizer': visualizer,
        'results': results,
        'data': processor.processed_data
    }


if __name__ == "__main__":
    # Run full pipeline
    pipeline_results = run_full_pipeline(tune_hyperparameters=False)

    # Test LLM if API key is available
    if GROQ_API_KEY:
        logger.info("\nTesting LLM integration...")
        llm = LLMAssistant()

        # Generate sample report
        insights = {
            'Most violations occur during evening rush hour (17:00-19:00)': True,
            'Heavy vehicles have lower violation rates': True,
            'Weekend violation rates are 15% higher than weekdays': True
        }

        report = llm.generate_weekly_report(
            pipeline_results['data'],
            insights
        )

        with open(REPORTS_DIR / 'sample_weekly_report.txt', 'w') as f:
            f.write(report)

        logger.info("Sample LLM report generated")
