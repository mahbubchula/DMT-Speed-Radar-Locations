"""
Traffic Vision System Configuration
===================================
Central configuration file for all settings, paths, and parameters.
Designed for Q1 journal publication quality analysis.

Author: Traffic Vision Research Team
Version: 1.0.0
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Load environment variables from .env file (secure API key storage)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed, will use system environment variables

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent  # Parent directory contains month folders
PROJECT_DIR = BASE_DIR

# Output directories
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Raw data folder names (in DATA_DIR)
RAW_DATA_FOLDERS = [
    "1_Febuary_2021",
    "2_May_2021",
    "3_June_2021",
    "4_August_2021",
    "5_November_2021"
]

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Groq API Configuration (Free LLM API)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Set via environment variable
GROQ_MODEL = "llama-3.3-70b-versatile"  # Free tier model (updated)
GROQ_RATE_LIMIT = 30  # requests per minute
GROQ_MAX_TOKENS = 4096
GROQ_TEMPERATURE = 0.3  # Lower for more consistent outputs

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# CSV column names
CSV_COLUMNS = {
    "device_id": "deviceid",
    "lane": "lane",
    "timestamp": "reftime",
    "speed": "speed",
    "vehicle_class": "classify"
}

# Vehicle classification mapping (from radar specifications)
VEHICLE_CLASSES: Dict[int, Dict] = {
    4: {"name": "Car", "length_range": "4.4-5.4m", "category": "light"},
    6: {"name": "Transporter", "length_range": "5.6-9.4m", "category": "medium"},
    7: {"name": "Short Truck", "length_range": "9.6-12m", "category": "heavy"},
    8: {"name": "Long Truck", "length_range": "12.2-25m", "category": "heavy"}
}

# Lane position mapping
LANE_POSITIONS: Dict[int, str] = {
    1: "Left Lane",
    2: "Middle Lane",
    3: "Right Lane",
    4: "Right Lane"  # Some sensors use 4 for right lane
}

# Radar device locations (approximate coordinates for Bangkok, Thailand)
# Format: device_id -> (latitude, longitude, location_name)
RADAR_LOCATIONS: Dict[str, Tuple[float, float, str]] = {
    "TCU-BK-RDO-01": (13.7563, 100.5018, "Bangkok Central 1"),
    "TCU-BK-RDO-02": (13.7580, 100.5035, "Bangkok Central 2"),
    "TCU-BK-RDO-03": (13.7545, 100.5052, "Bangkok Central 3"),
    "TCU-BK-RDS-01": (13.7528, 100.5069, "Bangkok South 1"),
    "TCU-BK-RDS-02": (13.7511, 100.5086, "Bangkok South 2"),
    "TCU-SS-RDO-01": (13.7494, 100.4950, "Samsen 1"),
    "TCU-SS-RDO-02": (13.7477, 100.4933, "Samsen 2"),
    "TCU-LP-RDO-01": (13.7600, 100.5100, "Lat Phrao 1"),
    "TCU-AN-RDO-01": (13.7617, 100.5117, "Ari-Nana 1"),
    "TCU-DD-RDS-01": (13.7460, 100.5134, "Din Daeng South 1"),
    "TCU-DD-RDS-02": (13.7443, 100.5151, "Din Daeng South 2"),
    "TCU-DM-RDO-01": (13.7634, 100.5168, "Don Muang 1"),
    "TCU-DM-RDS-01": (13.7651, 100.5185, "Don Muang South 1"),
    "TCU-DM-RDS-02": (13.7668, 100.5202, "Don Muang South 2")
}

# =============================================================================
# SPEED LIMIT CONFIGURATION
# =============================================================================

# Speed limits (km/h)
SPEED_LIMIT_DEFAULT = 60  # Default urban road speed limit
SPEED_LIMIT_HIGHWAY = 90  # Highway speed limit
SPEED_LIMIT_SEVERE = 80  # Threshold for severe violation

# Speed filtering thresholds (for data cleaning)
SPEED_MIN_VALID = 0  # Minimum valid speed (km/h)
SPEED_MAX_VALID = 200  # Maximum valid speed (km/h)
SPEED_OUTLIER_LOW = 5  # Very low speeds (possibly stationary)
SPEED_OUTLIER_HIGH = 180  # Very high speeds (likely sensor error)

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Time-based features
RUSH_HOUR_MORNING = (7, 9)  # 7:00 AM - 9:00 AM
RUSH_HOUR_EVENING = (17, 19)  # 5:00 PM - 7:00 PM
WEEKEND_DAYS = [5, 6]  # Saturday=5, Sunday=6

# Feature lists
TEMPORAL_FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_rush_hour",
    "month", "day_of_month", "week_of_year", "is_night"
]

SPEED_FEATURES = [
    "speed", "speed_over_limit", "violation_severity",
    "speed_ratio", "speed_category"
]

LOCATION_FEATURES = [
    "radar_id_encoded", "latitude", "longitude",
    "location_avg_speed", "location_violation_rate"
]

VEHICLE_FEATURES = [
    "vehicle_class_encoded", "vehicle_category_encoded",
    "is_heavy_vehicle"
]

TRAFFIC_FEATURES = [
    "vehicles_per_hour", "congestion_index",
    "hourly_avg_speed", "lane_avg_speed"
]

LAG_FEATURES = [
    "speed_lag_1", "speed_lag_2", "speed_lag_3",
    "rolling_avg_speed_5", "rolling_avg_speed_10",
    "speed_diff_from_avg"
]

# All features for modeling
ALL_FEATURES = (
    TEMPORAL_FEATURES + SPEED_FEATURES[1:] +  # Exclude 'speed' as it's the target-related
    LOCATION_FEATURES + VEHICLE_FEATURES + TRAFFIC_FEATURES + LAG_FEATURES
)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Cross-validation
CV_FOLDS = 5

# Class balance threshold (for stratification)
MIN_CLASS_RATIO = 0.1

# Model hyperparameter grids
HYPERPARAMETER_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "max_iter": [1000]
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    },
    "lightgbm": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, -1],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [31, 50]
    },
    "mlp": {
        "hidden_layer_sizes": [(100,), (100, 50), (100, 50, 25)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01]
    }
}

# =============================================================================
# VISUALIZATION CONFIGURATION (Publication Quality)
# =============================================================================

# Color palette (professional, colorblind-friendly)
COLORS = {
    "primary": "#2E86AB",      # Blue
    "secondary": "#A23B72",    # Purple
    "accent": "#F18F01",       # Orange
    "success": "#06A77D",      # Green
    "danger": "#C73E1D",       # Red
    "neutral": "#6C757D",      # Gray
    "background": "#FFFFFF",   # White
    "text": "#212529"          # Dark gray
}

# Extended color palette for multiple categories
COLOR_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#06A77D", "#C73E1D",
    "#5C4D7D", "#FFB703", "#219EBC", "#8ECAE6", "#023047"
]

# Figure settings (Q1 journal standards)
FIGURE_SETTINGS = {
    "font_family": "Times New Roman",
    "font_size_title": 14,  # Not used (no titles)
    "font_size_labels": 13,
    "font_size_ticks": 12,
    "font_size_legend": 11,
    "font_size_annotation": 10,
    "dpi": 300,
    "figure_size": (10, 6),
    "figure_size_small": (8, 5),
    "figure_size_large": (12, 8),
    "line_width": 1.5,
    "marker_size": 6,
    "grid_alpha": 0.3,
    "legend_frameon": False,
    "tight_layout": True,
    "format": "png"  # Also save as PDF for vector
}

# Matplotlib style configuration
MPL_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False
}

# =============================================================================
# SHAP CONFIGURATION
# =============================================================================

SHAP_SETTINGS = {
    "max_display": 15,  # Maximum features to display
    "sample_size": 1000,  # Sample size for SHAP calculation
    "plot_type": "beeswarm",  # Default plot type
    "interaction_features": 5  # Number of features for interaction analysis
}

# =============================================================================
# LLM PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES = {
    "weekly_report": """
You are a traffic safety analyst. Generate a professional weekly report based on the following data summary:

Data Period: {date_range}
Total Vehicles Detected: {total_vehicles:,}
Total Violations: {total_violations:,}
Violation Rate: {violation_rate:.2f}%

Key Statistics:
- Average Speed: {avg_speed:.2f} km/h
- Maximum Speed: {max_speed:.2f} km/h
- Most Common Violation Time: {peak_violation_hour}
- Highest Violation Location: {top_violation_location}

Top Findings:
{key_findings}

Generate a concise professional report (300-400 words) with:
1. Executive Summary
2. Key Findings
3. Recommendations for traffic management
""",

    "prediction_explanation": """
You are a traffic safety expert. Explain the following speed violation prediction in simple terms:

Prediction: {prediction} (Confidence: {confidence:.2f}%)
Input Features:
{feature_values}

Top Contributing Factors (SHAP values):
{shap_contributions}

Provide a clear, non-technical explanation (100-150 words) of:
1. Why this prediction was made
2. Which factors were most influential
3. What this means for traffic safety
""",

    "location_recommendation": """
You are a traffic safety consultant. Based on the following location analysis, provide recommendations:

Location: {location_name}
Analysis Period: {date_range}
Average Speed: {avg_speed:.2f} km/h
Violation Rate: {violation_rate:.2f}%
Peak Violation Hours: {peak_hours}
Most Common Violators: {vehicle_types}

Historical Patterns:
{patterns}

Provide specific, actionable recommendations (200-250 words) for:
1. Infrastructure improvements
2. Enforcement strategies
3. Public awareness measures
""",

    "query_response": """
You are a traffic data analyst assistant. Answer the following question based on the provided context:

Question: {question}

Available Data Context:
{context}

Provide a clear, accurate answer based only on the data provided. If the data doesn't support a definitive answer, acknowledge the limitations.
"""
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

DASHBOARD_SETTINGS = {
    "page_title": "Traffic Vision - Speed Violation Analysis",
    "page_icon": "🚗",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": COLORS["primary"],
        "backgroundColor": COLORS["background"],
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": COLORS["text"]
    }
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": LOGS_DIR / "traffic_vision.log"
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

EXPORT_SETTINGS = {
    "csv_encoding": "utf-8",
    "excel_engine": "openpyxl",
    "pdf_page_size": "A4",
    "zip_compression": "ZIP_DEFLATED"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_vehicle_name(class_code: int) -> str:
    """Get vehicle name from classification code."""
    return VEHICLE_CLASSES.get(class_code, {}).get("name", "Unknown")

def get_vehicle_category(class_code: int) -> str:
    """Get vehicle category from classification code."""
    return VEHICLE_CLASSES.get(class_code, {}).get("category", "unknown")

def get_lane_name(lane_code: int) -> str:
    """Get lane name from lane code."""
    return LANE_POSITIONS.get(lane_code, f"Lane {lane_code}")

def get_radar_location(device_id: str) -> Tuple[float, float, str]:
    """Get radar location coordinates and name."""
    return RADAR_LOCATIONS.get(
        device_id,
        (13.7563, 100.5018, "Unknown Location")  # Default to Bangkok center
    )

def is_violation(speed: float, limit: float = SPEED_LIMIT_DEFAULT) -> bool:
    """Check if speed is a violation."""
    return speed > limit

def get_violation_severity(speed: float, limit: float = SPEED_LIMIT_DEFAULT) -> str:
    """Categorize violation severity."""
    if speed <= limit:
        return "No Violation"
    elif speed <= limit + 10:
        return "Minor"
    elif speed <= limit + 20:
        return "Moderate"
    elif speed <= limit + 40:
        return "Severe"
    else:
        return "Extreme"
