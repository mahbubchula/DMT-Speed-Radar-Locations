"""
Traffic Vision Streamlit Dashboard
==================================
Interactive dashboard for traffic speed violation analysis.
Designed for Q1 journal publication quality presentation.

Author: Mahbub Hassan
        Graduate Student & Non-ASEAN Scholar
        Department of Civil Engineering, Faculty of Engineering
        Chulalongkorn University, Bangkok, Thailand
        Email: 6870376421@student.chula.ac.th
        GitHub: https://github.com/mahbubchula

Version: 1.0.0

Run with: streamlit run app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import zipfile
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, PROCESSED_DATA_DIR,
    COLORS, COLOR_PALETTE, DASHBOARD_SETTINGS, SPEED_LIMIT_DEFAULT,
    VEHICLE_CLASSES, RADAR_LOCATIONS, GROQ_API_KEY,
    get_vehicle_name, get_violation_severity
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=DASHBOARD_SETTINGS['page_title'],
    page_icon=DASHBOARD_SETTINGS['page_icon'],
    layout=DASHBOARD_SETTINGS['layout'],
    initial_sidebar_state=DASHBOARD_SETTINGS['initial_sidebar_state']
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86AB;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6C757D;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def load_processed_data() -> Optional[pd.DataFrame]:
    """Load processed data from CSV."""
    data_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_resource
def load_model():
    """Load trained model and scaler."""
    model_path = MODELS_DIR / "best_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None


@st.cache_data
def load_model_results() -> Optional[pd.DataFrame]:
    """Load model comparison results."""
    results_path = REPORTS_DIR / "model_results.csv"
    if results_path.exists():
        return pd.read_csv(results_path, index_col=0)
    return None


@st.cache_data
def load_feature_importance() -> Optional[pd.DataFrame]:
    """Load feature importance data."""
    importance_path = REPORTS_DIR / "feature_importance.csv"
    if importance_path.exists():
        return pd.read_csv(importance_path)
    return None


def get_llm_assistant():
    """Initialize LLM assistant."""
    if GROQ_API_KEY:
        try:
            from analysis import LLMAssistant
            return LLMAssistant()
        except Exception:
            return None
    return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_metric_card(value: Any, label: str, delta: Optional[str] = None) -> None:
    """Create a styled metric card."""
    delta_html = f'<div style="color: {"#06A77D" if "+" in str(delta) else "#C73E1D"}">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def plot_speed_gauge(speed: float, limit: float = SPEED_LIMIT_DEFAULT) -> go.Figure:
    """Create a speed gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=speed,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 150], 'tickwidth': 1},
            'bar': {'color': COLORS['danger'] if speed > limit else COLORS['success']},
            'steps': [
                {'range': [0, limit], 'color': '#E8F5E9'},
                {'range': [limit, limit + 20], 'color': '#FFF3E0'},
                {'range': [limit + 20, 150], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': COLORS['danger'], 'width': 4},
                'thickness': 0.75,
                'value': limit
            }
        },
        number={'suffix': ' km/h', 'font': {'size': 24}}
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig


def plot_hourly_pattern(data: pd.DataFrame) -> go.Figure:
    """Create hourly speed pattern chart."""
    hourly = data.groupby('hour').agg({
        'speed': 'mean',
        'is_violation': 'mean'
    }).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=hourly['hour'],
            y=hourly['speed'],
            name='Average Speed',
            line=dict(color=COLORS['primary'], width=3),
            mode='lines+markers'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(
            x=hourly['hour'],
            y=hourly['is_violation'] * 100,
            name='Violation Rate (%)',
            marker_color=COLORS['secondary'],
            opacity=0.6
        ),
        secondary_y=True
    )

    fig.add_hline(
        y=SPEED_LIMIT_DEFAULT,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text="Speed Limit"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    fig.update_xaxes(title_text="Hour of Day")
    fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
    fig.update_yaxes(title_text="Violation Rate (%)", secondary_y=True)

    return fig


def plot_location_map(data: pd.DataFrame) -> go.Figure:
    """Create interactive map of radar locations."""
    location_stats = data.groupby(['device_id', 'latitude', 'longitude', 'location_name']).agg({
        'speed': ['count', 'mean'],
        'is_violation': 'mean'
    }).reset_index()
    location_stats.columns = ['device_id', 'lat', 'lon', 'name', 'count', 'avg_speed', 'violation_rate']

    fig = px.scatter_mapbox(
        location_stats,
        lat='lat',
        lon='lon',
        size='count',
        color='violation_rate',
        color_continuous_scale='YlOrRd',
        hover_name='name',
        hover_data={
            'avg_speed': ':.1f',
            'violation_rate': ':.2%',
            'count': ':,',
            'lat': False,
            'lon': False
        },
        zoom=11,
        height=500
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


def plot_vehicle_distribution(data: pd.DataFrame) -> go.Figure:
    """Create vehicle type distribution chart."""
    vehicle_stats = data.groupby('vehicle_name').agg({
        'speed': 'count',
        'is_violation': 'mean'
    }).reset_index()
    vehicle_stats.columns = ['Vehicle Type', 'Count', 'Violation Rate']

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'bar'}]])

    fig.add_trace(
        go.Pie(
            labels=vehicle_stats['Vehicle Type'],
            values=vehicle_stats['Count'],
            hole=0.4,
            marker_colors=COLOR_PALETTE[:len(vehicle_stats)]
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=vehicle_stats['Vehicle Type'],
            y=vehicle_stats['Violation Rate'] * 100,
            marker_color=COLOR_PALETTE[:len(vehicle_stats)]
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )

    return fig


def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Create model comparison chart."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    available_metrics = [m for m in metrics if m in results_df.columns]

    fig = go.Figure()

    for i, metric in enumerate(available_metrics):
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=results_df.index,
            y=results_df[metric],
            marker_color=COLOR_PALETTE[i]
        ))

    fig.update_layout(
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_range=[0, 1]
    )

    return fig


def plot_shap_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create SHAP feature importance chart."""
    top_features = importance_df.head(top_n)

    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color=COLORS['primary']
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Mean |SHAP Value|",
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Header
    st.markdown('<div class="main-header">Traffic Vision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Speed Violation Analysis System for Q1 Research</div>',
               unsafe_allow_html=True)

    # Load data
    data = load_processed_data()
    model, scaler = load_model()
    model_results = load_model_results()
    feature_importance = load_feature_importance()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=80)
        st.title("Controls")

        if data is not None:
            # Date range filter
            st.subheader("Date Range")
            min_date = data['timestamp'].min().date()
            max_date = data['timestamp'].max().date()
            date_range = st.date_input(
                "Select dates",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            # Location filter
            st.subheader("Location")
            locations = ['All'] + sorted(data['location_name'].unique().tolist())
            selected_location = st.selectbox("Select location", locations)

            # Vehicle type filter
            st.subheader("Vehicle Type")
            vehicle_types = ['All'] + sorted(data['vehicle_name'].unique().tolist())
            selected_vehicle = st.selectbox("Select vehicle type", vehicle_types)

            # Apply filters
            filtered_data = data.copy()
            if len(date_range) == 2:
                filtered_data = filtered_data[
                    (filtered_data['timestamp'].dt.date >= date_range[0]) &
                    (filtered_data['timestamp'].dt.date <= date_range[1])
                ]
            if selected_location != 'All':
                filtered_data = filtered_data[filtered_data['location_name'] == selected_location]
            if selected_vehicle != 'All':
                filtered_data = filtered_data[filtered_data['vehicle_name'] == selected_vehicle]

            st.divider()
            st.metric("Filtered Records", f"{len(filtered_data):,}")
        else:
            filtered_data = None
            st.warning("No processed data found. Run analysis.py first.")

    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Predictions", "Explainability",
        "AI Insights", "Export"
    ])

    # ==========================================================================
    # TAB 1: OVERVIEW
    # ==========================================================================
    with tab1:
        if filtered_data is not None and len(filtered_data) > 0:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_violations = filtered_data['is_violation'].sum()
                st.metric("Total Violations", f"{total_violations:,}")

            with col2:
                violation_rate = filtered_data['is_violation'].mean() * 100
                st.metric("Violation Rate", f"{violation_rate:.1f}%")

            with col3:
                avg_speed = filtered_data['speed'].mean()
                st.metric("Average Speed", f"{avg_speed:.1f} km/h")

            with col4:
                hotspots = filtered_data.groupby('location_name')['is_violation'].mean().nlargest(1)
                hotspot_name = hotspots.index[0] if len(hotspots) > 0 else "N/A"
                st.metric("Top Hotspot", hotspot_name[:15])

            st.divider()

            # Charts row 1
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Hourly Traffic Pattern")
                fig = plot_hourly_pattern(filtered_data)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Vehicle Distribution")
                fig = plot_vehicle_distribution(filtered_data)
                st.plotly_chart(fig, use_container_width=True)

            # Map
            st.subheader("Radar Locations & Violation Hotspots")
            fig = plot_location_map(filtered_data)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed statistics
            with st.expander("Detailed Statistics"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Speed Statistics**")
                    speed_stats = filtered_data['speed'].describe()
                    st.dataframe(speed_stats)

                with col2:
                    st.write("**Violations by Day**")
                    daily_violations = filtered_data.groupby(
                        filtered_data['timestamp'].dt.day_name()
                    )['is_violation'].mean().sort_values(ascending=False)
                    st.dataframe(daily_violations)
        else:
            st.info("No data available. Please run the analysis pipeline first.")

    # ==========================================================================
    # TAB 2: PREDICTIONS
    # ==========================================================================
    with tab2:
        st.subheader("Real-time Violation Prediction")

        if model is not None and scaler is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**Enter Traffic Parameters**")

                hour = st.slider("Hour of Day", 0, 23, 12)
                day_of_week = st.selectbox(
                    "Day of Week",
                    options=list(range(7)),
                    format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday',
                                          'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
                )
                vehicle_class = st.selectbox(
                    "Vehicle Type",
                    options=list(VEHICLE_CLASSES.keys()),
                    format_func=lambda x: VEHICLE_CLASSES[x]['name']
                )
                current_speed = st.slider("Current Speed (km/h)", 0, 150, 70)

                location = st.selectbox(
                    "Location",
                    options=list(RADAR_LOCATIONS.keys()),
                    format_func=lambda x: RADAR_LOCATIONS[x][2]
                )

                predict_button = st.button("Predict Violation", type="primary")

            with col2:
                if predict_button:
                    # Prepare features for prediction
                    is_weekend = 1 if day_of_week >= 5 else 0
                    is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
                    is_night = 1 if hour >= 22 or hour <= 5 else 0
                    speed_over_limit = max(0, current_speed - SPEED_LIMIT_DEFAULT)
                    speed_ratio = current_speed / SPEED_LIMIT_DEFAULT
                    is_heavy = 1 if VEHICLE_CLASSES[vehicle_class]['category'] == 'heavy' else 0

                    lat, lon, _ = RADAR_LOCATIONS[location]

                    # Create feature vector (simplified - would need full features in production)
                    features = np.array([[
                        hour, day_of_week, is_weekend, is_rush_hour, is_night,
                        speed_over_limit, speed_ratio,
                        list(RADAR_LOCATIONS.keys()).index(location), lat, lon,
                        70.0, 0.4,  # location_avg_speed, location_violation_rate (defaults)
                        vehicle_class, 0 if is_heavy else 1, is_heavy,
                        100, 0.5, 70.0, 70.0,  # traffic features (defaults)
                        current_speed, current_speed, current_speed, current_speed, 0.0  # lag features
                    ]])

                    try:
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)[0]
                        probability = model.predict_proba(features_scaled)[0][1]

                        # Display prediction
                        st.write("**Prediction Result**")
                        fig = plot_speed_gauge(current_speed)
                        st.plotly_chart(fig, use_container_width=True)

                        if prediction == 1:
                            st.error(f"**VIOLATION PREDICTED** (Confidence: {probability:.1%})")
                            severity = get_violation_severity(current_speed)
                            st.warning(f"Violation Severity: {severity}")
                        else:
                            st.success(f"**No Violation Expected** (Confidence: {1-probability:.1%})")

                        # Show similar cases
                        if filtered_data is not None:
                            st.write("**Similar Historical Cases**")
                            similar = filtered_data[
                                (filtered_data['hour'] == hour) &
                                (filtered_data['vehicle_class'] == vehicle_class)
                            ].head(5)[['timestamp', 'speed', 'is_violation', 'location_name']]
                            st.dataframe(similar)

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        else:
            st.warning("Model not loaded. Please run analysis.py to train models first.")

    # ==========================================================================
    # TAB 3: EXPLAINABILITY
    # ==========================================================================
    with tab3:
        st.subheader("Model Explainability")

        if model_results is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Model Performance Comparison**")
                fig = plot_model_comparison(model_results)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Model Metrics**")
                st.dataframe(model_results.style.format({
                    'accuracy': '{:.3f}',
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'f1': '{:.3f}',
                    'roc_auc': '{:.3f}'
                }))

        if feature_importance is not None:
            st.write("**Feature Importance (SHAP Values)**")
            fig = plot_shap_importance(feature_importance)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Feature Importance Table"):
                st.dataframe(feature_importance)

        # Load and display saved figures
        st.write("**Analysis Figures**")
        figure_files = list(FIGURES_DIR.glob("*.png"))

        if figure_files:
            cols = st.columns(2)
            for i, fig_file in enumerate(figure_files[:6]):
                with cols[i % 2]:
                    st.image(str(fig_file), caption=fig_file.stem.replace('_', ' ').title())
        else:
            st.info("No figures found. Run analysis.py to generate figures.")

    # ==========================================================================
    # TAB 4: AI INSIGHTS
    # ==========================================================================
    with tab4:
        st.subheader("AI-Powered Insights")

        llm = get_llm_assistant()

        if llm is not None and llm.client is not None:
            insight_type = st.radio(
                "Select Insight Type",
                ["Weekly Report", "Location Analysis", "Custom Query"],
                horizontal=True
            )

            if insight_type == "Weekly Report":
                if st.button("Generate Weekly Report", type="primary"):
                    if filtered_data is not None:
                        with st.spinner("Generating report..."):
                            insights = {
                                'Peak violation hours': '17:00-19:00',
                                'Highest risk vehicle': filtered_data.groupby('vehicle_name')['is_violation'].mean().idxmax(),
                                'Safest day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][
                                    filtered_data.groupby('day_of_week')['is_violation'].mean().idxmin()
                                ]
                            }
                            report = llm.generate_weekly_report(filtered_data, insights)
                            st.markdown(report)
                    else:
                        st.warning("No data available for report generation.")

            elif insight_type == "Location Analysis":
                if filtered_data is not None:
                    location = st.selectbox(
                        "Select Location for Analysis",
                        filtered_data['location_name'].unique()
                    )

                    if st.button("Analyze Location", type="primary"):
                        with st.spinner("Analyzing location..."):
                            location_data = filtered_data[filtered_data['location_name'] == location]
                            recommendations = llm.generate_location_recommendations(
                                location, location_data
                            )
                            st.markdown(recommendations)

            else:  # Custom Query
                query = st.text_area(
                    "Ask a question about the traffic data",
                    placeholder="e.g., What are the main factors contributing to speed violations?"
                )

                if st.button("Get Answer", type="primary") and query:
                    if filtered_data is not None:
                        with st.spinner("Processing query..."):
                            context = f"""
                            Data Summary:
                            - Total records: {len(filtered_data):,}
                            - Date range: {filtered_data['timestamp'].min()} to {filtered_data['timestamp'].max()}
                            - Average speed: {filtered_data['speed'].mean():.1f} km/h
                            - Violation rate: {filtered_data['is_violation'].mean()*100:.1f}%
                            - Locations: {filtered_data['location_name'].nunique()}
                            - Vehicle types: {', '.join(filtered_data['vehicle_name'].unique())}
                            """
                            answer = llm.answer_query(query, context)
                            st.markdown(answer)
        else:
            st.warning("""
            LLM features require a Groq API key.

            To enable AI insights:
            1. Get a free API key from https://console.groq.com
            2. Set the GROQ_API_KEY environment variable
            3. Restart the application
            """)

    # ==========================================================================
    # TAB 5: EXPORT
    # ==========================================================================
    with tab5:
        st.subheader("Export Data & Reports")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Download Data**")

            # Export filtered data
            if filtered_data is not None:
                csv_data = filtered_data.to_csv(index=False)
                st.download_button(
                    "Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name="filtered_traffic_data.csv",
                    mime="text/csv"
                )

            # Export model results
            if model_results is not None:
                results_csv = model_results.to_csv()
                st.download_button(
                    "Download Model Results (CSV)",
                    data=results_csv,
                    file_name="model_results.csv",
                    mime="text/csv"
                )

            # Export feature importance
            if feature_importance is not None:
                importance_csv = feature_importance.to_csv(index=False)
                st.download_button(
                    "Download Feature Importance (CSV)",
                    data=importance_csv,
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )

        with col2:
            st.write("**Download Figures**")

            figure_files = list(FIGURES_DIR.glob("*.png"))
            if figure_files:
                # Create ZIP of all figures
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fig_file in figure_files:
                        zf.write(fig_file, fig_file.name)

                st.download_button(
                    "Download All Figures (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="traffic_vision_figures.zip",
                    mime="application/zip"
                )

                # Individual figure downloads
                with st.expander("Download Individual Figures"):
                    for fig_file in figure_files:
                        with open(fig_file, 'rb') as f:
                            st.download_button(
                                f"{fig_file.stem}.png",
                                data=f.read(),
                                file_name=fig_file.name,
                                mime="image/png",
                                key=fig_file.name
                            )
            else:
                st.info("No figures available. Run analysis.py first.")

        st.divider()

        # Reports section
        st.write("**Reports**")
        report_files = list(REPORTS_DIR.glob("*.txt")) + list(REPORTS_DIR.glob("*.csv"))

        if report_files:
            for report_file in report_files:
                with open(report_file, 'r') as f:
                    content = f.read()
                st.download_button(
                    f"Download {report_file.name}",
                    data=content,
                    file_name=report_file.name,
                    mime="text/plain" if report_file.suffix == '.txt' else "text/csv",
                    key=f"report_{report_file.name}"
                )

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6C757D; font-size: 0.8rem;">
        Traffic Vision Analysis System | Designed for Q1 Journal Research<br>
        Powered by XGBoost, SHAP, and Groq LLM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
