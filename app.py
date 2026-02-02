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
from typing import Optional
import zipfile
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "outputs" / "figures"
EDA_DIR = FIGURES_DIR / "eda"
REPORTS_DIR = SCRIPT_DIR / "outputs" / "reports"
PROCESSED_DATA_DIR = SCRIPT_DIR / "data" / "processed"
MODELS_DIR = SCRIPT_DIR / "models"

# Configuration
SPEED_LIMIT_DEFAULT = 60
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#06A77D",
    "danger": "#C73E1D",
}
COLOR_PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D", "#C73E1D"]

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Traffic Vision - Speed Violation Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .author-info {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

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

@st.cache_data
def load_eda_statistics() -> dict:
    """Load EDA statistics."""
    stats = {}

    stats_path = REPORTS_DIR / "eda_statistics_summary.csv"
    if stats_path.exists():
        stats['summary'] = pd.read_csv(stats_path)

    vehicle_path = REPORTS_DIR / "eda_vehicle_statistics.csv"
    if vehicle_path.exists():
        stats['vehicle'] = pd.read_csv(vehicle_path)

    hourly_path = REPORTS_DIR / "eda_hourly_statistics.csv"
    if hourly_path.exists():
        stats['hourly'] = pd.read_csv(hourly_path)

    location_path = REPORTS_DIR / "eda_location_statistics.csv"
    if location_path.exists():
        stats['location'] = pd.read_csv(location_path)

    return stats

def get_figure_files(directory: Path) -> list:
    """Get list of PNG figure files in directory."""
    if directory.exists():
        return sorted([f for f in directory.glob("*.png")])
    return []

def check_data_available() -> bool:
    """Check if processed data is available."""
    data_path = PROCESSED_DATA_DIR / "processed_data.csv"
    return data_path.exists()

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Create model comparison chart."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    available_metrics = [m for m in metrics if m in results_df.columns]

    fig = go.Figure()

    for i, metric in enumerate(available_metrics):
        fig.add_trace(go.Bar(
            name=metric.upper().replace('_', '-'),
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

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create feature importance chart."""
    top_features = importance_df.head(top_n).sort_values('importance', ascending=True)

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
    st.markdown('<div class="main-header">🚦 Traffic Vision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Speed Violation Analysis System for Q1 Research</div>',
               unsafe_allow_html=True)

    # Load data
    model_results = load_model_results()
    feature_importance = load_feature_importance()
    eda_stats = load_eda_statistics()
    data_available = check_data_available()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="author-info">
        <strong>👤 Author</strong><br>
        Mahbub Hassan<br>
        <small>Graduate Student & Non-ASEAN Scholar<br>
        Dept. of Civil Engineering<br>
        Chulalongkorn University</small>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        if data_available:
            st.success("✅ Data Available")
        else:
            st.info("📊 Viewing pre-generated results")
            st.caption("Full interactive mode requires local deployment with data files.")

    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🔬 EDA Figures", "🤖 ML Results",
        "🔍 Explainability", "💬 AI Chat", "📥 Export"
    ])

    # ==========================================================================
    # TAB 1: OVERVIEW
    # ==========================================================================
    with tab1:
        st.subheader("Study Overview")

        # Key metrics from EDA statistics
        if 'summary' in eda_stats:
            summary = eda_stats['summary']

            col1, col2, col3, col4 = st.columns(4)

            # Extract values from summary
            summary_dict = dict(zip(summary['Metric'], summary['Value']))

            with col1:
                st.metric("Total Records", summary_dict.get('Total Records', 'N/A'))
            with col2:
                st.metric("Violation Rate", summary_dict.get('Overall Violation Rate (%)', 'N/A') + '%')
            with col3:
                st.metric("Mean Speed", summary_dict.get('Mean Speed (km/h)', 'N/A') + ' km/h')
            with col4:
                st.metric("Locations", summary_dict.get('Number of Locations', 'N/A'))

            st.divider()

            # Display summary table
            st.subheader("Dataset Statistics")
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.info("Statistics summary not available.")

        # Vehicle statistics
        if 'vehicle' in eda_stats:
            st.subheader("Vehicle Type Statistics")
            st.dataframe(eda_stats['vehicle'], use_container_width=True)

    # ==========================================================================
    # TAB 2: EDA FIGURES
    # ==========================================================================
    with tab2:
        st.subheader("Exploratory Data Analysis Figures")
        st.caption("All figures are publication-ready (300 DPI, Times New Roman)")

        eda_figures = get_figure_files(EDA_DIR)

        if eda_figures:
            # Create figure gallery
            figure_names = {
                'eda_01': 'Speed Distribution',
                'eda_02': 'Speed by Vehicle Type (Box)',
                'eda_03': 'Speed by Vehicle (Violin)',
                'eda_04': 'Hourly Pattern',
                'eda_05': 'Daily Pattern',
                'eda_06': 'Speed Heatmap (Hour×Day)',
                'eda_07': 'Violation Heatmap',
                'eda_08': 'Vehicle Composition',
                'eda_09': 'Speed by Location',
                'eda_10': 'Lane Analysis',
                'eda_11': 'Monthly Trend',
                'eda_12': 'Correlation Matrix',
                'eda_13': 'Speed Percentiles',
                'eda_14': 'Rush Hour Comparison',
                'eda_15': 'Cumulative Distribution',
            }

            # Display figures in grid
            cols = st.columns(2)
            for i, fig_path in enumerate(eda_figures):
                fig_key = fig_path.stem[:6]
                fig_name = figure_names.get(fig_key, fig_path.stem)

                with cols[i % 2]:
                    st.image(str(fig_path), caption=fig_name, use_container_width=True)
        else:
            st.warning("EDA figures not found. Run `python generate_eda_figures.py` first.")

        # Also show main figures
        st.divider()
        st.subheader("Additional Visualizations")

        main_figures = get_figure_files(FIGURES_DIR)
        main_figures = [f for f in main_figures if 'eda' not in f.stem.lower()]

        if main_figures:
            cols = st.columns(2)
            for i, fig_path in enumerate(main_figures[:6]):
                with cols[i % 2]:
                    st.image(str(fig_path), caption=fig_path.stem.replace('_', ' ').title(),
                            use_container_width=True)

    # ==========================================================================
    # TAB 3: ML RESULTS
    # ==========================================================================
    with tab3:
        st.subheader("Machine Learning Model Results")

        if model_results is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Model Performance Comparison**")
                fig = plot_model_comparison(model_results)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Model Metrics Table**")
                st.dataframe(model_results.style.format({
                    col: '{:.4f}' for col in model_results.columns if col != 'best_params'
                }), use_container_width=True)

            # Best model highlight
            best_model = model_results['f1'].idxmax()
            best_f1 = model_results.loc[best_model, 'f1']

            st.success(f"**Best Model**: {best_model} with F1 Score = {best_f1:.4f}")

            # Show ML figures
            st.divider()
            st.write("**Model Performance Figures**")

            ml_figures = ['model_comparison.png', 'roc_curves.png', 'confusion_matrix.png',
                         'precision_recall_curves.png']

            cols = st.columns(2)
            for i, fig_name in enumerate(ml_figures):
                fig_path = FIGURES_DIR / fig_name
                if fig_path.exists():
                    with cols[i % 2]:
                        st.image(str(fig_path), caption=fig_name.replace('_', ' ').replace('.png', '').title(),
                                use_container_width=True)
        else:
            st.warning("Model results not found. Run `python generate_figures_corrected.py` first.")

    # ==========================================================================
    # TAB 4: EXPLAINABILITY
    # ==========================================================================
    with tab4:
        st.subheader("Explainable AI (SHAP Analysis)")

        if feature_importance is not None:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Feature Importance (SHAP Values)**")
                fig = plot_feature_importance(feature_importance)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Top Features**")
                st.dataframe(feature_importance.head(10), use_container_width=True, hide_index=True)

        # SHAP figures
        st.divider()
        st.write("**SHAP Explanation Figures**")

        shap_figures = ['shap_beeswarm.png', 'shap_importance_bar.png',
                       'shap_dependence_top4.png', 'shap_waterfall_examples.png',
                       'shap_interaction.png']

        cols = st.columns(2)
        for i, fig_name in enumerate(shap_figures):
            fig_path = FIGURES_DIR / fig_name
            if fig_path.exists():
                with cols[i % 2]:
                    st.image(str(fig_path), caption=fig_name.replace('_', ' ').replace('.png', '').title(),
                            use_container_width=True)

    # ==========================================================================
    # TAB 5: AI CHAT
    # ==========================================================================
    with tab5:
        st.subheader("💬 AI-Powered Traffic Analysis Chat")

        # Get API key from Streamlit secrets or environment
        groq_api_key = None
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY", None)
        except:
            pass

        if not groq_api_key:
            import os
            groq_api_key = os.environ.get("GROQ_API_KEY", None)

        if groq_api_key:
            try:
                from groq import Groq

                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Create context from statistics
                context = ""
                if 'summary' in eda_stats:
                    summary_dict = dict(zip(eda_stats['summary']['Metric'], eda_stats['summary']['Value']))
                    context = f"""
Traffic Data Analysis Context:
- Total Records: {summary_dict.get('Total Records', 'N/A')}
- Date Range: {summary_dict.get('Date Range', 'N/A')}
- Number of Locations: {summary_dict.get('Number of Locations', 'N/A')}
- Mean Speed: {summary_dict.get('Mean Speed (km/h)', 'N/A')} km/h
- Speed Limit: 60 km/h
- Overall Violation Rate: {summary_dict.get('Overall Violation Rate (%)', 'N/A')}%
- Peak Hour Violation Rate: {summary_dict.get('Peak Hour Violation Rate (%)', 'N/A')}%
- Weekend Violation Rate: {summary_dict.get('Weekend Violation Rate (%)', 'N/A')}%
- Weekday Violation Rate: {summary_dict.get('Weekday Violation Rate (%)', 'N/A')}%

Best ML Model: Neural Network (F1 = 0.968)
Key finding: 99% of vehicles exceed the 60 km/h speed limit.
"""

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("Ask about traffic violations, patterns, or recommendations..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            client = Groq(api_key=groq_api_key)

                            system_prompt = f"""You are an AI traffic safety analyst assistant.
You have access to the following traffic violation analysis data:

{context}

Provide helpful, accurate insights based on this data. Be concise and professional.
If asked about something not in the data, say so clearly."""

                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    *[{"role": m["role"], "content": m["content"]}
                                      for m in st.session_state.messages]
                                ],
                                max_tokens=1024,
                                temperature=0.3
                            )

                            assistant_response = response.choices[0].message.content
                            st.markdown(assistant_response)
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                # Clear chat button
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    st.rerun()

                # Example questions
                st.divider()
                st.write("**Example Questions:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("• What is the overall violation rate?")
                    st.caption("• When do most violations occur?")
                    st.caption("• Which vehicle types violate most?")
                with col2:
                    st.caption("• How to reduce speed violations?")
                    st.caption("• What does the ML model predict?")
                    st.caption("• Recommendations for traffic management?")

            except ImportError:
                st.warning("Groq library not installed. Install with: `pip install groq`")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("""
            **API Key Required**

            To enable AI Chat, add your Groq API key:

            **For Streamlit Cloud:**
            1. Go to App Settings → Secrets
            2. Add: `GROQ_API_KEY = "your_key_here"`

            **Get free API key:** https://console.groq.com
            """)

    # ==========================================================================
    # TAB 6: EXPORT
    # ==========================================================================
    with tab6:
        st.subheader("Download Figures & Reports")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**📊 EDA Figures**")

            eda_figures = list(EDA_DIR.glob("*.png")) if EDA_DIR.exists() else []
            if eda_figures:
                # Create ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fig_file in eda_figures:
                        zf.write(fig_file, f"eda/{fig_file.name}")
                    # Add PDFs too
                    for pdf_file in EDA_DIR.glob("*.pdf"):
                        zf.write(pdf_file, f"eda/{pdf_file.name}")

                st.download_button(
                    "📥 Download All EDA Figures (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="eda_figures.zip",
                    mime="application/zip"
                )

                st.caption(f"{len(eda_figures)} PNG + PDF figures")

        with col2:
            st.write("**🤖 ML & SHAP Figures**")

            ml_figures = [f for f in FIGURES_DIR.glob("*.png") if 'eda' not in str(f)]
            if ml_figures:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fig_file in ml_figures:
                        zf.write(fig_file, fig_file.name)
                    for pdf_file in FIGURES_DIR.glob("*.pdf"):
                        if 'eda' not in str(pdf_file):
                            zf.write(pdf_file, pdf_file.name)

                st.download_button(
                    "📥 Download ML/SHAP Figures (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="ml_shap_figures.zip",
                    mime="application/zip"
                )

                st.caption(f"{len(ml_figures)} PNG + PDF figures")

        st.divider()

        st.write("**📋 Statistical Reports**")

        report_files = list(REPORTS_DIR.glob("*.csv")) if REPORTS_DIR.exists() else []

        if report_files:
            cols = st.columns(3)
            for i, report_file in enumerate(report_files):
                with cols[i % 3]:
                    with open(report_file, 'r') as f:
                        content = f.read()
                    st.download_button(
                        f"📄 {report_file.name}",
                        data=content,
                        file_name=report_file.name,
                        mime="text/csv",
                        key=f"report_{report_file.name}"
                    )

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6C757D; font-size: 0.8rem;">
        Traffic Vision Analysis System | Designed for Q1 Journal Research<br>
        Author: Mahbub Hassan | Chulalongkorn University<br>
        Powered by XGBoost, SHAP, and Groq LLM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
