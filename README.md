# Traffic Vision - Speed Violation Analysis System

A comprehensive traffic speed violation analysis system using Machine Learning and Explainable AI (XAI), designed for Q1 journal publication research.

## Author

**Mahbub Hassan**
Graduate Student & Non-ASEAN Scholar
Department of Civil Engineering, Faculty of Engineering
Chulalongkorn University, Bangkok, Thailand
Email: 6870376421@student.chula.ac.th
GitHub: [@mahbubchula](https://github.com/mahbubchula)

## Overview

This system analyzes DMT (Department of Motor Transport) speed radar data from Bangkok, Thailand to:
- Predict speed violations based on contextual factors
- Provide explainable AI insights using SHAP values
- Generate publication-quality figures
- Offer an interactive Streamlit dashboard

## Key Features

- **5 ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network
- **Explainability**: SHAP beeswarm, dependence plots, waterfall explanations
- **LLM Integration**: Groq API for natural language insights
- **Publication Figures**: 16+ figures at 300 DPI with Times New Roman font
- **Interactive Dashboard**: Streamlit-based web interface

## Model Performance

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Neural Network | 0.938 | 0.968 | 0.856 |
| Random Forest | 0.890 | 0.942 | 0.887 |
| XGBoost | 0.820 | 0.901 | 0.817 |
| Logistic Regression | 0.714 | 0.832 | 0.843 |
| LightGBM | 0.578 | 0.730 | 0.833 |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-vision.git
cd traffic-vision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

2. Get a free Groq API key from: https://console.groq.com

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

### Generate Analysis Figures
```bash
python generate_figures_corrected.py
```

### Run Full Pipeline
```bash
python analysis.py
```

## Project Structure

```
traffic_vision/
├── config.py                    # Configuration settings
├── analysis.py                  # ML + XAI pipeline
├── app.py                       # Streamlit dashboard
├── generate_figures_corrected.py # Figure generation (no leakage)
├── requirements.txt             # Dependencies
├── .env                         # API keys (not in repo)
├── .gitignore                   # Git ignore rules
├── data/
│   └── processed/               # Processed CSV data
├── models/                      # Trained model files
└── outputs/
    ├── figures/                 # Publication figures
    └── reports/                 # Analysis reports
```

## Data

The system uses DMT speed radar data with the following structure:
- `deviceid`: Radar device identifier
- `lane`: Lane number (1-4)
- `reftime`: Timestamp
- `speed`: Vehicle speed (km/h)
- `classify`: Vehicle classification (4=Car, 6=Transporter, 7=Short Truck, 8=Long Truck)

**Note**: Raw data files are not included due to size. Place your CSV files in the parent directory.

## Key Findings

1. **99% of vehicles exceed the 60 km/h speed limit** in the study area
2. Most important predictors (without data leakage):
   - Vehicle class
   - Lane selection
   - Congestion index
   - Time of day
3. Heavy vehicles have different violation patterns than light vehicles

## Generated Figures

### ML Figures
- `model_comparison.png` - Performance comparison
- `roc_curves.png` - ROC curves for all models
- `precision_recall_curves.png` - PR curves
- `confusion_matrix.png` - Best model confusion matrix
- `confusion_matrices_all.png` - All models

### XAI Figures
- `shap_beeswarm.png` - Global feature importance
- `shap_importance_bar.png` - SHAP bar chart
- `shap_dependence_top4.png` - Top 4 feature dependence
- `shap_waterfall_examples.png` - Local explanations
- `shap_interaction.png` - Feature interactions

### Data Visualization
- `temporal_patterns.png` - Hourly/daily patterns
- `vehicle_distribution.png` - Vehicle type analysis
- `speed_distribution.png` - Speed histograms
- `violation_heatmap.png` - Hour vs day heatmap
- `study_area_map.png` - Radar locations

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `GROQ_API_KEY` to secrets

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## Citation

If you use this work in your research, please cite:
```bibtex
@article{hassan2024traffic,
  title={Traffic Speed Violation Analysis using Machine Learning and Explainable AI: A Case Study of Bangkok Urban Roads},
  author={Hassan, Mahbub},
  journal={},
  year={2024},
  institution={Chulalongkorn University}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Department of Civil Engineering, Faculty of Engineering, Chulalongkorn University
- Non-ASEAN Scholarship Program
- Department of Motor Transport (DMT), Thailand
- Groq for free LLM API access

## Contact

For questions or collaboration:
- **Email**: 6870376421@student.chula.ac.th
- **GitHub**: https://github.com/mahbubchula
