# Streamlit Labs

Interactive ML dashboard for data exploration, model training, and visualization using Streamlit.

## Features

- **Interactive Dashboard**: Multi-tab interface for comprehensive ML workflow
- **Data Exploration**: Visual analysis of multiple datasets (Iris, Wine, Breast Cancer)
- **Model Training**: Train and compare multiple ML algorithms
- **Advanced Analytics**: PCA, t-SNE, clustering, and feature importance
- **Real-time Predictions**: Interactive prediction interface
- **Model Management**: Save and load trained models

## Uniqueness vs Reference Lab

- **Multi-Dataset Support**: Works with Iris, Wine, and Breast Cancer datasets
- **Advanced Visualizations**: Plotly-based interactive charts
- **Model Comparison**: Side-by-side performance comparison
- **Dimensionality Reduction**: PCA and t-SNE analysis
- **Clustering**: K-means clustering with visualization
- **Modern UI**: Custom CSS styling and responsive design

## Quick Start

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Visit: http://localhost:8501

### Docker

```bash
# Build image
docker build -t streamlit-labs .

# Run container
docker run -p 8501:8501 streamlit-labs
```

## Dashboard Features

### 📊 Data Exploration Tab
- Dataset overview and statistics
- Interactive scatter plots
- Class distribution analysis
- Feature correlation heatmap

### 🤖 Model Training Tab
- Train multiple models (Logistic Regression, Random Forest, SVM)
- Cross-validation scoring
- Performance metrics display

### 📈 Model Comparison Tab
- Side-by-side model performance comparison
- Confusion matrices
- Detailed results table

### 🔮 Predictions Tab
- Interactive prediction interface
- Real-time model predictions
- Probability distributions

### 💾 Model Management Tab
- Save trained models
- Load existing models
- Model metadata display

### 📊 Advanced Analytics Page
- PCA analysis and visualization
- t-SNE dimensionality reduction
- K-means clustering
- Feature importance analysis
- Distribution analysis

## Usage Examples

### Basic Workflow
1. Select a dataset from the sidebar
2. Explore data in the "Data Exploration" tab
3. Train models in the "Model Training" tab
4. Compare performance in "Model Comparison" tab
5. Make predictions in the "Predictions" tab

### Advanced Analysis
1. Navigate to "Advanced Analytics" page
2. Run PCA or t-SNE for dimensionality reduction
3. Perform clustering analysis
4. Analyze feature importance

## API Integration

The dashboard can be extended to integrate with:
- REST APIs for real-time data
- Cloud storage for model persistence
- External ML services
- Database connections

## Deployment Options

### Streamlit Cloud
```bash
# Push to GitHub repository
# Connect to Streamlit Cloud
# Deploy automatically
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t streamlit-labs .
docker run -p 8501:8501 streamlit-labs
```

### Cloud Platforms
- Google Cloud Run
- AWS App Runner
- Azure Container Instances
- Heroku

## Project Structure

```
Streamlit_Labs/
├── app.py                    # Main Streamlit application
├── pages/
│   └── advanced_analytics.py # Advanced analytics page
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── models/                  # Saved models directory
└── README.md
```

## Configuration

### Streamlit Config
The app uses a custom theme defined in `.streamlit/config.toml`:
- Primary color: Blue (#1f77b4)
- Clean, modern interface
- Responsive layout

### Performance
- Caching enabled for data loading and model training
- Optimized for datasets up to 10,000 samples
- Memory-efficient visualization

## Reference

- Source: https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs
- Streamlit Documentation: https://docs.streamlit.io/
- Plotly Documentation: https://plotly.com/python/
