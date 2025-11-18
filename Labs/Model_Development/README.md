# Model Development Lab - Heart Disease Prediction

This lab demonstrates a comprehensive **Model Development** workflow for binary classification, featuring Heart Disease prediction using XGBoost and an enhanced interactive dashboard.

## 🎯 Project Overview

This project replicates and enhances the Model Development lab from the reference repository with the following modifications:

- **Dataset**: Heart Disease dataset (UCI-inspired, binary classification)
- **Model**: XGBoost Classifier (instead of basic models)
- **Dashboard**: Enhanced Streamlit dashboard with model comparison, hyperparameter tuning visualization, and experiment tracking
- **Features**: Interactive model training, real-time comparison, advanced analytics

## 📁 Project Structure

```
Model_Development/
├── src/
│   ├── data_loader.py      # Heart Disease data loading and preprocessing
│   ├── model.py            # XGBoost model implementation
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── models/                 # Trained models, scalers, and metrics
├── data/                   # Dataset storage
├── assets/                 # Additional assets
├── app.py                  # Enhanced Streamlit dashboard
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Navigate to the project directory**:
   ```bash
   cd Labs/Model_Development
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

Train the XGBoost model:

```bash
python src/train.py
```

This will:
- Load and preprocess the Heart Disease dataset
- Train an XGBoost classifier
- Evaluate the model and display metrics
- Save the model, scaler, feature importance, and metrics

### Running the Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Visit: http://localhost:8501

## 📊 Dashboard Features

### 🏠 Home
- Project overview and dataset information
- Key statistics and metrics
- Lab modifications summary

### 📊 Data Exploration
- Dataset overview and statistics
- Feature distributions by target class
- Correlation heatmap
- Target class distribution

### 🤖 Model Training
- Interactive XGBoost model training
- Customizable hyperparameters (n_estimators, max_depth, learning_rate)
- Real-time performance metrics
- Confusion matrix visualization

### 📈 Model Comparison
- Compare multiple ML algorithms:
  - XGBoost
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
- Side-by-side performance metrics
- ROC curves comparison
- Interactive visualizations

### ⚙️ Hyperparameter Tuning
- 3D surface plot of hyperparameter space
- Heatmap visualization of grid search results
- Best parameter identification
- Learning rate vs Max depth analysis

### 🔮 Predictions
- Interactive prediction interface
- Real-time heart disease risk assessment
- Probability gauge visualization
- Patient information input form

### 📉 Model Performance
- Saved model metrics display
- Feature importance analysis
- Sample predictions on test set
- Performance visualization

## 🔬 Dataset Information

### Heart Disease Dataset

- **Problem Type**: Binary Classification
- **Target**: Heart Disease Presence (0 = No Disease, 1 = Disease)
- **Features**: 13 clinical and demographic features
  - `age`: Age in years
  - `sex`: Sex (0 = Female, 1 = Male)
  - `cp`: Chest pain type (0-3)
  - `trestbps`: Resting blood pressure
  - `chol`: Serum cholesterol in mg/dl
  - `fbs`: Fasting blood sugar > 120 mg/dl
  - `restecg`: Resting electrocardiographic results
  - `thalach`: Maximum heart rate achieved
  - `exang`: Exercise induced angina
  - `oldpeak`: ST depression induced by exercise
  - `slope`: Slope of peak exercise ST segment
  - `ca`: Number of major vessels
  - `thal`: Thalassemia

## 🤖 Model Details

### XGBoost Classifier

- **Algorithm**: Gradient Boosting Decision Trees
- **Default Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `random_state`: 42

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

## 📈 Usage Examples

### Basic Workflow

1. **Explore the data**:
   - Navigate to "Data Exploration" tab
   - Analyze feature distributions and correlations

2. **Train a model**:
   - Go to "Model Training" tab
   - Adjust hyperparameters
   - Click "Train Model"

3. **Compare models**:
   - Visit "Model Comparison" tab
   - Click "Compare Models" to see side-by-side performance

4. **Make predictions**:
   - Use "Predictions" tab
   - Enter patient information
   - Get real-time risk assessment

### Advanced Analysis

1. **Hyperparameter Tuning**:
   - Navigate to "Hyperparameter Tuning" tab
   - Explore 3D surface and heatmap visualizations
   - Identify optimal parameters

2. **Performance Analysis**:
   - Check "Model Performance" tab
   - Review feature importance
   - Analyze sample predictions

## 🛠️ Development

### Project Structure Details

- **`src/data_loader.py`**: Handles data loading, preprocessing, and scaling
- **`src/model.py`**: XGBoost model class with training and evaluation methods
- **`src/train.py`**: Main training script with metrics logging
- **`src/predict.py`**: Prediction interface for single and batch predictions
- **`app.py`**: Comprehensive Streamlit dashboard

### Adding New Models

To add a new model for comparison:

1. Import the model in `app.py`
2. Add it to the `train_comparison_models()` function
3. The dashboard will automatically include it in comparisons

### Customizing Hyperparameters

Modify hyperparameters in:
- Training script: `src/train.py`
- Dashboard: `app.py` (Model Training tab)

## 📦 Dependencies

- `streamlit`: Interactive dashboard framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning utilities
- `plotly`: Interactive visualizations
- `xgboost`: Gradient boosting classifier
- `joblib`: Model serialization

## 🚢 Deployment

### Local Development

```bash
streamlit run app.py
```

### Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t heart-disease-app .
docker run -p 8501:8501 heart-disease-app
```

### Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

## 📝 Notes

- The dataset is generated synthetically based on UCI Heart Disease characteristics
- For production use, replace with actual dataset loading
- Model performance may vary with different random seeds
- All visualizations are interactive using Plotly

## 🔗 References

- Original Lab: https://github.com/raminmohammadi/MLOps/tree/main/Labs/Model_Development
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Streamlit Documentation: https://docs.streamlit.io/
- UCI Heart Disease Dataset: https://archive.ics.uci.edu/ml/datasets/heart+disease

## 📄 License

This lab is part of the MLOps course materials and is intended for educational purposes.

