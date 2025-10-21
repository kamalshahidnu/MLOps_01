# Boston Housing Price Prediction - Docker Lab

This lab demonstrates containerized machine learning with Docker, featuring a Ridge Regression model for Boston Housing price prediction and an interactive Streamlit dashboard.

## 🏠 Project Overview

This project replicates and enhances the original Docker Lab 1 with the following modifications:
- **Dataset**: Boston Housing dataset (regression problem)
- **Model**: Ridge Regression with regularization
- **Dashboard**: Interactive Streamlit dashboard with data visualization and prediction interface

## 📁 Project Structure

```
Lab1/
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py            # Ridge regression model implementation
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── models/                 # Trained models and scalers
├── data/                   # Dataset storage
├── assets/                 # Additional assets
├── app.py                  # Streamlit dashboard
├── Dockerfile              # Docker configuration
├── docker-compose.yaml     # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Git (for cloning)

### Option 1: Using Docker Compose (Recommended)

1. **Clone and navigate to the project**:
   ```bash
   cd /Users/shahidkamal/Documents/MLOps_01/Labs/Docker_Labs/Lab1
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the dashboard**:
   Open your browser and go to `http://localhost:8501`

### Option 2: Using Docker directly

1. **Build the Docker image**:
   ```bash
   docker build -t boston-housing-app .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 boston-housing-app
   ```

3. **Access the dashboard**:
   Open your browser and go to `http://localhost:8501`

## 🎯 Features

### Dashboard Pages

1. **🏠 Home**: Overview and dataset statistics
2. **📊 Data Exploration**: 
   - Interactive data visualization
   - Feature distribution plots
   - Correlation heatmap
   - Price distribution analysis
3. **🤖 Model Prediction**: 
   - Interactive form for house features
   - Real-time price prediction
   - Prediction confidence metrics
4. **📈 Model Performance**: 
   - Feature importance visualization
   - Model evaluation metrics
   - Sample predictions vs actual

### Model Features

- **Ridge Regression**: Regularized linear regression for price prediction
- **Feature Scaling**: StandardScaler for consistent feature scaling
- **Model Persistence**: Save/load trained models and scalers
- **Feature Importance**: Coefficient analysis for model interpretability

## 🔧 Development

### Training the Model

To train the model locally (outside Docker):

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Make predictions
python src/predict.py
```

### Model Files

After training, the following files will be created in the `models/` directory:
- `boston_housing_model.pkl`: Trained Ridge regression model
- `scaler.pkl`: Feature scaler for preprocessing
- `feature_importance.csv`: Feature importance analysis

## 📊 Dataset Information

The Boston Housing dataset contains 506 samples with 13 features:

- **CRIM**: Crime rate per capita
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
- **LSTAT**: % lower status of the population

**Target**: MEDV - Median value of owner-occupied homes in $1000's

## 🐳 Docker Configuration

### Dockerfile Features
- Python 3.9 slim base image
- Optimized layer caching
- Health checks for container monitoring
- Proper port exposure for Streamlit

### Docker Compose Features
- Volume mounting for model persistence
- Environment variable configuration
- Health check configuration
- Automatic restart policy

## 🧪 Testing

### Health Check
The application includes a health check endpoint at `/_stcore/health`

### Manual Testing
1. Access the dashboard at `http://localhost:8501`
2. Navigate through all pages
3. Test the prediction interface with sample data
4. Verify model performance metrics

## 📈 Performance Metrics

The Ridge Regression model typically achieves:
- **R² Score**: ~0.75-0.85
- **RMSE**: ~4.5-5.5
- **MAE**: ~3.2-3.8

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Change port in docker-compose.yaml or use different port
   docker run -p 8502:8501 boston-housing-app
   ```

2. **Model not found**:
   - Ensure the model is trained first
   - Check that model files exist in the `models/` directory

3. **Memory issues**:
   - Increase Docker memory allocation
   - Use smaller batch sizes for predictions

### Logs
```bash
# View container logs
docker-compose logs -f

# View specific service logs
docker-compose logs boston-housing-app
```

## 🚀 Deployment

### Production Considerations
- Use environment variables for configuration
- Implement proper logging
- Add monitoring and alerting
- Use reverse proxy (nginx) for production
- Implement proper security measures

### Scaling
- Use Docker Swarm or Kubernetes for orchestration
- Implement load balancing for multiple instances
- Use external databases for model storage

## 📚 Learning Objectives

This lab demonstrates:
- Containerized machine learning applications
- Interactive data visualization with Streamlit
- Model training and inference in Docker
- Feature engineering and preprocessing
- Model evaluation and interpretation
- Docker best practices for ML applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the MLOps learning series and is intended for educational purposes.

## 🔗 References

- [Original MLOps Repository](https://github.com/raminmohammadi/MLOps)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
