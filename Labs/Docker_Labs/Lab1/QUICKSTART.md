# 🚀 Quick Start Guide

## Prerequisites
- Docker and Docker Compose installed
- Git (optional, for version control)

## 1. Navigate to the Lab Directory
```bash
cd /Users/shahidkamal/Documents/MLOps_01/Labs/Docker_Labs/Lab1
```

## 2. Run Setup Script (Optional)
```bash
./setup.sh
```

## 3. Start the Application
```bash
docker-compose up --build
```

## 4. Access the Dashboard
Open your browser and go to: **http://localhost:8501**

## 5. Explore the Features

### 🏠 Home Page
- Overview of the dataset
- Basic statistics

### 📊 Data Exploration
- Interactive data visualizations
- Feature distributions
- Correlation analysis

### 🤖 Model Prediction
- Input house features
- Get price predictions
- View prediction confidence

### 📈 Model Performance
- Feature importance
- Model evaluation metrics

## 6. Stop the Application
```bash
docker-compose down
```

## 🔧 Development Mode

### Train Model Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Test predictions
python src/predict.py
```

### Run Tests
```bash
python test_setup.py
```

## 🐳 Docker Commands

### Build Image
```bash
docker build -t boston-housing-app .
```

### Run Container
```bash
docker run -p 8501:8501 boston-housing-app
```

### View Logs
```bash
docker-compose logs -f
```

## 🆘 Troubleshooting

### Port Already in Use
```bash
# Use different port
docker run -p 8502:8501 boston-housing-app
```

### Permission Issues
```bash
chmod +x setup.sh
chmod +x test_setup.py
```

### Clean Docker
```bash
docker-compose down
docker system prune -f
```

## 📚 Next Steps

1. **Modify the Model**: Try different algorithms (SVM, Random Forest, etc.)
2. **Add Features**: Implement feature engineering
3. **Enhance Dashboard**: Add more visualizations
4. **Deploy**: Use cloud platforms (AWS, GCP, Azure)
5. **Monitor**: Add logging and monitoring

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Original MLOps Repository](https://github.com/raminmohammadi/MLOps)
