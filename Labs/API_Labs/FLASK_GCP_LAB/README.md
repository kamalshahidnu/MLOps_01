# Flask GCP Lab

This lab demonstrates deploying a Flask ML application to Google Cloud Platform with a web interface for iris flower classification.

## Features

- **Flask Web App**: Interactive web interface for iris classification
- **ML Model**: Logistic Regression trained on iris dataset
- **Model Retraining**: API endpoint to retrain the model
- **GCP Deployment**: Ready for Google App Engine and Cloud Run
- **Docker Support**: Containerized deployment option

## Uniqueness vs Reference Lab

- **Web Interface**: Interactive HTML form instead of just API endpoints
- **Model Retraining**: Live retraining capability via web interface
- **Enhanced UI**: Modern, responsive design with real-time feedback
- **GCP Integration**: Specific configuration for Google Cloud Platform

## Setup

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Visit: http://localhost:8080

### Docker

```bash
# Build image
docker build -t flask-gcp-lab .

# Run container
docker run -p 8080:8080 flask-gcp-lab
```

## GCP Deployment

### Option 1: Google App Engine

```bash
# Install Google Cloud SDK
# Configure authentication
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy to App Engine
gcloud app deploy app.yaml
```

### Option 2: Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/flask-gcp-lab

# Deploy to Cloud Run
gcloud run deploy flask-gcp-lab \
  --image gcr.io/YOUR_PROJECT_ID/flask-gcp-lab \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## API Endpoints

- `GET /` - Web interface
- `GET /api/health` - Health check
- `POST /api/predict` - Make prediction
- `POST /api/retrain` - Retrain model

### Example API Usage

```bash
# Health check
curl http://localhost:8080/api/health

# Make prediction
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Retrain model
curl -X POST http://localhost:8080/api/retrain
```

## Testing

```bash
# Run tests
pytest tests/test_app.py -v

# Run with coverage
pytest tests/test_app.py --cov=app --cov-report=html
```

## Project Structure

```
FLASK_GCP_LAB/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── app.yaml              # Google App Engine config
├── Dockerfile            # Container configuration
├── templates/
│   └── index.html        # Web interface
├── models/               # Saved ML models
├── tests/
│   └── test_app.py       # Test suite
└── README.md
```

## Reference

- Source: https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs
- Flask Documentation: https://flask.palletsprojects.com/
- Google Cloud Platform: https://cloud.google.com/
