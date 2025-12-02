# ELK Stack Setup for Mac - Wine Quality ML Model Monitoring

This lab demonstrates how to set up the ELK (Elasticsearch, Logstash, Kibana) stack on macOS to monitor and visualize machine learning model predictions, training metrics, and system events.

## 🍷 Project Overview

This project replicates and enhances the original ELK Setup lab with the following modifications:
- **Dataset**: Wine Quality dataset (white wine)
- **Model**: Random Forest Classifier (instead of simpler models)
- **Purpose**: Monitor ML model predictions, training metrics, and errors using ELK stack
- **Platform**: macOS with Homebrew installation

## 📁 Project Structure

```
Lab2_ELK_Setup_Mac/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py            # Random Forest model implementation
│   ├── logger.py           # ELK-compatible JSON logger
│   ├── train.py            # Training script with logging
│   └── predict.py          # Prediction script with logging
├── config/
│   └── logstash.conf       # Logstash configuration
├── models/                 # Trained models and scalers
├── logs/                   # ML model logs (JSON format)
├── setup.sh                # ELK stack installation script
├── start_logstash.sh       # Start Logstash script
├── stop_logstash.sh        # Stop Logstash script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- macOS
- Homebrew (if not installed: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- Python 3.8+

### Step 1: Setup ELK Stack

Run the setup script to install Elasticsearch, Logstash, and Kibana:

```bash
cd /Users/shahidkamal/Documents/MLOps_01/Labs/ELK_Labs/Lab2_ELK_Setup_Mac
chmod +x setup.sh start_logstash.sh stop_logstash.sh
./setup.sh
```

This script will:
- Install Java (if needed)
- Install Elasticsearch, Logstash, and Kibana via Homebrew
- Start Elasticsearch and Kibana services
- Create necessary directories

**Service URLs:**
- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Train the Model

Train the Wine Quality model (logs will be written to `logs/ml_model.log`):

```bash
python -m src.train
```

This will:
- Download/prepare the Wine Quality dataset
- Train a Random Forest classifier
- Log training metrics to JSON log file
- Save the trained model to `models/wine_quality_model.pkl`

### Step 4: Start Logstash

Start Logstash to process logs and send them to Elasticsearch:

```bash
./start_logstash.sh
```

Logstash will:
- Read JSON logs from `logs/ml_model.log`
- Parse and enrich the log data
- Send logs to Elasticsearch

### Step 5: Make Predictions

Make predictions (these will be logged to ELK):

```bash
python -m src.predict
```

This will:
- Load the trained model
- Make sample predictions
- Log all predictions to the JSON log file

### Step 6: View in Kibana

1. Open Kibana: http://localhost:5601
2. Create an index pattern:
   - Go to **Stack Management** → **Index Patterns** → **Create index pattern**
   - Pattern: `ml-model-logs-*`
   - Time field: `@timestamp`
   - Click **Create index pattern**

3. View logs in **Discover**:
   - Go to **Discover** in the left menu
   - Select the `ml-model-logs-*` index pattern
   - View and filter log events

4. Create visualizations (optional):
   - Go to **Visualize Library** → **Create visualization**
   - Create charts for:
     - Prediction distribution
     - Model accuracy over time
     - Error rate
     - Training metrics

## 📊 Log Event Types

The logger creates structured JSON logs with the following event types:

### 1. Prediction Events
```json
{
  "event_type": "prediction",
  "timestamp": "2025-01-09T10:30:00Z",
  "model_name": "wine_quality_rf",
  "prediction_id": "sample_0",
  "features": {...},
  "prediction": 1,
  "probability": 0.85
}
```

### 2. Training Events
```json
{
  "event_type": "training",
  "timestamp": "2025-01-09T10:00:00Z",
  "model_name": "wine_quality_rf",
  "metrics": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91
  },
  "training_time_seconds": 5.23,
  "dataset_size": 3918
}
```

### 3. Evaluation Events
```json
{
  "event_type": "evaluation",
  "timestamp": "2025-01-09T10:05:00Z",
  "model_name": "wine_quality_rf",
  "dataset_type": "test",
  "metrics": {...}
}
```

### 4. Error Events
```json
{
  "event_type": "error",
  "timestamp": "2025-01-09T10:10:00Z",
  "model_name": "wine_quality_rf",
  "error_message": "...",
  "error_type": "ValueError"
}
```

### 5. System Events
```json
{
  "event_type": "system",
  "timestamp": "2025-01-09T10:00:00Z",
  "system_event_type": "training_start",
  "message": "Starting model training"
}
```

## 🔧 Configuration

### Logstash Configuration

The Logstash configuration (`config/logstash.conf`) processes JSON logs and sends them to Elasticsearch. Key features:

- **Input**: Reads from `logs/ml_model.log` as JSON
- **Filter**: 
  - Parses JSON logs
  - Adds tags based on event type
  - Enriches prediction events with labels
- **Output**: Sends to Elasticsearch index `ml-model-logs-YYYY.MM.DD`

To modify the log path, update the `path` field in `config/logstash.conf` before starting Logstash.

## 🎯 Kibana Dashboard Setup

### Creating Visualizations

1. **Prediction Distribution**:
   - Type: Pie Chart
   - Field: `prediction_label.keyword`
   - Filter: `event_type:prediction`

2. **Accuracy Over Time**:
   - Type: Line Chart
   - Y-axis: Average of `metrics.accuracy`
   - X-axis: `@timestamp`
   - Filter: `event_type:evaluation`

3. **Prediction Probability Distribution**:
   - Type: Histogram
   - Field: `probability`
   - Filter: `event_type:prediction`

4. **Error Rate**:
   - Type: Metric
   - Field: `event_type.keyword`
   - Filter: `event_type:error`

### Creating a Dashboard

1. Go to **Dashboard** → **Create dashboard**
2. Add your visualizations
3. Save the dashboard

## 🛠️ Troubleshooting

### Elasticsearch not starting
```bash
# Check if port 9200 is in use
lsof -i :9200

# Check Elasticsearch logs
tail -f /opt/homebrew/var/log/elasticsearch.log
```

### Logstash not processing logs
```bash
# Check Logstash logs
tail -f logs/logstash.log

# Verify log file exists and has content
cat logs/ml_model.log | head -5
```

### Kibana not connecting to Elasticsearch
```bash
# Verify Elasticsearch is running
curl http://localhost:9200

# Check Kibana logs
tail -f /opt/homebrew/var/log/kibana.log
```

### No logs appearing in Kibana
1. Verify index pattern exists: `ml-model-logs-*`
2. Check time range in Discover (use "Last 24 hours" or "All time")
3. Verify logs are being written: `tail -f logs/ml_model.log`

## 📈 Advanced Usage

### Real-time Monitoring

1. Keep Logstash running in the background
2. Continuously make predictions: `python -m src.predict`
3. View logs in Kibana Discover in real-time

### Custom Predictions

Modify `src/predict.py` to make custom predictions:

```python
custom_wine = {
    'fixed acidity': 7.5,
    'volatile acidity': 0.3,
    # ... other features
}
```

### Batch Prediction Logging

Process a CSV file and log all predictions:

```python
# Add to predict.py
df = pd.read_csv('wine_data.csv')
for idx, row in df.iterrows():
    prediction, probability = predict_single(...)
    elk_logger.log_prediction(...)
```

## 🛑 Stopping Services

```bash
# Stop Logstash
./stop_logstash.sh

# Stop Elasticsearch
brew services stop elasticsearch

# Stop Kibana
brew services stop kibana
```

## 📝 Differences from Reference Lab

- **Model**: Uses Random Forest Classifier (more robust than simpler models)
- **Dataset**: White Wine Quality dataset (classification task)
- **Logging**: Comprehensive JSON logging with multiple event types
- **Features**: Feature importance tracking, batch prediction logging
- **Documentation**: Enhanced README with troubleshooting and Kibana setup

## 📚 Reference

- Source: https://github.com/raminmohammadi/MLOps/tree/main/Labs/ELK_Labs/Lab2_ELK_Setup_Mac
- Elasticsearch Docs: https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html
- Logstash Docs: https://www.elastic.co/guide/en/logstash/current/index.html
- Kibana Docs: https://www.elastic.co/guide/en/kibana/current/index.html

## 🎓 Learning Objectives

After completing this lab, you will:
- Understand how to set up ELK stack on macOS
- Learn to log ML model events in structured JSON format
- Configure Logstash to process and enrich log data
- Visualize ML metrics and predictions in Kibana
- Monitor model performance in real-time

## 🤝 Contributing

Feel free to enhance this lab by:
- Adding more visualization dashboards
- Implementing alerting for model degradation
- Adding more sophisticated log enrichment
- Integrating with model versioning

