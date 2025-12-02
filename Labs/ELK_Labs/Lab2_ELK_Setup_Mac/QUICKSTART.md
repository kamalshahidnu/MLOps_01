# Quick Start Guide

Get up and running with the ELK lab in 5 minutes!

## Prerequisites Check

```bash
# Check if Homebrew is installed
brew --version

# Check if Python 3 is installed
python3 --version
```

## Step-by-Step Setup

### 1. Setup ELK Stack (5 minutes)

```bash
cd /Users/shahidkamal/Documents/MLOps_01/Labs/ELK_Labs/Lab2_ELK_Setup_Mac
chmod +x *.sh
./setup.sh
```

Wait for services to start, then verify:

```bash
# Check Elasticsearch
curl http://localhost:9200

# Check Kibana (open in browser)
open http://localhost:5601
```

### 2. Install Python Dependencies (1 minute)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Train the Model (2 minutes)

```bash
python -m src.train
```

This will:
- Download the Wine Quality dataset
- Train a Random Forest model
- Log training metrics to `logs/ml_model.log`

### 4. Start Logstash (in a new terminal)

```bash
# Make sure you're in the lab directory and virtual env is activated
cd /Users/shahidkamal/Documents/MLOps_01/Labs/ELK_Labs/Lab2_ELK_Setup_Mac
source venv/bin/activate

./start_logstash.sh
```

Keep this terminal open - Logstash needs to keep running.

### 5. Make Predictions (1 minute)

In your original terminal:

```bash
python -m src.predict
```

This will make predictions and log them to ELK.

### 6. View in Kibana

1. Open Kibana: http://localhost:5601
2. Create index pattern:
   - Go to **Stack Management** → **Index Patterns**
   - Create pattern: `ml-model-logs-*`
   - Time field: `@timestamp`
3. View logs:
   - Go to **Discover**
   - Select the `ml-model-logs-*` pattern
   - See your prediction logs!

## Troubleshooting

### Elasticsearch not starting?
```bash
brew services list
brew services restart elasticsearch
```

### Logstash errors?
```bash
# Check Logstash logs
tail -f logs/logstash.log

# Verify log file exists
ls -lh logs/ml_model.log
```

### No logs in Kibana?
1. Check time range (use "Last 24 hours")
2. Verify index pattern is `ml-model-logs-*`
3. Check Elasticsearch has data: `curl 'localhost:9200/ml-model-logs-*/_count'`

## Next Steps

- Create visualizations (see `config/kibana_dashboard_guide.md`)
- Make more predictions
- Explore different queries in Discover
- Set up alerts for errors

## Stop Services

```bash
./stop_logstash.sh
brew services stop elasticsearch
brew services stop kibana
```

## Help

See the main [README.md](README.md) for detailed documentation.

