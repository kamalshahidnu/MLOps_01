# Kibana Dashboard Setup Guide

This guide will help you create visualizations and dashboards in Kibana for monitoring your ML model.

## Step 1: Create Index Pattern

1. Open Kibana: http://localhost:5601
2. Go to **Stack Management** → **Index Patterns** → **Create index pattern**
3. Index pattern name: `ml-model-logs-*`
4. Time field: Select `@timestamp`
5. Click **Create index pattern**

## Step 2: Create Visualizations

### Visualization 1: Prediction Distribution

**Type**: Pie Chart

1. Go to **Visualize Library** → **Create visualization** → **Pie**
2. Select index pattern: `ml-model-logs-*`
3. Buckets:
   - Add bucket: **Slice by**
   - Aggregation: **Terms**
   - Field: `prediction_label.keyword`
   - Size: 10
4. Filter: `event_type.keyword: prediction`
5. Save as: "Prediction Distribution"

### Visualization 2: Model Accuracy Over Time

**Type**: Line Chart

1. Go to **Visualize Library** → **Create visualization** → **Line**
2. Select index pattern: `ml-model-logs-*`
3. Metrics:
   - Y-axis: **Average** of `metrics.accuracy`
4. Buckets:
   - X-axis: **Date Histogram**
   - Field: `@timestamp`
   - Interval: **Auto**
5. Filter: `event_type.keyword: evaluation`
6. Save as: "Model Accuracy Over Time"

### Visualization 3: Prediction Probability Distribution

**Type**: Histogram

1. Go to **Visualize Library** → **Create visualization** → **Vertical Bar**
2. Select index pattern: `ml-model-logs-*`
3. Metrics:
   - Y-axis: **Count**
4. Buckets:
   - X-axis: **Histogram**
   - Field: `probability`
   - Interval: 0.1
5. Filter: `event_type.keyword: prediction`
6. Save as: "Prediction Probability Distribution"

### Visualization 4: Error Count

**Type**: Metric

1. Go to **Visualize Library** → **Create visualization** → **Metric**
2. Select index pattern: `ml-model-logs-*`
3. Metrics:
   - Metric: **Count**
4. Filter: `event_type.keyword: error`
5. Save as: "Error Count"

### Visualization 5: Training Metrics Summary

**Type**: Data Table

1. Go to **Visualize Library** → **Create visualization** → **Data Table**
2. Select index pattern: `ml-model-logs-*`
3. Metrics:
   - Metric: **Count**
4. Buckets:
   - Split rows
   - Aggregation: **Top Hits**
   - Field: `metrics.accuracy`
   - Sort by: `@timestamp` Descending
   - Size: 10
5. Filter: `event_type.keyword: evaluation`
6. Save as: "Latest Training Metrics"

## Step 3: Create Dashboard

1. Go to **Dashboard** → **Create dashboard**
2. Click **Add** → **Add an existing visualization**
3. Add all the visualizations you created:
   - Prediction Distribution
   - Model Accuracy Over Time
   - Prediction Probability Distribution
   - Error Count
   - Latest Training Metrics
4. Arrange them in a grid layout
5. Save dashboard as: "ML Model Monitoring Dashboard"

## Step 4: Set Auto-refresh (Optional)

1. In your dashboard, click the time picker (top right)
2. Set time range (e.g., "Last 24 hours")
3. Click **Auto refresh** → Select interval (e.g., 30 seconds)
4. This will automatically refresh the dashboard for real-time monitoring

## Step 5: Create Alerts (Optional)

1. Go to **Stack Management** → **Rules and Connectors** → **Create rule**
2. Rule type: **Threshold**
3. Index: `ml-model-logs-*`
4. Condition: When `metrics.accuracy` is below 0.8
5. Action: Email/Slack notification
6. Save the alert rule

## Sample Queries in Discover

### View all predictions
```
event_type:prediction
```

### View predictions with high probability
```
event_type:prediction AND probability:>0.9
```

### View errors
```
event_type:error
```

### View recent training metrics
```
event_type:evaluation
```

### Filter by time range
```
@timestamp:[now-1h TO now]
```

### Combine filters
```
event_type:prediction AND prediction_label:"Good Quality"
```

## Tips

- Use the time picker to focus on specific time ranges
- Save frequently used searches as saved searches
- Export visualizations as images or PDFs
- Set up alerts for model degradation
- Create different dashboards for different stakeholders (e.g., data scientists vs. product managers)

