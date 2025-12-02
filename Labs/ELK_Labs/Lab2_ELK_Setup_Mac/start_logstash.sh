#!/bin/bash

# Script to start Logstash with the ML model log configuration

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOGSTASH_CONFIG="$SCRIPT_DIR/config/logstash.conf"
LOGSTASH_LOG="$SCRIPT_DIR/logs/logstash.log"

echo "=========================================="
echo "Starting Logstash"
echo "=========================================="
echo ""

# Check if Logstash is installed
if ! command -v logstash &> /dev/null; then
    echo "Error: Logstash is not installed"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Check if Elasticsearch is running
if ! curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo "Error: Elasticsearch is not running"
    echo "Please start Elasticsearch first:"
    echo "  brew services start elasticsearch"
    exit 1
fi

echo "✓ Elasticsearch is running"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Get absolute path for log file
ABS_LOG_PATH="$SCRIPT_DIR/logs/ml_model.log"

# Generate logstash config with correct path if template exists
TEMPLATE_CONFIG="$SCRIPT_DIR/config/logstash.conf.template"
if [ -f "$TEMPLATE_CONFIG" ]; then
    sed "s|{{LOG_PATH}}|$ABS_LOG_PATH|g" "$TEMPLATE_CONFIG" > "$LOGSTASH_CONFIG"
    echo "✓ Generated Logstash config with path: $ABS_LOG_PATH"
fi

echo "Configuration: $LOGSTASH_CONFIG"
echo "Log file: $ABS_LOG_PATH"
echo ""
echo "Starting Logstash..."
echo "Press Ctrl+C to stop"
echo ""

# Start Logstash
logstash -f "$LOGSTASH_CONFIG" > "$LOGSTASH_LOG" 2>&1 &

LOGSTASH_PID=$!
echo "Logstash started with PID: $LOGSTASH_PID"
echo "Logs are being written to: $LOGSTASH_LOG"
echo ""
echo "To stop Logstash, run: kill $LOGSTASH_PID"
echo "Or use: ./stop_logstash.sh"

# Save PID for stopping later
echo $LOGSTASH_PID > "$SCRIPT_DIR/logs/logstash.pid"

