#!/bin/bash

# Script to stop Logstash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/logs/logstash.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping Logstash (PID: $PID)..."
    kill $PID 2>/dev/null || echo "Process not found or already stopped"
    rm "$PID_FILE"
    echo "Logstash stopped"
else
    echo "Logstash PID file not found. Trying to find and kill process..."
    pkill -f "logstash.*logstash.conf" && echo "Logstash stopped" || echo "No Logstash process found"
fi

