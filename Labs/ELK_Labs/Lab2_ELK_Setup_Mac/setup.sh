#!/bin/bash

# ELK Stack Setup Script for Mac
# This script installs and configures Elasticsearch, Logstash, and Kibana

set -e

echo "=========================================="
echo "ELK Stack Setup for Mac"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Error: Homebrew is not installed${NC}"
    echo "Please install Homebrew first:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

echo -e "${GREEN}✓ Homebrew is installed${NC}"

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo -e "${YELLOW}Java is not installed. Installing OpenJDK 17...${NC}"
    brew install openjdk@17
    
    # Add Java to PATH for this session
    export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
    
    # Add to shell profile
    if [ -f ~/.zshrc ]; then
        echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
    elif [ -f ~/.bash_profile ]; then
        echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.bash_profile
    fi
    
    echo -e "${GREEN}✓ Java installed${NC}"
else
    echo -e "${GREEN}✓ Java is already installed${NC}"
    java -version
fi

# Install Elasticsearch
echo ""
echo "Installing Elasticsearch..."
if ! command -v elasticsearch &> /dev/null; then
    brew tap elastic/tap
    brew install elastic/tap/elasticsearch-full
    echo -e "${GREEN}✓ Elasticsearch installed${NC}"
else
    echo -e "${GREEN}✓ Elasticsearch is already installed${NC}"
fi

# Install Logstash
echo ""
echo "Installing Logstash..."
if ! command -v logstash &> /dev/null; then
    brew install elastic/tap/logstash-full
    echo -e "${GREEN}✓ Logstash installed${NC}"
else
    echo -e "${GREEN}✓ Logstash is already installed${NC}"
fi

# Install Kibana
echo ""
echo "Installing Kibana..."
if ! command -v kibana &> /dev/null; then
    brew install elastic/tap/kibana-full
    echo -e "${GREEN}✓ Kibana installed${NC}"
else
    echo -e "${GREEN}✓ Kibana is already installed${NC}"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p logs models config
echo -e "${GREEN}✓ Directories created${NC}"

# Check if services are running
echo ""
echo "Checking ELK services status..."

# Start Elasticsearch
if brew services list | grep -q "elasticsearch.*stopped"; then
    echo "Starting Elasticsearch..."
    brew services start elasticsearch
    echo "Waiting for Elasticsearch to start..."
    sleep 10
    echo -e "${GREEN}✓ Elasticsearch started${NC}"
elif curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Elasticsearch is already running${NC}"
else
    echo "Starting Elasticsearch..."
    brew services start elasticsearch
    sleep 10
    echo -e "${GREEN}✓ Elasticsearch started${NC}"
fi

# Start Kibana
if brew services list | grep -q "kibana.*stopped"; then
    echo "Starting Kibana..."
    brew services start kibana
    sleep 5
    echo -e "${GREEN}✓ Kibana started${NC}"
elif curl -s http://localhost:5601 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Kibana is already running${NC}"
else
    echo "Starting Kibana..."
    brew services start kibana
    sleep 5
    echo -e "${GREEN}✓ Kibana started${NC}"
fi

# Display status
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  Elasticsearch: http://localhost:9200"
echo "  Kibana:        http://localhost:5601"
echo ""
echo "To verify Elasticsearch is running:"
echo "  curl http://localhost:9200"
echo ""
echo "Next steps:"
echo "  1. Install Python dependencies: pip install -r requirements.txt"
echo "  2. Train the model: python -m src.train"
echo "  3. Start Logstash: ./start_logstash.sh"
echo "  4. Make predictions: python -m src.predict"
echo "  5. Open Kibana: http://localhost:5601"
echo ""

