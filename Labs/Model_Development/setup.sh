#!/bin/bash

# Setup script for Model Development Lab

echo "=========================================="
echo "Model Development Lab - Setup"
echo "=========================================="

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p assets

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Train the model: python src/train.py"
echo "3. Run the dashboard: streamlit run app.py"
echo ""

