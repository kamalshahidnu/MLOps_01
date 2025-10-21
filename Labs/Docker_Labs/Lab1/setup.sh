#!/bin/bash

# Boston Housing Price Prediction - Setup Script
# This script sets up the Docker environment for the Boston Housing prediction lab

echo "🏠 Boston Housing Price Prediction - Setup Script"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data assets

# Set proper permissions
chmod +x setup.sh

echo "🔧 Building Docker image..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo ""
echo "🚀 Setup completed successfully!"
echo ""
echo "To start the application:"
echo "  docker-compose up"
echo ""
echo "To run in background:"
echo "  docker-compose up -d"
echo ""
echo "To stop the application:"
echo "  docker-compose down"
echo ""
echo "Access the dashboard at: http://localhost:8501"
echo ""
echo "📚 For more information, see README.md"
