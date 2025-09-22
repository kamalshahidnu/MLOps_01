#!/bin/bash
set -e

# Deployment script for FastAPI Labs
# Usage: ./deploy.sh [environment]

ENVIRONMENT=${1:-staging}
IMAGE_TAG=${2:-latest}

echo "Deploying FastAPI Labs to $ENVIRONMENT..."

# Build image
echo "Building Docker image..."
docker build -t fastapi-labs:$IMAGE_TAG .

# Run container
echo "Starting container..."
docker run -d \
  --name fastapi-labs-$ENVIRONMENT \
  -p 8000:8000 \
  --restart unless-stopped \
  fastapi-labs:$IMAGE_TAG \
  bash -lc "python -m src.train && uvicorn app.main:app --host 0.0.0.0 --port 8000"

# Health check
echo "Waiting for service to start..."
sleep 10

if curl -f http://localhost:8000/health; then
    echo "✅ Deployment successful!"
    echo "API available at: http://localhost:8000"
    echo "Docs available at: http://localhost:8000/docs"
else
    echo "❌ Deployment failed - health check failed"
    docker logs fastapi-labs-$ENVIRONMENT
    exit 1
fi
