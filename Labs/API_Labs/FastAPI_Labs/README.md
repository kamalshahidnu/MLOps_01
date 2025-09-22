# FastAPI Labs

This section introduces a minimal FastAPI app with a health check and a prediction endpoint. Uniqueness vs. the reference lab: we train a Logistic Regression model (instead of Decision Tree) and return the predicted class id, class name, and probability.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
python -m src.train
uvicorn app.main:app --reload
```

## Docker

Build image:

```bash
docker build -t fastapi-labs:latest .
```

Run container (train inside container, then serve):

```bash
docker run --rm -it -p 8000:8000 fastapi-labs:latest bash -lc "python -m src.train && uvicorn app.main:app --host 0.0.0.0 --port 8000"
```

- Local: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs

## Quick checks

```bash
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"petal_length": 1.4, "sepal_length": 5.1, "petal_width": 0.2, "sepal_width": 3.5}'
```

Example response:

```json
{
  "response": 0,
  "class_name": "setosa",
  "proba": 0.98
}
```

## Tests

```bash
pytest -q
```

## CI/CD and Deployment

### GitHub Actions
The repository includes a CI/CD pipeline that:
- Runs tests on Python 3.9, 3.10, 3.11
- Builds Docker image
- Runs container tests
- Deploys to staging (on main branch)

### Manual Deployment
```bash
# Deploy to staging
./deploy.sh staging

# Deploy with custom tag
./deploy.sh production v1.0.0
```

### Monitoring
Enhanced health endpoints:
- `/health` - Basic health check
- `/health/detailed` - System metrics (CPU, memory, disk)
- `/health/ready` - Readiness check (model loaded)

## Reference

- Source: https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs
- Blog walkthrough: https://www.mlwithramin.com/blog/fastapi-lab1
