import json
import pytest
from httpx import AsyncClient
from fastapi import status
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.main import app


@pytest.mark.asyncio
async def test_predict_requires_model(tmp_path, monkeypatch):
    # Mock the model path to be missing
    import os
    original_path = "model/iris_model.pkl"
    monkeypatch.setattr("src.predict.MODEL_PATH", tmp_path / "missing_model.pkl")
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/predict", json={
            "petal_length": 1.4,
            "sepal_length": 5.1,
            "petal_width": 0.2,
            "sepal_width": 3.5
        })
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    body = resp.json()
    assert "Model not found" in body["detail"]
