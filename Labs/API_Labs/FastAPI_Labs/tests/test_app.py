import pytest
from httpx import AsyncClient
from fastapi import status
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_predict_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/predict", json={
            "petal_length": 1.4,
            "sepal_length": 5.1,
            "petal_width": 0.2,
            "sepal_width": 3.5
        })
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert "response" in data
    assert "class_name" in data
    assert "proba" in data
