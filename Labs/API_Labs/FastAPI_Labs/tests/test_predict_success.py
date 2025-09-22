import pytest
from httpx import AsyncClient
from fastapi import status

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.main import app


@pytest.mark.asyncio
async def test_predict_success_flow():
    # Train model first
    import subprocess, sys
    res = subprocess.run([sys.executable, "-m", "src.train"], capture_output=True, text=True)
    assert res.returncode == 0

    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {
            "petal_length": 1.4,
            "sepal_length": 5.1,
            "petal_width": 0.2,
            "sepal_width": 3.5,
        }
        resp = await ac.post("/predict", json=payload)

    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert set(data.keys()) == {"response", "class_name", "proba"}
    assert isinstance(data["response"], int)
    assert isinstance(data["class_name"], str)
    assert 0.0 <= float(data["proba"]) <= 1.0
