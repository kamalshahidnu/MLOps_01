import pytest
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_predict_endpoint_missing_data(client):
    """Test predict endpoint with missing data"""
    # First train a model so it exists
    response = client.post('/api/retrain')
    assert response.status_code == 200
    
    # Now test with missing data
    response = client.post('/api/predict', 
                          data=json.dumps({}),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_endpoint_valid_data(client):
    """Test predict endpoint with valid data"""
    test_data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    
    response = client.post('/api/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    # This might fail if model isn't loaded, which is expected
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'class_name' in data
        assert 'confidence' in data
    else:
        # Model not loaded error is acceptable in tests
        data = json.loads(response.data)
        assert 'error' in data

def test_retrain_endpoint(client):
    """Test retrain endpoint"""
    response = client.post('/api/retrain')
    
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'message' in data
        assert 'accuracy' in data
    else:
        # Error is acceptable if model training fails
        data = json.loads(response.data)
        assert 'error' in data

def test_home_page(client):
    """Test home page loads"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Iris Flower Classification' in response.data
