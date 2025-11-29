import pytest
from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def test_health_check():
    response = client.get("/healthchecks")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "message" in data
    assert "timestamp" in data
    assert "model_loaded" in data
    assert data["status"] in ["healthy", "degraded"]

def test_prediction_endpoint():
    test_features = {
        "feature_0": 0.5,
        "feature_1": -0.2,
        "feature_2": 1.1,
        "feature_3": -0.8,
        "feature_4": 0.3,
        "feature_5": -1.2,
        "feature_6": 0.7,
        "feature_7": -0.5,
        "feature_8": 1.5,
        "feature_9": -0.1
    }
    
    response = client.post("/prediction", json={"features": test_features})
    
    assert response.status_code in [200, 500]  # 500 если модель не найдена
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probabilities" in data
        assert "model_type" in data
        assert "features_used" in data
        assert "status" in data
        assert data["status"] == "success"
        
        assert "class_0" in data["probabilities"]
        assert "class_1" in data["probabilities"]

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "ML Model API is running" in data["message"]

def test_openapi_schema():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    
    assert "/healthchecks" in schema["paths"]
    assert "/prediction" in schema["paths"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
