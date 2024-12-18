# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint():
    """Test query endpoint"""
    test_query = {
        "query": "What is machine learning?",
        "max_tokens": 100,
        "style": "academic"
    }
    
    response = client.post("/query", json=test_query)
    assert response.status_code == 200
    assert "query" in response.json()
    assert "response" in response.json()
    assert "references" in response.json()

def test_invalid_query():
    """Test invalid query handling"""
    test_query = {
        "query": "",  # Empty query
        "max_tokens": 100
    }
    
    response = client.post("/query", json=test_query)
    assert response.status_code == 422  # Validation error

def test_rate_limiting():
    """Test rate limiting"""
    test_query = {"query": "test", "max_tokens": 100}
    
    # Make multiple requests quickly
    responses = [
        client.post("/query", json=test_query)
        for _ in range(5)
    ]
    
    # Check if any request was rate limited
    assert any(r.status_code == 429 for r in responses)