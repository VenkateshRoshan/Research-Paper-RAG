# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json

from src.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_rag_model():
    with patch('src.api.main.RAGModel') as mock:
        mock_instance = Mock()
        mock_instance.generate_response.return_value = {
            "query": "test query",
            "response": "test response",
            "references": [],
            "metadata": {},
            "timestamp": "2024-01-01T00:00:00"
        }
        mock.return_value = mock_instance
        yield mock_instance

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint(mock_rag_model):
    test_query = {
        "query": "test query",
        "max_tokens": 150,
        "num_papers": 3
    }
    
    response = client.post("/query", json=test_query)
    assert response.status_code == 200
    assert "query" in response.json()
    assert "response" in response.json()
    assert "references" in response.json()

def test_invalid_query():
    test_query = {
        "query": "",  # Invalid empty query
        "max_tokens": 150,
        "num_papers": 3
    }
    
    response = client.post("/query", json=test_query)
    assert response.status_code == 422  # Validation error