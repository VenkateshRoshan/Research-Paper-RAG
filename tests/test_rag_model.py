import pytest
import asyncio
from unittest.mock import Mock, patch
from src.models.rag_model import RAGModel
from src.data.loader import DataLoader

@pytest.fixture
async def rag_model():
    """
        Fixture for RAG model instance
    """
    model = RAGModel(
        model_name="google/flan-t5-small",  # Use smaller model for testing
        max_length=512,
        quantization=None
    )
    yield model

@pytest.fixture
def sample_papers():
    """
        Fixture for sample papers
    """
    return [
        {
            "title": "Sample Paper 1",
            "authors": ["Author 1", "Author 2"],
            "categories": "cs.CL",
            "abstract": "This is a sample paper abstract.",
            "score": 0.8
        },
        {
            "title": "Sample Paper 2",
            "authors": ["Author 3", "Author 4"],
            "categories": "cs.LG",
            "abstract": "This is another sample paper abstract.",
            "score": 0.6
        }
    ]

@pytest.mark.asyncio
async def test_generate_response(rag_model, sample_papers):
    """Test response generation"""
    with patch.object(DataLoader, 'search', return_value=sample_papers):
        response = await rag_model.generate_response(
            query="What is machine learning?",
            max_new_tokens=100
        )
        
        assert isinstance(response, dict)
        assert "query" in response
        assert "response" in response
        assert "references" in response
        assert len(response["references"]) == len(sample_papers)

@pytest.mark.asyncio
async def test_error_handling(rag_model):
    """Test error handling in response generation"""
    with patch.object(DataLoader, 'search', side_effect=Exception("Test error")):
        response = await rag_model.generate_response(
            query="What is machine learning?",
            max_new_tokens=100
        )
        
        assert "error" in response["metadata"]
        assert response["metadata"]["status"] == "error"

@pytest.mark.asyncio
async def test_empty_query(rag_model):
    """Test handling of empty queries"""
    response = await rag_model.generate_response(
        query="",
        max_new_tokens=100
    )
    
    assert "error" in response["metadata"]

@pytest.mark.asyncio
async def test_long_query(rag_model, sample_papers):
    """Test handling of long queries"""
    with patch.object(DataLoader, 'search', return_value=sample_papers):
        long_query = "test " * 1000
        response = await rag_model.generate_response(
            query=long_query,
            max_new_tokens=100
        )
        
        assert isinstance(response, dict)
        assert "response" in response

# tests/test_data_loader.py
import pytest
from src.data.loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader()

def test_data_loader_initialization(data_loader):
    """Test DataLoader initialization"""
    assert data_loader.embedder is not None
    assert data_loader.cache_dir is not None

@pytest.mark.asyncio
async def test_paper_search(data_loader):
    """Test paper search functionality"""
    papers = await data_loader.search("machine learning", k=2)
    
    assert isinstance(papers, list)
    assert len(papers) <= 2
    for paper in papers:
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
        assert "categories" in paper
        assert "score" in paper

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